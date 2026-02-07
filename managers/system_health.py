import os
import platform
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Proje içi modüller
from config import Config

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.SystemHealth")

# psutil: CPU, RAM ve Disk takibi için
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("⚠️ psutil modülü eksik. Sistem sağlık verileri kısıtlı.")

# pynvml: NVIDIA GPU donanım seviyesi takibi (Sıcaklık, VRAM, Fan)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.debug("pynvml modülü bulunamadı. GPU izleme devre dışı.")

# torch: AI modellerinin GPU erişimini kontrol etmek için (Opsiyonel)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SystemHealthManager:
    """
    LotusAI Sunucu ve Donanım Sağlık Yöneticisi.
    
    Bu sınıf, sistemin hem genel donanım (CPU/RAM) hem de 
    yapay zeka operasyonları için kritik olan GPU kaynaklarını izler.
    """
    
    def __init__(self, system_state=None):
        self.lock = threading.RLock()
        self.state = system_state # core/system_state.py entegrasyonu
        self.start_time = datetime.now()
        self.last_net_io = None
        
        # GPU Modülü Başlatma
        self.gpu_active = False
        self.gpu_count = 0
        self.cuda_info = "Tespit Edilemedi"
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_active = True
                logger.info(f"🚀 GPU Takip Servisi Aktif: {self.gpu_count} cihaz tespit edildi.")
            except Exception as e:
                self.gpu_active = False
                logger.error(f"❌ NVML Başlatılamadı: {e}")

        # PyTorch/CUDA Yazılım Kontrolü
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.cuda_info = f"Aktif (v{torch.version.cuda})"
            else:
                self.cuda_info = "Pasif (Yazılım desteği yok)"

        if PSUTIL_AVAILABLE:
            self.last_net_io = psutil.net_io_counters()
            logger.info("✅ Sistem sağlık takip servisi hazır.")

    # --- DURUM ÖZETLERİ ---

    def get_status_summary(self) -> str:
        """Sistem durumunun tek satırlık özeti (Sidar Ajanı veya Dashboard için)."""
        if not PSUTIL_AVAILABLE:
            return "⚠️ Sistem izleme modülü pasif (psutil eksik)."

        with self.lock:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                
                # GPU Özeti
                gpu_info = ""
                if self.gpu_active:
                    gpu_load = self._get_gpu_load()
                    gpu_info = f" | GPU: %{gpu_load}"

                status = "SAĞLIKLI 🟢"
                if cpu > 85 or ram > 90: 
                    status = "KRİTİK 🔴"
                    self._trigger_system_warning("Yüksek donanım yükü tespit edildi.")
                elif cpu > 65 or ram > 75: 
                    status = "YORGUN 🟠"
                
                return f"Sistem Durumu: {status} | CPU: %{cpu} | RAM: %{ram}{gpu_info}"
            except Exception as e:
                return f"Özet alınamadı: {str(e)}"

    def get_detailed_report(self) -> str:
        """Tüm donanım bileşenlerini içeren kapsamlı teknik rapor."""
        if not PSUTIL_AVAILABLE:
            return "Sağlık raporu üretilemiyor: 'psutil' kütüphanesi eksik."

        with self.lock:
            try:
                # 1. Temel Kaynaklar
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage(str(Config.WORK_DIR.anchor)).percent
                
                # 2. Ağ ve Uptime
                net_report = self._get_network_speed()
                uptime = self._format_timedelta(datetime.now() - self.start_time)

                # 3. GPU ve AI Donanım Detayları
                gpu_report = ""
                if self.gpu_active:
                    gpu_report = self._get_detailed_gpu_info()

                # 4. Genel Durum Kararı
                status_icon = "🟢"
                warning = ""
                if cpu > 85 or mem.percent > 90:
                    status_icon = "🔴"
                    warning = " (KRİTİK!)"
                elif cpu > 70 or mem.percent > 80:
                    status_icon = "🟠"

                report = [
                    f"🖥️ LOTUSAI SİSTEM SAĞLIK RAPORU {status_icon}{warning}",
                    f"{'='*40}",
                    f"⏱️ Uptime: {uptime}",
                    f"🤖 CUDA Desteği: {self.cuda_info}",
                    f"⚙️ İşlemci (CPU): %{cpu}",
                    f"🧠 Bellek (RAM): %{mem.percent} ({round(mem.used/(1024**3), 2)}/{round(mem.total/(1024**3), 2)} GB)",
                    f"💾 Disk Doluluğu: %{disk}",
                    f"🌐 Ağ Trafiği: {net_report}",
                    f"📑 Toplam Süreç: {len(psutil.pids())}"
                ]

                if gpu_report:
                    report.append(f"{'-'*40}\n🎮 GPU DURUMU (NVIDIA):\n{gpu_report}")

                # Kaynak tüketen süreç tespiti
                if cpu > 70 or mem.percent > 80:
                    top_cpu = self._get_top_resource_process("cpu")
                    top_ram = self._get_top_resource_process("ram")
                    report.append(f"{'-'*40}")
                    if top_cpu: report.append(f"🔥 En Yoğun CPU: {top_cpu}")
                    if top_ram: report.append(f"📦 En Yoğun RAM: {top_ram}")

                return "\n".join(report)

            except Exception as e:
                logger.error(f"Rapor oluşturma hatası: {e}")
                return f"Hata: Sistem verileri okunamadı."

    # --- GPU YARDIMCILARI ---

    def _get_gpu_load(self) -> int:
        """Birinci GPU'nun yük yüzdesini döner."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except: return 0

    def _get_detailed_gpu_info(self) -> str:
        """Tüm GPU'ların sıcaklık, yük, VRAM ve süreç bilgilerini döner."""
        lines = []
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                vram_use = round(mem.used / (1024**2), 0)
                vram_total = round(mem.total / (1024**2), 0)
                
                # GPU üzerinde çalışan süreçleri bulalım
                gpu_procs = self._get_gpu_processes(handle)
                proc_info = f" | Süreçler: {gpu_procs}" if gpu_procs else ""
                
                lines.append(f"- GPU {i} [{name}]: %{util.gpu} Yük | {temp}°C | VRAM: {int(vram_use)}/{int(vram_total)} MB{proc_info}")
        except Exception as e:
            return f"GPU verisi çekilemedi: {e}"
        return "\n".join(lines)

    def _get_gpu_processes(self, handle) -> str:
        """Belirli bir GPU üzerinde çalışan aktif işlemlerin sayısını ve VRAM tüketimini bulur."""
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                return ""
            return f"{len(procs)} Aktif İşlem"
        except:
            return ""

    # --- TEKNİK YARDIMCILAR ---

    def _get_network_speed(self) -> str:
        """Ağ trafiğindeki anlık değişimi hesaplar (KB/s)."""
        try:
            current_net_io = psutil.net_io_counters()
            if not self.last_net_io:
                self.last_net_io = current_net_io
                return "Hesaplanıyor..."
            
            sent = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / 1024
            recv = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / 1024
            
            self.last_net_io = current_net_io
            return f"↑ {round(sent, 1)} KB/s | ↓ {round(recv, 1)} KB/s"
        except: return "Veri yok"

    def _get_top_resource_process(self, r_type="cpu") -> Optional[str]:
        """Sistemi en çok yoran işlemi ismen bulur."""
        try:
            procs = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
                try:
                    procs.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            key = 'cpu_percent' if r_type == "cpu" else 'memory_percent'
            procs.sort(key=lambda x: x[key] if x[key] is not None else 0, reverse=True)
            
            if procs:
                top = procs[0]
                return f"{top['name']} (%{round(top[key], 1)})"
        except: pass
        return None

    def _trigger_system_warning(self, reason: str):
        """Kritik donanım eşikleri aşıldığında sistemi uyarır."""
        if self.state and hasattr(self.state, 'set_error'):
            # Burası ileride SystemState üzerinden bir 'Olay' (Event) tetikleyebilir
            logger.warning(f"🚨 SİSTEM KRİTİK EŞİKTE: {reason}")

    def _format_timedelta(self, td: timedelta) -> str:
        """Zaman farkını okunabilir Türkçe metne dönüştürür."""
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        mins, _ = divmod(rem, 60)
        parts = []
        if days > 0: parts.append(f"{days} gün")
        if hours > 0: parts.append(f"{hours} saat")
        if mins > 0: parts.append(f"{mins} dakika")
        return ", ".join(parts) if parts else "Yeni başlatıldı"

    def stop(self):
        """Servis kapatılırken GPU bağlantılarını güvenli bir şekilde sonlandırır."""
        if self.gpu_active:
            try: 
                pynvml.nvmlShutdown()
                logger.info("🔌 GPU İzleme Servisi kapatıldı.")
            except: pass
        logger.info("🔌 Sağlık takip servisi durduruldu.")