"""
LotusAI System Health Manager
Sürüm: 2.5.4 (Eklendi: Erişim Seviyesi Desteği)
Açıklama: Sunucu ve donanım sağlık yönetimi

Özellikler:
- CPU/RAM/Disk izleme
- GPU monitoring (NVIDIA)
- Network tracking
- Process analysis
- Uptime tracking
- Critical alerts
- Erişim seviyesi desteği
"""

import os
import platform
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.SystemHealth")


# ═══════════════════════════════════════════════════════════════
# LIBRARIES
# ═══════════════════════════════════════════════════════════════
PSUTIL_AVAILABLE = False
NVML_AVAILABLE = False
TORCH_AVAILABLE = False

# psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ psutil yok, sistem sağlık verileri kısıtlı")

# pynvml
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pass

# torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class HealthStatus(Enum):
    """Sistem sağlık durumları"""
    HEALTHY = "SAĞLIKLI 🟢"
    TIRED = "YORGUN 🟠"
    CRITICAL = "KRİTİK 🔴"
    UNKNOWN = "BİLİNMİYOR ⚪"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class GPUInfo:
    """GPU bilgisi"""
    index: int
    name: str
    load: int
    temperature: int
    vram_used: float
    vram_total: float
    process_count: int = 0


@dataclass
class SystemMetrics:
    """Sistem metrikleri"""
    cpu_percent: float
    ram_percent: float
    disk_percent: float
    network_upload: float
    network_download: float
    uptime: timedelta
    gpu_info: Optional[List[GPUInfo]] = None


@dataclass
class HealthMetrics:
    """Health manager metrikleri"""
    status_checks: int = 0
    detailed_reports: int = 0
    warnings_triggered: int = 0


# ═══════════════════════════════════════════════════════════════
# SYSTEM HEALTH MANAGER
# ═══════════════════════════════════════════════════════════════
class SystemHealthManager:
    """
    LotusAI Sunucu ve Donanım Sağlık Yöneticisi
    
    Yetenekler:
    - CPU/RAM/Disk: psutil ile sistem izleme
    - GPU monitoring: NVIDIA GPU takibi (pynvml)
    - Network: Ağ trafiği izleme
    - Process analysis: Yoğun işlem tespiti
    - Uptime: Çalışma süresi takibi
    - Alerts: Kritik eşik uyarıları
    - Erişim seviyesine duyarlı raporlama (opsiyonel)
    
    Sistemin tüm donanım kaynaklarını izler ve raporlar.
    """
    
    # Thresholds
    CPU_CRITICAL = 85
    CPU_WARNING = 65
    RAM_CRITICAL = 90
    RAM_WARNING = 75
    GPU_CRITICAL = 85
    
    def __init__(self, system_state: Optional[Any] = None, access_level: str = "sandbox"):
        """
        System health manager başlatıcı
        
        Args:
            system_state: SystemState objesi (opsiyonel)
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level
        
        # Thread safety
        self.lock = threading.RLock()
        
        # State
        self.state = system_state
        
        # Metrics
        self.metrics = HealthMetrics()
        
        # Tracking
        self.start_time = datetime.now()
        self.last_net_io: Optional[Any] = None
        
        # GPU
        self.gpu_active = False
        self.gpu_count = 0
        self.cuda_info = "Pasif / CPU Modu"
        
        # Initialize GPU
        self._init_gpu()
        
        # Initialize network
        if PSUTIL_AVAILABLE:
            try:
                self.last_net_io = psutil.net_io_counters()
                logger.info("✅ Sistem sağlık takip servisi hazır")
            except Exception:
                pass
        
        logger.info(f"✅ SystemHealthManager başlatıldı (Erişim: {self.access_level})")
    
    def _init_gpu(self) -> None:
        """GPU monitoring başlat"""
        if not Config.USE_GPU:
            logger.info("ℹ️ Sistem sağlık CPU modunda (Config)")
            return
        
        # NVML
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_active = True
                logger.info(f"🚀 GPU takip aktif: {self.gpu_count} cihaz")
            except Exception as e:
                logger.error(f"NVML başlatma hatası: {e}")
        else:
            logger.info("ℹ️ GPU izleme için pynvml eksik")
        
        # CUDA info
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.cuda_info = f"Aktif (v{torch.version.cuda})"
            else:
                self.cuda_info = "Pasif (Donanım Yok)"
        else:
            self.cuda_info = "Pasif (Torch Yok)"
    
    # ───────────────────────────────────────────────────────────
    # STATUS SUMMARY
    # ───────────────────────────────────────────────────────────
    
    def get_status_summary(self) -> str:
        """
        Durum özeti - Tüm erişim seviyelerinde kullanılabilir.
        
        Returns:
            Tek satır özet
        """
        if not PSUTIL_AVAILABLE:
            return "⚠️ Sistem izleme pasif (psutil eksik)"
        
        with self.lock:
            try:
                # CPU & RAM
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                
                # GPU
                gpu_info = ""
                if self.gpu_active:
                    gpu_load = self._get_gpu_load()
                    gpu_info = f" | GPU: %{gpu_load}"
                
                # Determine status
                status = self._determine_health_status(cpu, ram)
                
                # Trigger warning if critical
                if status == HealthStatus.CRITICAL:
                    self._trigger_system_warning(
                        "Yüksek donanım yükü tespit edildi"
                    )
                
                self.metrics.status_checks += 1
                
                return (
                    f"Sistem: {status.value} | "
                    f"CPU: %{cpu} | RAM: %{ram}{gpu_info}"
                )
            
            except Exception as e:
                return f"Özet alınamadı: {str(e)[:50]}"
    
    def _determine_health_status(
        self,
        cpu: float,
        ram: float
    ) -> HealthStatus:
        """Sağlık durumu belirle"""
        if cpu > self.CPU_CRITICAL or ram > self.RAM_CRITICAL:
            return HealthStatus.CRITICAL
        elif cpu > self.CPU_WARNING or ram > self.RAM_WARNING:
            return HealthStatus.TIRED
        else:
            return HealthStatus.HEALTHY
    
    # ───────────────────────────────────────────────────────────
    # DETAILED REPORT
    # ───────────────────────────────────────────────────────────
    
    def get_detailed_report(self) -> str:
        """
        Detaylı rapor - Tüm erişim seviyelerinde kullanılabilir.
        (İsteğe bağlı olarak kısıtlı modda bazı detaylar filtrelenebilir)
        
        Returns:
            Formatlanmış teknik rapor
        """
        if not PSUTIL_AVAILABLE:
            return "Rapor üretilemiyor: psutil eksik"
        
        with self.lock:
            try:
                # Get metrics
                system_metrics = self._get_system_metrics()
                
                # Status
                status = self._determine_health_status(
                    system_metrics.cpu_percent,
                    system_metrics.ram_percent
                )
                
                warning = " (KRİTİK!)" if status == HealthStatus.CRITICAL else ""
                
                # Build report
                report_lines = [
                    f"🖥️ LOTUSAI SİSTEM SAĞLIK RAPORU {status.value}{warning}",
                    "═" * 40,
                    f"🔐 Erişim: {self.access_level.upper()}",
                    f"⏱️ Uptime: {self._format_timedelta(system_metrics.uptime)}",
                    f"🤖 CUDA: {self.cuda_info}",
                    f"⚙️ CPU: %{system_metrics.cpu_percent}",
                    f"🧠 RAM: %{system_metrics.ram_percent}",
                    f"💾 Disk: %{system_metrics.disk_percent}",
                    f"🌐 Network: ↑{system_metrics.network_upload:.1f} KB/s "
                    f"↓{system_metrics.network_download:.1f} KB/s",
                    f"📑 Processes: {len(psutil.pids())}"
                ]
                
                # GPU info (her erişim seviyesinde gösterilebilir, ancak istenirse kısıtlanabilir)
                if system_metrics.gpu_info:
                    report_lines.append("─" * 40)
                    report_lines.append("🎮 GPU DURUMU (NVIDIA):")
                    
                    for gpu in system_metrics.gpu_info:
                        report_lines.append(
                            f"- GPU {gpu.index} [{gpu.name}]: "
                            f"%{gpu.load} Yük | {gpu.temperature}°C | "
                            f"VRAM: {gpu.vram_used:.0f}/{gpu.vram_total:.0f} MB"
                        )
                        
                        if gpu.process_count > 0:
                            report_lines[-1] += f" | {gpu.process_count} İşlem"
                
                # Resource-heavy processes (opsiyonel olarak kısıtlanabilir)
                # Şimdilik herkese açık
                if (system_metrics.cpu_percent > self.CPU_WARNING or
                    system_metrics.ram_percent > self.RAM_WARNING):
                    
                    report_lines.append("─" * 40)
                    
                    top_cpu = self._get_top_resource_process("cpu")
                    if top_cpu:
                        report_lines.append(f"🔥 En Yoğun CPU: {top_cpu}")
                    
                    top_ram = self._get_top_resource_process("ram")
                    if top_ram:
                        report_lines.append(f"📦 En Yoğun RAM: {top_ram}")
                
                self.metrics.detailed_reports += 1
                
                return "\n".join(report_lines)
            
            except Exception as e:
                logger.error(f"Rapor oluşturma hatası: {e}")
                return "Hata: Sistem verileri okunamadı"
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Sistem metriklerini topla"""
        # CPU & RAM
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        # Disk
        work_dir = Config.WORK_DIR
        drive_path = os.path.splitdrive(
            os.path.abspath(str(work_dir))
        )[0]
        
        if not drive_path:
            drive_path = "/"  # Linux/Unix
        
        disk = psutil.disk_usage(drive_path).percent
        
        # Network
        net_upload, net_download = self._get_network_speed()
        
        # Uptime
        uptime = datetime.now() - self.start_time
        
        # GPU
        gpu_info = None
        if self.gpu_active:
            gpu_info = self._get_all_gpu_info()
        
        return SystemMetrics(
            cpu_percent=cpu,
            ram_percent=mem.percent,
            disk_percent=disk,
            network_upload=net_upload,
            network_download=net_download,
            uptime=uptime,
            gpu_info=gpu_info
        )
    
    # ───────────────────────────────────────────────────────────
    # GPU HELPERS
    # ───────────────────────────────────────────────────────────
    
    def _get_gpu_load(self) -> int:
        """İlk GPU'nun yük yüzdesi"""
        if not self.gpu_active:
            return 0
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0
    
    def _get_all_gpu_info(self) -> List[GPUInfo]:
        """Tüm GPU'ların detaylı bilgisi"""
        gpu_list = []
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Stats
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Processes
                proc_count = self._get_gpu_process_count(handle)
                
                gpu_list.append(GPUInfo(
                    index=i,
                    name=name,
                    load=util.gpu,
                    temperature=temp,
                    vram_used=mem.used / (1024 ** 2),
                    vram_total=mem.total / (1024 ** 2),
                    process_count=proc_count
                ))
        
        except Exception as e:
            logger.error(f"GPU bilgi hatası: {e}")
        
        return gpu_list
    
    def _get_gpu_process_count(self, handle: Any) -> int:
        """GPU üzerindeki işlem sayısı"""
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            return len(procs)
        except Exception:
            return 0
    
    # ───────────────────────────────────────────────────────────
    # NETWORK
    # ───────────────────────────────────────────────────────────
    
    def _get_network_speed(self) -> Tuple[float, float]:
        """
        Ağ hızı (KB/s)
        
        Returns:
            (upload, download) tuple
        """
        try:
            current_net_io = psutil.net_io_counters()
            
            if not self.last_net_io:
                self.last_net_io = current_net_io
                return 0.0, 0.0
            
            sent = (
                (current_net_io.bytes_sent - self.last_net_io.bytes_sent) /
                1024
            )
            recv = (
                (current_net_io.bytes_recv - self.last_net_io.bytes_recv) /
                1024
            )
            
            self.last_net_io = current_net_io
            
            return sent, recv
        
        except Exception:
            return 0.0, 0.0
    
    # ───────────────────────────────────────────────────────────
    # PROCESS ANALYSIS
    # ───────────────────────────────────────────────────────────
    
    def _get_top_resource_process(self, r_type: str = "cpu") -> Optional[str]:
        """
        En yoğun işlem
        
        Args:
            r_type: cpu veya ram
        
        Returns:
            İşlem bilgisi
        """
        try:
            procs = []
            attrs = ['name', 'cpu_percent', 'memory_percent']
            
            for proc in psutil.process_iter(attrs):
                try:
                    procs.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort
            key = 'cpu_percent' if r_type == "cpu" else 'memory_percent'
            procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
            
            if procs:
                top = procs[0]
                val = top.get(key) or 0
                return f"{top['name']} (%{round(val, 1)})"
        
        except Exception:
            pass
        
        return None
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def _trigger_system_warning(self, reason: str) -> None:
        """
        Sistem uyarısı tetikle
        
        Args:
            reason: Uyarı nedeni
        """
        if self.state and hasattr(self.state, 'set_error'):
            logger.warning(f"🚨 SİSTEM KRİTİK: {reason}")
        
        self.metrics.warnings_triggered += 1
    
    def _format_timedelta(self, td: timedelta) -> str:
        """
        Timedelta formatla
        
        Args:
            td: Timedelta
        
        Returns:
            Okunabilir metin
        """
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        mins, _ = divmod(rem, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} gün")
        if hours > 0:
            parts.append(f"{hours} saat")
        if mins > 0:
            parts.append(f"{mins} dakika")
        
        return ", ".join(parts) if parts else "Yeni başlatıldı"
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Health metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "status_checks": self.metrics.status_checks,
            "detailed_reports": self.metrics.detailed_reports,
            "warnings_triggered": self.metrics.warnings_triggered,
            "gpu_active": self.gpu_active,
            "gpu_count": self.gpu_count,
            "cuda_info": self.cuda_info,
            "psutil_available": PSUTIL_AVAILABLE,
            "access_level": self.access_level
        }
    
    def stop(self) -> None:
        """Servisi kapat"""
        if self.gpu_active:
            try:
                pynvml.nvmlShutdown()
                logger.info("🔌 GPU izleme kapatıldı")
            except Exception:
                pass
        
        logger.info("🔌 Sağlık takip servisi durduruldu")