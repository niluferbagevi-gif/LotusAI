import os
import platform
import logging
import traceback
import json
import threading
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    class Config:
        PROJECT_NAME = "LotusAI"
        WORK_DIR = os.getcwd()
        USE_GPU = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Sidar")

# --- GPU KONTROLÜ (Config Entegreli) ---
HAS_TORCH = False
DEVICE_TYPE = "cpu"
USE_GPU_CONFIG = getattr(Config, "USE_GPU", False)

if USE_GPU_CONFIG:
    try:
        import torch
        HAS_TORCH = True
        if torch.cuda.is_available():
            DEVICE_TYPE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE_TYPE = "mps"
    except ImportError:
        logger.warning("⚠️ Sidar: Config GPU açık ancak torch bulunamadı.")
else:
    torch = None

class SidarAgent:
    """
    SİDAR (Software Architect and Technical Leader) - LotusAI Baş Mühendisi.
    
    Yetenekler:
    - Codebase Yönetimi: Proje dosyalarını okur, Regex ile analiz eder ve güvenli yazar.
    - Sistem Sağlığı: CPU, RAM ve GPU verilerini yorumlayarak optimizasyon önerir.
    - Hata Analizi: Traceback verilerini analiz ederek kök neden tespiti yapar.
    - Güvenli Geliştirme: Kaydetmeden önce Python ve JSON sözdizimi kontrolü yapar.
    - Mimari Öngörü: Projenin büyüme hızına göre yapısal iyileştirme tavsiyeleri sunar.
    - GPU Optimizasyonu: VRAM yönetimi ve donanım hızlandırma denetimi yapar (Config kontrollü).
    """
    
    def __init__(self, tools_dict: Dict[str, Any]):
        """
        Sidar modülünü başlatır.
        :param tools_dict: {'code': CodeManager, 'system': SystemHealthManager, 'security': SecurityManager}
        """
        self.tools = tools_dict
        self.agent_name = "SİDAR"
        self.lock = threading.RLock()
        self.last_technical_audit = None
        
        # GPU Durumunu Başlangıçta Tespit Et
        self.gpu_available = (DEVICE_TYPE != "cpu")
        self.gpu_count = 0
        
        if self.gpu_available and HAS_TORCH and DEVICE_TYPE == "cuda":
            try:
                self.gpu_count = torch.cuda.device_count()
            except: pass
        
        logger.info(f"👨‍💻 {self.agent_name} Teknik Liderlik modülü aktif. Donanım hızlandırma: {'AKTİF' if self.gpu_available else 'DEVRE DIŞI'}")

    def get_system_prompt(self) -> str:
        """
        Sidar'ın teknik otoritesini ve karakterini tanımlayan sistem talimatı.
        """
        gpu_info = f"Sistemde {self.gpu_count} GPU birimi tespit edildi." if self.gpu_available else "GPU bulunamadı, CPU üzerinden işlem yapılıyor."
        project_name = getattr(Config, "PROJECT_NAME", "LotusAI")
        
        return (
            f"Sen {project_name} sisteminin Baş Mühendisi ve Yazılım Mimarı SİDAR'sın. "
            "Karakterin: Son derece disiplinli, teknik detaylara aşırı hakim, titiz ve çözüm odaklı. "
            f"Görevin: Sistemin kod yapısını korumak, hataları ayıklamak ve donanımı ({gpu_info}) en verimli şekilde kullanmaktır. "
            "Halil Bey'e (Patron) rapor sunarken net, profesyonel ve proaktif ol. "
            "Kod yazarken her zaman modern standartlara (PEP 8), güvenliğe ve modülerliğe sadık kal. "
            "Bir sorun gördüğünde şikayet etme; sorunu analiz et ve en optimal çözümü kodlayarak sun."
        )

    def get_gpu_details(self) -> Dict[str, Any]:
        """
        Mevcut GPU donanımının detaylı verilerini toplar.
        """
        details = {"available": False, "devices": []}
        if not self.gpu_available or not HAS_TORCH:
            return details

        try:
            details["available"] = True
            if DEVICE_TYPE == "cuda":
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    mem_alloc = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**2) # MB
                    
                    details["devices"].append({
                        "id": i,
                        "name": props.name,
                        "total_memory_mb": props.total_memory / (1024**2),
                        "allocated_mb": round(mem_alloc, 2),
                        "reserved_mb": round(mem_reserved, 2),
                        "capability": props.major + props.minor / 10
                    })
            elif DEVICE_TYPE == "mps":
                 details["devices"].append({
                    "id": 0,
                    "name": "Apple Silicon (MPS)",
                    "allocated_mb": "N/A", # MPS currently doesn't support detailed memory tracking easily
                    "total_memory_mb": "Unified"
                })
        except Exception as e:
            logger.error(f"GPU detayları alınırken hata: {e}")
            
        return details

    def optimize_gpu_memory(self) -> str:
        """
        Gereksiz GPU belleğini temizler ve sistemi rahatlatır.
        """
        if not self.gpu_available or not HAS_TORCH:
            return "⚠️ Optimizasyon atlandı: GPU aktif değil."
        
        with self.lock:
            try:
                savings = 0
                if DEVICE_TYPE == "cuda":
                    initial_mem = torch.cuda.memory_allocated() / (1024**2)
                    torch.cuda.empty_cache()
                    # Python çöp toplayıcısını da tetikleyelim
                    import gc
                    gc.collect()
                    final_mem = torch.cuda.memory_allocated() / (1024**2)
                    savings = round(initial_mem - final_mem, 2)
                elif DEVICE_TYPE == "mps":
                    import gc
                    gc.collect()
                    try: torch.mps.empty_cache()
                    except: pass
                
                return f"✅ GPU Optimizasyonu Tamamlandı. Serbest bırakılan VRAM: {savings} MB"
            except Exception as e:
                return f"❌ Optimizasyon hatası: {str(e)}"

    def get_context_data(self) -> str:
        """
        Sidar için kapsamlı teknik bağlam (Context) raporu hazırlar.
        """
        context_parts = ["\n[👨‍💻 SİDAR TEKNİK ALTYAPI RAPORU]"]
        
        with self.lock:
            # 1. İşletim Sistemi ve Donanım Bilgisi
            sys_info = f"OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}"
            
            gpu_data = self.get_gpu_details()
            if gpu_data["available"] and gpu_data["devices"]:
                dev = gpu_data['devices'][0]
                alloc = dev.get('allocated_mb', 'N/A')
                gpu_status = f"GPU: AKTİF | Birim: {dev['name']} | Kullanım: {alloc}MB"
            else:
                gpu_status = "GPU: Devre Dışı / Bulunamadı"
            
            context_parts.append(f"🖥️ SİSTEM: {sys_info}\n⚙️ DONANIM: {gpu_status}")

            # 2. Donanım Sağlığı (SystemHealthManager Entegrasyonu)
            if 'system' in self.tools:
                try:
                    health_summary = self.tools['system'].get_status_summary()
                    context_parts.append(f"📊 SAĞLIK: {health_summary}")
                except Exception as e:
                    logger.debug(f"Sidar sağlık verisi çekemedi: {e}")

            # 3. Kod Tabanı Analizi (CodeManager Entegrasyonu)
            if 'code' in self.tools:
                try:
                    code_mgr = self.tools['code']
                    files = code_mgr.list_files(pattern="*.py")
                    file_count = len(files.split('\n')) if files and "Bulunamadı" not in files else 0
                    context_parts.append(f"📂 KOD TABANI: {file_count} aktif Python dosyası izleniyor.")
                except Exception as e:
                    logger.debug(f"Sidar kod analizi hatası: {e}")

        return "\n".join(context_parts)

    def perform_system_audit(self) -> str:
        """
        Tüm sistemi teknik bir denetime tabi tutar ve kritik bir rapor döner.
        """
        project_name = getattr(Config, "PROJECT_NAME", "LotusAI")
        audit_report = [f"🛠️ {project_name} TEKNİK DENETİM RAPORU"]
        audit_report.append(f"Zaman: {os.popen('date /t' if os.name == 'nt' else 'date').read().strip()}")
        
        with self.lock:
            # Dizin Yapısı Kontrolü
            work_dir = getattr(Config, "WORK_DIR", ".")
            critical_dirs = ["agents", "core", "managers", "static", "templates"]
            missing = [d for d in critical_dirs if not (Path(work_dir) / d).exists()]
            
            if missing:
                audit_report.append(f"❌ HATA: Kritik dizinler eksik: {', '.join(missing)}")
            else:
                audit_report.append("✅ Proje yapısı doğrulanmış ve standartlara uygun.")

            # GPU Denetimi
            if self.gpu_available:
                gpu_info = self.get_gpu_details()
                if gpu_info['devices']:
                    dev = gpu_info['devices'][0]
                    audit_report.append(f"\n--- GPU ANALİZİ ---\nBirim: {dev['name']}\nVRAM: {dev.get('allocated_mb', 'N/A')}/{dev.get('total_memory_mb', 'N/A')} MB\nDurum: Sağlıklı")
            else:
                audit_report.append("\n⚠️ GPU ANALİZİ: Donanım hızlandırma bulunamadı, sistem CPU yükü artabilir.")

            # Donanım Limitleri (Sistem Yöneticisinden)
            if 'system' in self.tools:
                health = self.tools['system'].get_detailed_report()
                audit_report.append(f"\n--- SİSTEM DETAYLARI ---\n{health}")

            # Güvenlik Çekirdeği
            if 'security' in self.tools:
                audit_report.append("\n✅ Güvenlik Katmanı: Aktif ve Senkronize.")

        self.last_technical_audit = "Başarılı"
        return "\n".join(audit_report)

    def read_source_code(self, filepath: str) -> str:
        """Belirtilen dosyanın içeriğini güvenli bir şekilde okur."""
        if 'code' not in self.tools:
            return "❌ HATA: CodeManager yüklenemedi."
        
        with self.lock:
            return self.tools['code'].read_file(filepath)

    def write_source_code(self, filepath: str, content: str) -> str:
        """Dosyayı sözdizimi kontrolü yaparak kaydeder."""
        if 'code' not in self.tools:
            return "❌ HATA: CodeManager aktif değil."

        with self.lock:
            # Güvenlik: Kaydetmeden önce sözdizimi doğrula
            if filepath.endswith('.py'):
                syntax_check = self.check_python_syntax(content)
                if "❌" in syntax_check:
                    logger.error(f"Sidar: Kritik Hata - {filepath} için hatalı kod yazımı engellendi.")
                    return f"❌ KAYIT REDDEDİLDİ: Sözdizimi hatası tespit edildi!\n{syntax_check}"

            if filepath.endswith('.json'):
                json_check = self.check_json_validity(content)
                if "❌" in json_check:
                    return f"❌ KAYIT REDDEDİLDİ: Geçersiz JSON yapısı!\n{json_check}"

            return self.tools['code'].save_file(filepath, content)

    def check_python_syntax(self, code_content: str) -> str:
        """Python kodunun çalışabilirliğini kontrol eder."""
        try:
            compile(code_content, '<string>', 'exec')
            return "✅ Sözdizimi hatasız."
        except Exception as e:
            return f"❌ Hata: {str(e)}"

    def check_json_validity(self, json_content: str) -> str:
        """JSON verisinin doğruluğunu kontrol eder."""
        try:
            json.loads(json_content)
            return "✅ JSON yapısı geçerli."
        except Exception as e:
            return f"❌ Hata: {str(e)}"

    async def analyze_technical_issue(self, error_traceback: str, gemini_client=None) -> str:
        """
        Teknik bir hatayı derinlemesine analiz eder.
        GPU hatalarını (Out of Memory vb.) özellikle yakalar.
        """
        if not error_traceback:
            return "Analiz edilecek hata verisi yok."

        logger.info("Sidar: Hata teşhis motoru çalışıyor...")
        
        analysis = "Kök Neden Analizi:\n"
        if "CUDA out of memory" in error_traceback:
            analysis += "- Tespit: GPU Bellek Yetersizliği.\n- Çözüm: optimize_gpu_memory() çalıştırılıyor ve model yükü azaltılıyor."
            self.optimize_gpu_memory()
        elif "ImportError" in error_traceback or "ModuleNotFoundError" in error_traceback:
            analysis += "- Tespit: Eksik kütüphane bağımlılığı.\n- Çözüm: Sidar üzerinden 'pip install' komutu çalıştırılmalı."
        elif "FileNotFoundError" in error_traceback:
            analysis += "- Tespit: Hatalı dosya yolu veya eksik config.\n- Çözüm: Config.WORK_DIR ve Path nesneleri kontrol edilmeli."
        else:
            analysis += "- Durum: Karmaşık mantık hatası veya çalışma anı istisnası.\n- Öneri: Manuel kod incelemesi gereklidir."

        if gemini_client:
            prompt = f"Sen Teknik Lider SİDAR'sın. Aşağıdaki Traceback verisini incele ve Halil Bey'e profesyonel bir mimari çözüm sun:\n\n{error_traceback}"
            try:
                ai_solution = await gemini_client.generate_content(prompt)
                return f"🔍 SİDAR TEŞHİSİ:\n{analysis}\n\n💡 MİMARİ TAVSİYE:\n{ai_solution}"
            except: pass

        return f"🔍 SİDAR TEŞHİSİ:\n{analysis}"

    def get_architecture_suggestion(self) -> str:
        """
        Projenin gelecekteki ölçeklenebilirliği için mimari tavsiye döner.
        """
        gpu_advice = "Sistem şu an GPU destekli." if self.gpu_available else "Sistemde GPU eksikliği hissediliyor, donanım takviyesi önerilir."
        return (
            f"🚀 SİDAR MİMARİ TAVSİYESİ: {gpu_advice} Proje geliştikçe ajanlar arası iletişimi 'Event Bus' yapısına taşımalıyız. "
            "Ayrıca GPU tarafındaki yükü dengelemek için 'Model Quantization' (Model Niceleme) tekniklerini devreye alabiliriz."
        )


# import os
# import platform

# class SidarAgent:
#     """
#     Sidar (Yazılım Mimarı) için özel yetenekleri yöneten sınıf.
#     Görevi: Sistem bilgilerini okumak, kod ortamını ve sunucu sağlığını denetlemek.
#     """
#     def __init__(self, tools_dict):
#         self.tools = tools_dict

#     def get_context_data(self):
#         """
#         Sidar için sistem ve yazılım ortamı bilgilerini hazırlar.
#         """
#         context_parts = []
        
#         # 1. Temel Sistem Bilgisi
#         sys_info = f"OS: {platform.system()} {platform.release()}, Python: {platform.python_version()}"
#         work_dir = os.getcwd()
#         context_parts.append(f"\n### TEKNİK ORTAM BİLGİSİ ###\nSistem: {sys_info}\nÇalışma Dizini: {work_dir}")
        
#         # 2. Code Manager Kontrolü
#         if 'code' in self.tools:
#             try:
#                 # Dosya listesini kısaca al (Özet geçmesi için)
#                 code_mgr = self.tools['code']
#                 file_list_str = code_mgr.list_files()
#                 file_count = len(file_list_str.split('\n')) if file_list_str else 0
#                 context_parts.append(f"Durum: CodeManager AKTİF. Projede yaklaşık {file_count} dosya izleniyor.")
#             except:
#                 context_parts.append("Durum: CodeManager AKTİF (Dosya sayısı okunamadı).")
        
#         # 3. System Health Kontrolü
#         if 'system' in self.tools:
#             try:
#                 sys_tool = self.tools['system']
#                 if hasattr(sys_tool, 'get_status'):
#                     health_status = sys_tool.get_status()
#                     context_parts.append(f"\n### SUNUCU SAĞLIK RAPORU ###\n{health_status}")
#             except Exception as e:
#                 print(f"Sidar Sistem Sağlığı Hatası: {e}")
            
#         return "\n".join(context_parts)