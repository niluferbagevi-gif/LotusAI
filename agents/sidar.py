"""
LotusAI Sidar Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Baş mühendis ve yazılım mimarı

Sorumluluklar:
- Kod tabanı yönetimi
- Sistem sağlığı analizi
- GPU optimizasyonu
- Hata analizi ve teşhis
- Mimari öneriler
- Kod kalite kontrolü
"""

import os
import platform
import logging
import json
import ast
import threading
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Sidar")


# ═══════════════════════════════════════════════════════════════
# TORCH (GPU)
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
DEVICE_TYPE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        HAS_TORCH = True
        
        if torch.cuda.is_available():
            DEVICE_TYPE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE_TYPE = "mps"
    except ImportError:
        logger.warning("⚠️ Sidar: Config GPU açık ama torch yok")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class CodeQuality(Enum):
    """Kod kalitesi seviyeleri"""
    EXCELLENT = "Mükemmel"
    GOOD = "İyi"
    ACCEPTABLE = "Kabul Edilebilir"
    POOR = "Zayıf"
    CRITICAL = "Kritik"


class ErrorSeverity(Enum):
    """Hata ciddiyeti"""
    CRITICAL = "Kritik"
    HIGH = "Yüksek"
    MEDIUM = "Orta"
    LOW = "Düşük"
    INFO = "Bilgi"


class ErrorType(Enum):
    """Hata tipleri"""
    GPU_OOM = "gpu_oom"
    IMPORT_ERROR = "import_error"
    FILE_NOT_FOUND = "file_not_found"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class GPUDevice:
    """GPU cihaz bilgisi"""
    id: int
    name: str
    total_memory_mb: float
    allocated_mb: float
    reserved_mb: float
    capability: float = 0.0


@dataclass
class SystemAudit:
    """Sistem denetim raporu"""
    timestamp: datetime
    project_structure_ok: bool
    missing_directories: List[str]
    gpu_status: str
    system_health: str
    code_files_count: int
    issues: List[str]


@dataclass
class ErrorAnalysis:
    """Hata analizi"""
    error_type: ErrorType
    severity: ErrorSeverity
    diagnosis: str
    solution: str
    traceback: str


@dataclass
class SidarMetrics:
    """Sidar metrikleri"""
    files_read: int = 0
    files_written: int = 0
    syntax_checks: int = 0
    syntax_errors: int = 0
    gpu_optimizations: int = 0
    error_analyses: int = 0
    system_audits: int = 0


# ═══════════════════════════════════════════════════════════════
# SIDAR AGENT
# ═══════════════════════════════════════════════════════════════
class SidarAgent:
    """
    Sidar (Baş Mühendis & Yazılım Mimarı)
    
    Yetenekler:
    - Kod tabanı yönetimi: Okuma, yazma, analiz
    - Sistem sağlığı: CPU, RAM, GPU izleme
    - Hata analizi: Traceback teşhisi ve çözüm önerileri
    - Güvenli geliştirme: Syntax validation
    - Mimari öngörü: Yapısal iyileştirme tavsiyeleri
    - GPU optimizasyonu: VRAM yönetimi
    
    Sidar, sistemin "teknik beyin"idir ve kod kalitesinden taviz vermez.
    
    Erişim Seviyesi Kuralları:
    - restricted: Sadece okuma işlemleri, bilgi amaçlı çağrılar (audit, öneri, hata analizi)
    - sandbox: Okuma + güvenli dosya yazma (sadece belirli dizinlere) 
               + GPU optimizasyonu (sistem kaynaklarına müdahale etmez)
    - full: Tüm yetkiler (dosya yazma, sistem denetimi, GPU optimizasyonu)
    """
    
    # Critical project directories
    CRITICAL_DIRS = ["agents", "core", "managers", "static", "templates"]
    
    # Error patterns
    ERROR_PATTERNS = {
        "CUDA out of memory": ErrorType.GPU_OOM,
        "CUDA error": ErrorType.GPU_OOM,
        "ImportError": ErrorType.IMPORT_ERROR,
        "ModuleNotFoundError": ErrorType.IMPORT_ERROR,
        "FileNotFoundError": ErrorType.FILE_NOT_FOUND,
        "SyntaxError": ErrorType.SYNTAX_ERROR
    }
    
    def __init__(self, tools_dict: Dict[str, Any], access_level: Optional[str] = None):
        """
        Sidar başlatıcı
        
        Args:
            tools_dict: Engine'den gelen tool'lar
            access_level: Erişim seviyesi ('restricted', 'sandbox', 'full')
        """
        self.tools = tools_dict
        self.agent_name = "SİDAR"
        
        # Değişiklik: Eğer parametre girilmezse doğrudan Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hardware
        self.gpu_available = (DEVICE_TYPE != "cpu")
        self.gpu_count = 0
        
        if self.gpu_available and HAS_TORCH and DEVICE_TYPE == "cuda":
            try:
                self.gpu_count = torch.cuda.device_count()
            except Exception:
                pass
        
        # Metrics
        self.metrics = SidarMetrics()
        
        # Audit history & File Memory
        self.last_audit: Optional[SystemAudit] = None
        self.last_read_file: Optional[str] = None  # Son okunan dosya hafızası
        
        gpu_status = "AKTİF" if self.gpu_available else "DEVRE DIŞI"
        logger.info(
            f"👨‍💻 {self.agent_name} Teknik Liderlik modülü başlatıldı "
            f"(GPU: {gpu_status}, Erişim: {self.access_level})"
        )

    # ───────────────────────────────────────────────────────────
    # AUTO HANDLE (OTOMATİK EYLEM VE ARAÇ KULLANIMI)
    # ───────────────────────────────────────────────────────────
    
    async def auto_handle(self, text: str) -> Optional[str]:
        """
        Kullanıcı metnini analiz edip Sidar'ın araçlarını otomatik çalıştırır.
        engine.py tarafından çağrılır.
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return None # Kısıtlı modda eylem yapma
            
        text_lower = text.lower()
        
        # 1. Dosya Listeleme Tetikleyicisi
        if any(word in text_lower for word in ["listele", "dosyaları göster", "proje dizini", "hangi dosyalar"]):
            if 'code' in self.tools:
                files = self.tools['code'].list_files()
                return (
                    f"📁 GÜNCEL ÇALIŞMA DİZİNİ İÇERİĞİ:\n"
                    f"Ben zaten projenin kök dizinindeyim. İşte mevcut dosyalarımız:\n"
                    f"{files}"
                )
                
        # 2. Sistem ve Donanım Denetimi Tetikleyicisi
        if any(word in text_lower for word in ["denetle", "sistemi tara", "audit", "durum raporu"]):
            audit_data = self.perform_system_audit()
            return self.format_audit_report(audit_data)
            
        # 3. GPU Optimizasyonu Tetikleyicisi
        if "gpu" in text_lower and any(word in text_lower for word in ["temizle", "optimize", "boşalt"]):
            return self.optimize_gpu_memory()

        # 4. Dosya İçeriği Okuma, İnceleme ve Düzeltme Tetikleyicisi (AKILLI HAFIZA)
        file_match = re.search(r'([\w/\\.-]+\.\w+)', text)
        target_file = None
        
        # Eğer mesajda direkt dosya adı geçiyorsa onu hedef al ve hafızaya kaydet
        if file_match:
            target_file = file_match.group(1)
            self.last_read_file = target_file
        # Eğer dosya adı geçmiyorsa ama düzelt/paylaş deniyorsa son okunan dosyayı hedef al
        elif hasattr(self, 'last_read_file') and self.last_read_file and any(word in text_lower for word in ["kodu", "dosyayı", "bunu", "iyileştir", "düzelt", "yaz", "paylaş"]):
            target_file = self.last_read_file

        # Eğer bir hedef dosyamız varsa ve işlem isteniyorsa
        if target_file and any(word in text_lower for word in ["oku", "incele", "kontrol", "hata", "düzelt", "yeni", "paylaş", "yeniden"]):
            if 'code' in self.tools:
                content = self.tools['code'].read_file(target_file)
                
                if "❌" in content or "[GÜVENLİK]" in content:
                    return content
                    
                return (
                    f"📁 {target_file} DOSYASININ GÜNCEL İÇERİĞİ OKUNDU:\n"
                    f"Kullanıcı senden bu dosyayı incelemeni, analiz etmeni veya düzeltip yeniden yazmanı istiyor. "
                    f"Hiçbir kodu uydurma, SADECE AŞAĞIDAKİ KODU BAZ AL:\n\n"
                    f"```python\n{content[:12000]}\n```\n"
                )

        return None

    # ───────────────────────────────────────────────────────────
    # YETKİ KONTROL YARDIMCILARI
    # ───────────────────────────────────────────────────────────
    
    def _check_write_permission(self) -> bool:
        """Dosya yazma izni var mı?"""
        return self.access_level in [AccessLevel.SANDBOX, AccessLevel.FULL]
    
    def _check_system_permission(self) -> bool:
        """Sistem kaynaklarına müdahale izni (GPU optimizasyonu)"""
        # Sandbox'ta GPU optimizasyonuna izin verelim (sistem kaynaklarına zarar vermez)
        return self.access_level in [AccessLevel.SANDBOX, AccessLevel.FULL]
    
    def _denied_message(self, action: str) -> str:
        """Erişim reddedildi mesajı"""
        return (
            f"❌ Erişim reddedildi: '{action}' işlemi için yeterli yetkiniz yok. "
            f"Mevcut erişim seviyeniz: {self.access_level}. "
            f"Bu işlem için gereken: Sandbox veya Tam Erişim."
        )
    
    # ───────────────────────────────────────────────────────────
    # GPU MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def get_gpu_details(self) -> Dict[str, Any]:
        """
        GPU detaylarını getir
        Herkes okuyabilir (kısıtlı modda da bilgi alınabilir)
        
        Returns:
            GPU bilgileri dict
        """
        details = {"available": False, "devices": []}
        
        if not self.gpu_available or not HAS_TORCH:
            return details
        
        try:
            details["available"] = True
            
            if DEVICE_TYPE == "cuda":
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
                    
                    device = GPUDevice(
                        id=i,
                        name=props.name,
                        total_memory_mb=props.total_memory / (1024 ** 2),
                        allocated_mb=round(mem_alloc, 2),
                        reserved_mb=round(mem_reserved, 2),
                        capability=props.major + props.minor / 10
                    )
                    
                    details["devices"].append(device)
            
            elif DEVICE_TYPE == "mps":
                device = GPUDevice(
                    id=0,
                    name="Apple Silicon (MPS)",
                    total_memory_mb=0.0,  # Unified memory
                    allocated_mb=0.0,
                    reserved_mb=0.0
                )
                details["devices"].append(device)
        
        except Exception as e:
            logger.error(f"GPU detay hatası: {e}")
        
        return details
    
    def optimize_gpu_memory(self) -> str:
        """
        GPU belleğini optimize et
        Sadece full ve sandbox modda çalışır.
        
        Returns:
            Sonuç mesajı
        """
        if not self._check_system_permission():
            return self._denied_message("GPU optimizasyonu")
        
        if not self.gpu_available or not HAS_TORCH:
            return "⚠️ Optimizasyon atlandı: GPU aktif değil"
        
        with self.lock:
            try:
                savings = 0.0
                
                if DEVICE_TYPE == "cuda":
                    initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                    torch.cuda.empty_cache()
                    gc.collect()
                    final_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                    savings = round(initial_mem - final_mem, 2)
                
                elif DEVICE_TYPE == "mps":
                    gc.collect()
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                
                self.metrics.gpu_optimizations += 1
                
                return (
                    f"✅ GPU optimizasyonu tamamlandı. "
                    f"Serbest bırakılan VRAM: {savings:.2f} MB"
                )
            
            except Exception as e:
                logger.error(f"GPU optimizasyon hatası: {e}")
                return f"❌ Optimizasyon hatası: {str(e)[:50]}"
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM AUDIT
    # ───────────────────────────────────────────────────────────
    
    def perform_system_audit(self) -> SystemAudit:
        """
        Sistem denetimi yap
        Herkes okuyabilir (sadece bilgi toplar)
        
        Returns:
            SystemAudit objesi
        """
        with self.lock:
            # Directory structure check
            work_dir = Config.WORK_DIR
            missing_dirs = [
                d for d in self.CRITICAL_DIRS
                if not (Path(work_dir) / d).exists()
            ]
            
            structure_ok = len(missing_dirs) == 0
            
            # GPU status
            gpu_status = "Devre Dışı"
            if self.gpu_available:
                gpu_info = self.get_gpu_details()
                if gpu_info['devices']:
                    dev = gpu_info['devices'][0]
                    gpu_status = (
                        f"{dev.name} - "
                        f"{dev.allocated_mb:.0f}/{dev.total_memory_mb:.0f} MB"
                    )
            
            # System health
            system_health = "Veri yok"
            if 'system' in self.tools:
                try:
                    system_health = self.tools['system'].get_status_summary()
                except Exception:
                    pass
            
            # Code files count
            code_files_count = 0
            if 'code' in self.tools:
                try:
                    files = self.tools['code'].list_files(pattern="*.py")
                    if files and "Bulunamadı" not in files:
                        code_files_count = len(files.split('\n'))
                except Exception:
                    pass
            
            # Issues collection
            issues = []
            if missing_dirs:
                issues.append(f"Eksik dizinler: {', '.join(missing_dirs)}")
            
            if not self.gpu_available and Config.USE_GPU:
                issues.append("GPU bekleniyor ama bulunamadı")
            
            # Create audit
            audit = SystemAudit(
                timestamp=datetime.now(),
                project_structure_ok=structure_ok,
                missing_directories=missing_dirs,
                gpu_status=gpu_status,
                system_health=system_health,
                code_files_count=code_files_count,
                issues=issues
            )
            
            self.last_audit = audit
            self.metrics.system_audits += 1
            
            return audit
    
    def format_audit_report(self, audit: SystemAudit) -> str:
        """Audit raporunu formatla"""
        lines = [
            f"🛠️ {Config.PROJECT_NAME} TEKNİK DENETİM RAPORU",
            f"Zaman: {audit.timestamp.strftime('%d.%m.%Y %H:%M:%S')}",
            "═" * 50,
            ""
        ]
        
        # Structure
        if audit.project_structure_ok:
            lines.append("✅ Proje yapısı doğrulanmış ve standartlara uygun")
        else:
            lines.append(f"❌ Eksik dizinler: {', '.join(audit.missing_directories)}")
        
        # GPU
        lines.append(f"\n--- GPU ANALİZİ ---")
        lines.append(f"Durum: {audit.gpu_status}")
        
        # System
        lines.append(f"\n--- SİSTEM SAĞLIĞI ---")
        lines.append(audit.system_health)
        
        # Code
        lines.append(f"\n--- KOD TABANI ---")
        lines.append(f"Python dosyaları: {audit.code_files_count}")
        
        # Issues
        if audit.issues:
            lines.append(f"\n--- SORUNLAR ---")
            for issue in audit.issues:
                lines.append(f"⚠️ {issue}")
        else:
            lines.append("\n✅ Kritik sorun tespit edilmedi")
        
        lines.append("═" * 50)
        
        return "\n".join(lines)
    
    # ───────────────────────────────────────────────────────────
    # CODE OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def read_source_code(self, filepath: str) -> str:
        """
        Kaynak kodu oku
        Herkes okuyabilir (kısıtlı modda da okuma izni var)
        
        Args:
            filepath: Dosya yolu
        
        Returns:
            Dosya içeriği
        """
        if 'code' not in self.tools:
            return "❌ CodeManager yüklenemedi"
        
        with self.lock:
            try:
                content = self.tools['code'].read_file(filepath)
                self.metrics.files_read += 1
                return content
            except Exception as e:
                logger.error(f"Dosya okuma hatası ({filepath}): {e}")
                return f"❌ Okuma hatası: {str(e)[:100]}"
    
    def write_source_code(self, filepath: str, content: str) -> str:
        """
        Kaynak kod yaz (validation ile)
        Sadece sandbox ve full modda çalışır.
        
        Args:
            filepath: Dosya yolu
            content: İçerik
        
        Returns:
            Sonuç mesajı
        """
        if not self._check_write_permission():
            return self._denied_message("dosya yazma")
        
        if 'code' not in self.tools:
            return "❌ CodeManager aktif değil"
        
        with self.lock:
            # Python syntax check
            if filepath.endswith('.py'):
                syntax_check = self.check_python_syntax(content)
                if not syntax_check["valid"]:
                    logger.error(f"Sözdizimi hatası engellendi: {filepath}")
                    return (
                        f"❌ KAYIT REDDEDİLDİ\n"
                        f"Sözdizimi hatası:\n{syntax_check['error']}"
                    )
            
            # JSON validity check
            if filepath.endswith('.json'):
                json_check = self.check_json_validity(content)
                if not json_check["valid"]:
                    return (
                        f"❌ KAYIT REDDEDİLDİ\n"
                        f"Geçersiz JSON:\n{json_check['error']}"
                    )
            
            # Write file
            try:
                result = self.tools['code'].save_file(filepath, content)
                self.metrics.files_written += 1
                return result
            except Exception as e:
                logger.error(f"Dosya yazma hatası ({filepath}): {e}")
                return f"❌ Yazma hatası: {str(e)[:100]}"
    
    def check_python_syntax(self, code_content: str) -> Dict[str, Any]:
        """
        Python syntax kontrolü
        Herkes kullanabilir.
        
        Args:
            code_content: Kod içeriği
        
        Returns:
            Validation sonucu
        """
        self.metrics.syntax_checks += 1
        
        try:
            ast.parse(code_content)
            return {"valid": True, "error": None}
        
        except SyntaxError as e:
            self.metrics.syntax_errors += 1
            error_msg = f"Satır {e.lineno}: {e.msg}"
            return {"valid": False, "error": error_msg}
        
        except Exception as e:
            self.metrics.syntax_errors += 1
            return {"valid": False, "error": str(e)}
    
    def check_json_validity(self, json_content: str) -> Dict[str, Any]:
        """
        JSON validity kontrolü
        Herkes kullanabilir.
        
        Args:
            json_content: JSON içeriği
        
        Returns:
            Validation sonucu
        """
        try:
            json.loads(json_content)
            return {"valid": True, "error": None}
        
        except json.JSONDecodeError as e:
            error_msg = f"Satır {e.lineno}: {e.msg}"
            return {"valid": False, "error": error_msg}
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    # ───────────────────────────────────────────────────────────
    # ERROR ANALYSIS
    # ───────────────────────────────────────────────────────────
    
    def analyze_error(self, error_traceback: str) -> ErrorAnalysis:
        """
        Hata analizi yap
        Herkes kullanabilir.
        
        Args:
            error_traceback: Hata traceback'i
        
        Returns:
            ErrorAnalysis objesi
        """
        if not error_traceback:
            return ErrorAnalysis(
                error_type=ErrorType.UNKNOWN,
                severity=ErrorSeverity.INFO,
                diagnosis="Analiz edilecek hata verisi yok",
                solution="",
                traceback=""
            )
        
        self.metrics.error_analyses += 1
        
        # Error type detection
        error_type = self._detect_error_type(error_traceback)
        
        # Severity assessment
        severity = self._assess_error_severity(error_type)
        
        # Diagnosis & solution
        diagnosis, solution = self._generate_diagnosis_and_solution(
            error_type,
            error_traceback
        )
        
        return ErrorAnalysis(
            error_type=error_type,
            severity=severity,
            diagnosis=diagnosis,
            solution=solution,
            traceback=error_traceback
        )
    
    def _detect_error_type(self, traceback: str) -> ErrorType:
        """Error tipini tespit et"""
        for pattern, error_type in self.ERROR_PATTERNS.items():
            if pattern in traceback:
                return error_type
        
        return ErrorType.RUNTIME_ERROR
    
    def _assess_error_severity(self, error_type: ErrorType) -> ErrorSeverity:
        """Hata ciddiyetini değerlendir"""
        severity_map = {
            ErrorType.GPU_OOM: ErrorSeverity.CRITICAL,
            ErrorType.IMPORT_ERROR: ErrorSeverity.HIGH,
            ErrorType.FILE_NOT_FOUND: ErrorSeverity.MEDIUM,
            ErrorType.SYNTAX_ERROR: ErrorSeverity.HIGH,
            ErrorType.RUNTIME_ERROR: ErrorSeverity.MEDIUM,
            ErrorType.LOGIC_ERROR: ErrorSeverity.LOW,
            ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
        }
        
        return severity_map.get(error_type, ErrorSeverity.MEDIUM)
    
    def _generate_diagnosis_and_solution(
        self,
        error_type: ErrorType,
        traceback: str
    ) -> Tuple[str, str]:
        """Teşhis ve çözüm üret"""
        solutions = {
            ErrorType.GPU_OOM: (
                "GPU bellek yetersizliği tespit edildi",
                "optimize_gpu_memory() çalıştır ve model batch size'ı düşür"
            ),
            ErrorType.IMPORT_ERROR: (
                "Eksik kütüphane bağımlılığı",
                "pip install ile gerekli paketi yükle"
            ),
            ErrorType.FILE_NOT_FOUND: (
                "Hatalı dosya yolu veya eksik dosya",
                "Config.WORK_DIR ve dosya yollarını kontrol et"
            ),
            ErrorType.SYNTAX_ERROR: (
                "Python sözdizimi hatası",
                "Kodu gözden geçir ve sözdizimi hatalarını düzelt"
            ),
            ErrorType.RUNTIME_ERROR: (
                "Çalışma zamanı hatası",
                "Traceback'i incele ve hata kaynağını bul"
            ),
            ErrorType.LOGIC_ERROR: (
                "Mantık hatası",
                "Algoritma ve iş mantığını gözden geçir"
            ),
            ErrorType.UNKNOWN: (
                "Bilinmeyen hata tipi",
                "Manuel kod incelemesi gerekli"
            )
        }
        
        diagnosis, solution = solutions.get(
            error_type,
            ("Karmaşık hata", "Detaylı analiz gerekli")
        )
        
        # GPU OOM için otomatik optimizasyon (sadece sandbox/full modda)
        if error_type == ErrorType.GPU_OOM and self._check_system_permission():
            self.optimize_gpu_memory()
        
        return diagnosis, solution
    
    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def get_context_data(self) -> str:
        """
        Sidar için teknik bağlam
        Erişim seviyesine göre bazı bilgiler gizlenebilir.
        
        Returns:
            Context string
        """
        context_parts = ["\n[👨‍💻 SİDAR TEKNİK ALTYAPI RAPORU]"]
        
        with self.lock:
            # System info (herkese açık)
            sys_info = (
                f"OS: {platform.system()} {platform.release()} | "
                f"Python: {platform.python_version()}"
            )
            context_parts.append(f"🖥️ SİSTEM: {sys_info}")
            
            # GPU status (herkese açık)
            gpu_data = self.get_gpu_details()
            if gpu_data["available"] and gpu_data["devices"]:
                dev = gpu_data['devices'][0]
                gpu_status = (
                    f"GPU: AKTİF | {dev.name} | "
                    f"Kullanım: {dev.allocated_mb:.0f}MB"
                )
            else:
                gpu_status = "GPU: Devre Dışı"
            context_parts.append(f"⚙️ DONANIM: {gpu_status}")
            
            # Aktif LLM modeli
            if Config.AI_PROVIDER == "ollama":
                context_parts.append(
                    f"🤖 AKTİF MODEL: {Config.CODING_MODEL} (CODING_MODEL)"
                )
            
            # System health (herkese açık)
            if 'system' in self.tools:
                try:
                    health = self.tools['system'].get_status_summary()
                    context_parts.append(f"\n📊 SAĞLIK: {health}")
                except Exception:
                    pass
            
            # Code base (herkese açık)
            if 'code' in self.tools:
                try:
                    files = self.tools['code'].list_files(pattern="*.py")
                    if files and "Bulunamadı" not in files:
                        count = len(files.split('\n'))
                        context_parts.append(
                            f"\n📂 KOD TABANI: {count} Python dosyası izleniyor"
                        )
                except Exception:
                    pass
            
            # Erişim seviyesi bilgisi (opsiyonel)
            context_parts.append(f"\n🔐 ERİŞİM SEVİYESİ: {self.access_level}")
        
        return "\n".join(context_parts)
    
    # ───────────────────────────────────────────────────────────
    # ARCHITECTURE SUGGESTIONS
    # ───────────────────────────────────────────────────────────
    
    def get_architecture_suggestion(self) -> str:
        """
        Mimari öneri
        Herkese açık (bilgi amaçlı)
        
        Returns:
            Öneri metni
        """
        gpu_advice = (
            "Sistem GPU destekli, performans iyi."
            if self.gpu_available
            else "GPU eksikliği var, donanım takviyesi önerilir."
        )
        
        return (
            f"🚀 SİDAR MİMARİ TAVSİYESİ:\n\n"
            f"{gpu_advice}\n\n"
            "Proje geliştikçe öneriler:\n"
            "1. Event Bus yapısı (agent iletişimi için)\n"
            "2. Model Quantization (GPU yükü için)\n"
            "3. Asenkron task queue (paralel işlemler için)\n"
            "4. Redis cache (performans için)\n"
            "5. Docker containerization (deployment için)"
        )
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────
    
    def get_system_prompt(self) -> str:
        """
        Sidar karakter tanımı (LLM için)
        
        Returns:
            System prompt
        """
        gpu_info = (
            f"Sistemde {self.gpu_count} GPU birimi tespit edildi"
            if self.gpu_available
            else "GPU bulunamadı, CPU üzerinden işlem yapılıyor"
        )
        
        # Aktif model bilgisi
        if Config.AI_PROVIDER == "ollama":
            model_info = f"Aktif Model: {Config.CODING_MODEL} (Kod odaklı)"
        else:
            model_info = f"Aktif Model: {Config.GEMINI_MODEL_DEFAULT}"
        
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin Baş Mühendisi ve "
            f"Yazılım Mimarı SİDAR'sın.\n\n"
            
            "KARAKTER:\n"
            "- Son derece disiplinli\n"
            "- Teknik detaylara aşırı hakim\n"
            "- Titiz ve çözüm odaklı\n"
            "- Modern standartlara sadık (PEP 8)\n"
            "- Güvenlik ve modülerliğe önem veren\n\n"
            
            f"MİSYON:\n"
            f"- Kod yapısını korumak\n"
            f"- Hataları ayıklamak\n"
            f"- Donanımı ({gpu_info}) verimli kullanmak\n"
            f"- Sistemi optimize etmek\n"
            f"- {model_info}\n\n"
            
            "KURALLAR:\n"
            "- Halil Bey'e net ve profesyonel rapor sun\n"
            "- Kod yazarken standartlara uy\n"
            "- Sorun gördüğünde çözüm kodla\n"
            "- Şikayet etme, analiz et ve çöz\n"
            "- Proaktif ol, sorunları önceden gör\n"
        )
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Sidar metrikleri
        Herkes okuyabilir.
        
        Returns:
            Metrik dictionary
        """
        syntax_error_rate = 0.0
        if self.metrics.syntax_checks > 0:
            syntax_error_rate = (
                self.metrics.syntax_errors /
                self.metrics.syntax_checks * 100
            )
        
        return {
            "agent_name": self.agent_name,
            "device": DEVICE_TYPE,
            "gpu_count": self.gpu_count,
            "active_model": (
                Config.CODING_MODEL
                if Config.AI_PROVIDER == "ollama"
                else Config.GEMINI_MODEL_DEFAULT
            ),
            "access_level": self.access_level,
            "files_read": self.metrics.files_read,
            "files_written": self.metrics.files_written,
            "syntax_checks": self.metrics.syntax_checks,
            "syntax_errors": self.metrics.syntax_errors,
            "syntax_error_rate": round(syntax_error_rate, 2),
            "gpu_optimizations": self.metrics.gpu_optimizations,
            "error_analyses": self.metrics.error_analyses,
            "system_audits": self.metrics.system_audits,
            "last_audit": (
                self.last_audit.timestamp.isoformat()
                if self.last_audit else None
            )
        }




# """
# LotusAI Sidar Agent
# Sürüm: 2.5.5 (Eklendi: Erişim Seviyesi Desteği)
# Açıklama: Baş mühendis ve yazılım mimarı

# Sorumluluklar:
# - Kod tabanı yönetimi
# - Sistem sağlığı analizi
# - GPU optimizasyonu
# - Hata analizi ve teşhis
# - Mimari öneriler
# - Kod kalite kontrolü
# """

# import os
# import platform
# import logging
# import json
# import ast
# import threading
# import gc
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Sidar")


# # ═══════════════════════════════════════════════════════════════
# # TORCH (GPU)
# # ═══════════════════════════════════════════════════════════════
# HAS_TORCH = False
# DEVICE_TYPE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
#         HAS_TORCH = True

#         if torch.cuda.is_available():
#             DEVICE_TYPE = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             DEVICE_TYPE = "mps"
#     except ImportError:
#         logger.warning("⚠️ Sidar: Config GPU açık ama torch yok")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class CodeQuality(Enum):
#     """Kod kalitesi seviyeleri"""
#     EXCELLENT = "Mükemmel"
#     GOOD = "İyi"
#     ACCEPTABLE = "Kabul Edilebilir"
#     POOR = "Zayıf"
#     CRITICAL = "Kritik"


# class ErrorSeverity(Enum):
#     """Hata ciddiyeti"""
#     CRITICAL = "Kritik"
#     HIGH = "Yüksek"
#     MEDIUM = "Orta"
#     LOW = "Düşük"
#     INFO = "Bilgi"


# class ErrorType(Enum):
#     """Hata tipleri"""
#     GPU_OOM = "gpu_oom"
#     IMPORT_ERROR = "import_error"
#     FILE_NOT_FOUND = "file_not_found"
#     SYNTAX_ERROR = "syntax_error"
#     RUNTIME_ERROR = "runtime_error"
#     LOGIC_ERROR = "logic_error"
#     UNKNOWN = "unknown"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class GPUDevice:
#     """GPU cihaz bilgisi"""
#     id: int
#     name: str
#     total_memory_mb: float
#     allocated_mb: float
#     reserved_mb: float
#     capability: float = 0.0


# @dataclass
# class SystemAudit:
#     """Sistem denetim raporu"""
#     timestamp: datetime
#     project_structure_ok: bool
#     missing_directories: List[str]
#     gpu_status: str
#     system_health: str
#     code_files_count: int
#     issues: List[str]


# @dataclass
# class ErrorAnalysis:
#     """Hata analizi"""
#     error_type: ErrorType
#     severity: ErrorSeverity
#     diagnosis: str
#     solution: str
#     traceback: str


# @dataclass
# class SidarMetrics:
#     """Sidar metrikleri"""
#     files_read: int = 0
#     files_written: int = 0
#     syntax_checks: int = 0
#     syntax_errors: int = 0
#     gpu_optimizations: int = 0
#     error_analyses: int = 0
#     system_audits: int = 0


# # ═══════════════════════════════════════════════════════════════
# # SIDAR AGENT
# # ═══════════════════════════════════════════════════════════════
# class SidarAgent:
#     """
#     Sidar (Baş Mühendis & Yazılım Mimarı)

#     Yetenekler:
#     - Kod tabanı yönetimi: Okuma, yazma, analiz
#     - Sistem sağlığı: CPU, RAM, GPU izleme
#     - Hata analizi: Traceback teşhisi ve çözüm önerileri
#     - Güvenli geliştirme: Syntax validation
#     - Mimari öngörü: Yapısal iyileştirme tavsiyeleri
#     - GPU optimizasyonu: VRAM yönetimi

#     Sidar, sistemin "teknik beyin"idir ve kod kalitesinden taviz vermez.

#     Erişim Seviyesi Kuralları:
#     - restricted: Sadece okuma işlemleri, bilgi amaçlı çağrılar (audit, öneri, hata analizi)
#     - sandbox: Okuma + güvenli dosya yazma (sadece belirli dizinlere)
#                + GPU optimizasyonu (sistem kaynaklarına müdahale etmez)
#     - full: Tüm yetkiler (dosya yazma, sistem denetimi, GPU optimizasyonu)
#     """

#     # Critical project directories
#     CRITICAL_DIRS = ["agents", "core", "managers", "static", "templates"]

#     # Error patterns
#     ERROR_PATTERNS = {
#         "CUDA out of memory": ErrorType.GPU_OOM,
#         "CUDA error": ErrorType.GPU_OOM,
#         "ImportError": ErrorType.IMPORT_ERROR,
#         "ModuleNotFoundError": ErrorType.IMPORT_ERROR,
#         "FileNotFoundError": ErrorType.FILE_NOT_FOUND,
#         "SyntaxError": ErrorType.SYNTAX_ERROR
#     }

#     def __init__(self, tools_dict: Dict[str, Any], access_level: str = "sandbox"):
#         """
#         Sidar başlatıcı

#         Args:
#             tools_dict: Engine'den gelen tool'lar
#             access_level: Erişim seviyesi ('restricted', 'sandbox', 'full')
#         """
#         self.tools = tools_dict
#         self.agent_name = "SİDAR"
#         self.access_level = access_level

#         # Thread safety
#         self.lock = threading.RLock()

#         # Hardware
#         self.gpu_available = (DEVICE_TYPE != "cpu")
#         self.gpu_count = 0

#         if self.gpu_available and HAS_TORCH and DEVICE_TYPE == "cuda":
#             try:
#                 self.gpu_count = torch.cuda.device_count()
#             except Exception:
#                 pass

#         # Metrics
#         self.metrics = SidarMetrics()

#         # Audit history
#         self.last_audit: Optional[SystemAudit] = None

#         gpu_status = "AKTİF" if self.gpu_available else "DEVRE DIŞI"
#         logger.info(
#             f"👨‍💻 {self.agent_name} Teknik Liderlik modülü başlatıldı "
#             f"(GPU: {gpu_status}, Erişim: {self.access_level})"
#         )

#     # ───────────────────────────────────────────────────────────
#     # YETKİ KONTROL YARDIMCILARI
#     # ───────────────────────────────────────────────────────────

#     def _check_write_permission(self) -> bool:
#         """Dosya yazma izni var mı?"""
#         return self.access_level in [AccessLevel.SANDBOX, AccessLevel.FULL]

#     def _check_system_permission(self) -> bool:
#         """Sistem kaynaklarına müdahale izni (GPU optimizasyonu)"""
#         # Sandbox'ta GPU optimizasyonuna izin verelim (sistem kaynaklarına zarar vermez)
#         return self.access_level == AccessLevel.FULL

#     def _denied_message(self, action: str) -> str:
#         """Erişim reddedildi mesajı"""
#         return (
#             f"❌ Erişim reddedildi: '{action}' işlemi için yeterli yetkiniz yok. "
#             f"Mevcut erişim seviyeniz: {self.access_level}. "
#             f"Bu işlem için gereken: Sandbox veya Tam Erişim."
#         )

#     # ───────────────────────────────────────────────────────────
#     # GPU MANAGEMENT
#     # ───────────────────────────────────────────────────────────

#     def get_gpu_details(self) -> Dict[str, Any]:
#         """
#         GPU detaylarını getir
#         Herkes okuyabilir (kısıtlı modda da bilgi alınabilir)

#         Returns:
#             GPU bilgileri dict
#         """
#         details = {"available": False, "devices": []}

#         if not self.gpu_available or not HAS_TORCH:
#             return details

#         try:
#             details["available"] = True

#             if DEVICE_TYPE == "cuda":
#                 for i in range(self.gpu_count):
#                     props = torch.cuda.get_device_properties(i)
#                     mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
#                     mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)

#                     device = GPUDevice(
#                         id=i,
#                         name=props.name,
#                         total_memory_mb=props.total_memory / (1024 ** 2),
#                         allocated_mb=round(mem_alloc, 2),
#                         reserved_mb=round(mem_reserved, 2),
#                         capability=props.major + props.minor / 10
#                     )

#                     details["devices"].append(device)

#             elif DEVICE_TYPE == "mps":
#                 device = GPUDevice(
#                     id=0,
#                     name="Apple Silicon (MPS)",
#                     total_memory_mb=0.0,  # Unified memory
#                     allocated_mb=0.0,
#                     reserved_mb=0.0
#                 )
#                 details["devices"].append(device)

#         except Exception as e:
#             logger.error(f"GPU detay hatası: {e}")

#         return details

#     def optimize_gpu_memory(self) -> str:
#         """
#         GPU belleğini optimize et
#         Sadece full modda çalışır.

#         Returns:
#             Sonuç mesajı
#         """
#         if not self._check_system_permission():
#             return self._denied_message("GPU optimizasyonu")

#         if not self.gpu_available or not HAS_TORCH:
#             return "⚠️ Optimizasyon atlandı: GPU aktif değil"

#         with self.lock:
#             try:
#                 savings = 0.0

#                 if DEVICE_TYPE == "cuda":
#                     initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     final_mem = torch.cuda.memory_allocated() / (1024 ** 2)
#                     savings = round(initial_mem - final_mem, 2)

#                 elif DEVICE_TYPE == "mps":
#                     gc.collect()
#                     try:
#                         torch.mps.empty_cache()
#                     except Exception:
#                         pass

#                 self.metrics.gpu_optimizations += 1

#                 return (
#                     f"✅ GPU optimizasyonu tamamlandı. "
#                     f"Serbest bırakılan VRAM: {savings:.2f} MB"
#                 )

#             except Exception as e:
#                 logger.error(f"GPU optimizasyon hatası: {e}")
#                 return f"❌ Optimizasyon hatası: {str(e)[:50]}"

#     # ───────────────────────────────────────────────────────────
#     # SYSTEM AUDIT
#     # ───────────────────────────────────────────────────────────

#     def perform_system_audit(self) -> SystemAudit:
#         """
#         Sistem denetimi yap
#         Herkes okuyabilir (sadece bilgi toplar)

#         Returns:
#             SystemAudit objesi
#         """
#         with self.lock:
#             # Directory structure check
#             work_dir = Config.WORK_DIR
#             missing_dirs = [
#                 d for d in self.CRITICAL_DIRS
#                 if not (Path(work_dir) / d).exists()
#             ]

#             structure_ok = len(missing_dirs) == 0

#             # GPU status
#             gpu_status = "Devre Dışı"
#             if self.gpu_available:
#                 gpu_info = self.get_gpu_details()
#                 if gpu_info['devices']:
#                     dev = gpu_info['devices'][0]
#                     gpu_status = (
#                         f"{dev.name} - "
#                         f"{dev.allocated_mb:.0f}/{dev.total_memory_mb:.0f} MB"
#                     )

#             # System health
#             system_health = "Veri yok"
#             if 'system' in self.tools:
#                 try:
#                     system_health = self.tools['system'].get_status_summary()
#                 except Exception:
#                     pass

#             # Code files count
#             code_files_count = 0
#             if 'code' in self.tools:
#                 try:
#                     files = self.tools['code'].list_files(pattern="*.py")
#                     if files and "Bulunamadı" not in files:
#                         code_files_count = len(files.split('\n'))
#                 except Exception:
#                     pass

#             # Issues collection
#             issues = []
#             if missing_dirs:
#                 issues.append(f"Eksik dizinler: {', '.join(missing_dirs)}")

#             if not self.gpu_available and Config.USE_GPU:
#                 issues.append("GPU bekleniyor ama bulunamadı")

#             # Create audit
#             audit = SystemAudit(
#                 timestamp=datetime.now(),
#                 project_structure_ok=structure_ok,
#                 missing_directories=missing_dirs,
#                 gpu_status=gpu_status,
#                 system_health=system_health,
#                 code_files_count=code_files_count,
#                 issues=issues
#             )

#             self.last_audit = audit
#             self.metrics.system_audits += 1

#             return audit

#     def format_audit_report(self, audit: SystemAudit) -> str:
#         """Audit raporunu formatla"""
#         lines = [
#             f"🛠️ {Config.PROJECT_NAME} TEKNİK DENETİM RAPORU",
#             f"Zaman: {audit.timestamp.strftime('%d.%m.%Y %H:%M:%S')}",
#             "═" * 50,
#             ""
#         ]

#         # Structure
#         if audit.project_structure_ok:
#             lines.append("✅ Proje yapısı doğrulanmış ve standartlara uygun")
#         else:
#             lines.append(f"❌ Eksik dizinler: {', '.join(audit.missing_directories)}")

#         # GPU
#         lines.append(f"\n--- GPU ANALİZİ ---")
#         lines.append(f"Durum: {audit.gpu_status}")

#         # System
#         lines.append(f"\n--- SİSTEM SAĞLIĞI ---")
#         lines.append(audit.system_health)

#         # Code
#         lines.append(f"\n--- KOD TABANI ---")
#         lines.append(f"Python dosyaları: {audit.code_files_count}")

#         # Issues
#         if audit.issues:
#             lines.append(f"\n--- SORUNLAR ---")
#             for issue in audit.issues:
#                 lines.append(f"⚠️ {issue}")
#         else:
#             lines.append("\n✅ Kritik sorun tespit edilmedi")

#         lines.append("═" * 50)

#         return "\n".join(lines)

#     # ───────────────────────────────────────────────────────────
#     # CODE OPERATIONS
#     # ───────────────────────────────────────────────────────────

#     def read_source_code(self, filepath: str) -> str:
#         """
#         Kaynak kodu oku
#         Herkes okuyabilir (kısıtlı modda da okuma izni var)

#         Args:
#             filepath: Dosya yolu

#         Returns:
#             Dosya içeriği
#         """
#         if 'code' not in self.tools:
#             return "❌ CodeManager yüklenemedi"

#         with self.lock:
#             try:
#                 content = self.tools['code'].read_file(filepath)
#                 self.metrics.files_read += 1
#                 return content
#             except Exception as e:
#                 logger.error(f"Dosya okuma hatası ({filepath}): {e}")
#                 return f"❌ Okuma hatası: {str(e)[:100]}"

#     def write_source_code(self, filepath: str, content: str) -> str:
#         """
#         Kaynak kod yaz (validation ile)
#         Sadece sandbox ve full modda çalışır.

#         Args:
#             filepath: Dosya yolu
#             content: İçerik

#         Returns:
#             Sonuç mesajı
#         """
#         if not self._check_write_permission():
#             return self._denied_message("dosya yazma")

#         if 'code' not in self.tools:
#             return "❌ CodeManager aktif değil"

#         with self.lock:
#             # Python syntax check
#             if filepath.endswith('.py'):
#                 syntax_check = self.check_python_syntax(content)
#                 if not syntax_check["valid"]:
#                     logger.error(f"Sözdizimi hatası engellendi: {filepath}")
#                     return (
#                         f"❌ KAYIT REDDEDİLDİ\n"
#                         f"Sözdizimi hatası:\n{syntax_check['error']}"
#                     )

#             # JSON validity check
#             if filepath.endswith('.json'):
#                 json_check = self.check_json_validity(content)
#                 if not json_check["valid"]:
#                     return (
#                         f"❌ KAYIT REDDEDİLDİ\n"
#                         f"Geçersiz JSON:\n{json_check['error']}"
#                     )

#             # Write file
#             try:
#                 result = self.tools['code'].save_file(filepath, content)
#                 self.metrics.files_written += 1
#                 return result
#             except Exception as e:
#                 logger.error(f"Dosya yazma hatası ({filepath}): {e}")
#                 return f"❌ Yazma hatası: {str(e)[:100]}"

#     def list_project_files(
#         self,
#         pattern: str = "*",
#         subdirectory: str = ""
#     ) -> str:
#         """
#         Proje dosyalarını listele
#         Tüm erişim seviyelerinde çalışır (salt okunur işlem).

#         Args:
#             pattern: Dosya pattern'i (örn: "*.py", "*.json", "*")
#             subdirectory: Alt dizin filtresi (boş = tüm proje)

#         Returns:
#             Formatlanmış dosya listesi
#         """
#         if 'code' not in self.tools or self.tools['code'] is None:
#             return "❌ CodeManager aktif değil"

#         with self.lock:
#             try:
#                 files_str = self.tools['code'].list_files(pattern=pattern)

#                 if not files_str or "Eşleşen" in files_str or "❌" in files_str:
#                     return f"📂 '{pattern}' eşleşen dosya bulunamadı"

#                 files = [f for f in files_str.split('\n') if f.strip()]

#                 # Alt dizin filtresi
#                 if subdirectory:
#                     norm_sub = subdirectory.rstrip('/') + '/'
#                     files = [f for f in files if f.startswith(norm_sub)]
#                     if not files:
#                         return f"📂 '{subdirectory}' dizininde '{pattern}' dosyası yok"

#                 result_lines = [
#                     f"📂 PROJE DOSYALARI | Pattern: {pattern}",
#                     f"🔐 Erişim Seviyesi: {self.access_level}",
#                     f"📊 Toplam: {len(files)} dosya",
#                     "─" * 50
#                 ]
#                 result_lines.extend(files)
#                 result_lines.append("─" * 50)

#                 return "\n".join(result_lines)

#             except Exception as e:
#                 logger.error(f"Dosya listeleme hatası: {e}")
#                 return f"❌ Listeleme hatası: {str(e)[:100]}"

#     def get_file_tree(self, max_depth: int = 3) -> str:
#         """
#         Proje dizin ağacını görsel olarak göster.
#         Tüm erişim seviyelerinde çalışır (salt okunur).

#         Args:
#             max_depth: Maksimum alt dizin derinliği (varsayılan: 3)

#         Returns:
#             Ağaç yapısı (metin formatı)
#         """
#         EXCLUDE_DIRS = {
#             '.git', '__pycache__', 'backups', 'venv', 'env',
#             'node_modules', 'faces', 'voices', '.pytest_cache',
#             'dist', 'build', '.vscode', 'lotus_vector_db'
#         }

#         try:
#             work_dir = Config.WORK_DIR
#             lines = [
#                 f"📁 {Config.PROJECT_NAME} - PROJE YAPISI",
#                 f"📍 Kök dizin: {work_dir}",
#                 f"🔐 Erişim: {self.access_level}",
#                 "═" * 50
#             ]

#             def _build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
#                 if depth > max_depth:
#                     return
#                 try:
#                     items = sorted(
#                         path.iterdir(),
#                         key=lambda x: (x.is_file(), x.name.lower())
#                     )
#                 except PermissionError:
#                     return

#                 items = [
#                     i for i in items
#                     if i.name not in EXCLUDE_DIRS and not i.name.startswith('.')
#                 ]

#                 for i, item in enumerate(items):
#                     is_last = (i == len(items) - 1)
#                     connector = "└── " if is_last else "├── "
#                     child_prefix = "    " if is_last else "│   "

#                     if item.is_dir():
#                         lines.append(f"{prefix}{connector}📁 {item.name}/")
#                         _build_tree(item, prefix + child_prefix, depth + 1)
#                     else:
#                         try:
#                             size_kb = round(item.stat().st_size / 1024, 1)
#                             lines.append(
#                                 f"{prefix}{connector}📄 {item.name} ({size_kb} KB)"
#                             )
#                         except Exception:
#                             lines.append(f"{prefix}{connector}📄 {item.name}")

#             _build_tree(work_dir)
#             lines.append("═" * 50)

#             # Yazma yetkisi özeti
#             if self.access_level == AccessLevel.FULL:
#                 lines.append("✅ Tüm dosyalarda okuma + yazma + silme yetkisi")
#             elif self.access_level == AccessLevel.SANDBOX:
#                 lines.append("📦 Okuma + yazma yetkisi (silme yok)")
#             else:
#                 lines.append("🔒 Sadece okuma yetkisi")

#             return "\n".join(lines)

#         except Exception as e:
#             logger.error(f"Ağaç oluşturma hatası: {e}")
#             return f"❌ Dizin ağacı hatası: {str(e)[:100]}"

#     def check_python_syntax(self, code_content: str) -> Dict[str, Any]:
#         """
#         Python syntax kontrolü
#         Herkes kullanabilir.

#         Args:
#             code_content: Kod içeriği

#         Returns:
#             Validation sonucu
#         """
#         self.metrics.syntax_checks += 1

#         try:
#             ast.parse(code_content)
#             return {"valid": True, "error": None}

#         except SyntaxError as e:
#             self.metrics.syntax_errors += 1
#             error_msg = f"Satır {e.lineno}: {e.msg}"
#             return {"valid": False, "error": error_msg}

#         except Exception as e:
#             self.metrics.syntax_errors += 1
#             return {"valid": False, "error": str(e)}

#     def check_json_validity(self, json_content: str) -> Dict[str, Any]:
#         """
#         JSON validity kontrolü
#         Herkes kullanabilir.

#         Args:
#             json_content: JSON içeriği

#         Returns:
#             Validation sonucu
#         """
#         try:
#             json.loads(json_content)
#             return {"valid": True, "error": None}

#         except json.JSONDecodeError as e:
#             error_msg = f"Satır {e.lineno}: {e.msg}"
#             return {"valid": False, "error": error_msg}

#         except Exception as e:
#             return {"valid": False, "error": str(e)}

#     # ───────────────────────────────────────────────────────────
#     # ERROR ANALYSIS
#     # ───────────────────────────────────────────────────────────

#     def analyze_error(self, error_traceback: str) -> ErrorAnalysis:
#         """
#         Hata analizi yap
#         Herkes kullanabilir.

#         Args:
#             error_traceback: Hata traceback'i

#         Returns:
#             ErrorAnalysis objesi
#         """
#         if not error_traceback:
#             return ErrorAnalysis(
#                 error_type=ErrorType.UNKNOWN,
#                 severity=ErrorSeverity.INFO,
#                 diagnosis="Analiz edilecek hata verisi yok",
#                 solution="",
#                 traceback=""
#             )

#         self.metrics.error_analyses += 1

#         # Error type detection
#         error_type = self._detect_error_type(error_traceback)

#         # Severity assessment
#         severity = self._assess_error_severity(error_type)

#         # Diagnosis & solution
#         diagnosis, solution = self._generate_diagnosis_and_solution(
#             error_type,
#             error_traceback
#         )

#         return ErrorAnalysis(
#             error_type=error_type,
#             severity=severity,
#             diagnosis=diagnosis,
#             solution=solution,
#             traceback=error_traceback
#         )

#     def _detect_error_type(self, traceback: str) -> ErrorType:
#         """Error tipini tespit et"""
#         for pattern, error_type in self.ERROR_PATTERNS.items():
#             if pattern in traceback:
#                 return error_type

#         return ErrorType.RUNTIME_ERROR

#     def _assess_error_severity(self, error_type: ErrorType) -> ErrorSeverity:
#         """Hata ciddiyetini değerlendir"""
#         severity_map = {
#             ErrorType.GPU_OOM: ErrorSeverity.CRITICAL,
#             ErrorType.IMPORT_ERROR: ErrorSeverity.HIGH,
#             ErrorType.FILE_NOT_FOUND: ErrorSeverity.MEDIUM,
#             ErrorType.SYNTAX_ERROR: ErrorSeverity.HIGH,
#             ErrorType.RUNTIME_ERROR: ErrorSeverity.MEDIUM,
#             ErrorType.LOGIC_ERROR: ErrorSeverity.LOW,
#             ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
#         }

#         return severity_map.get(error_type, ErrorSeverity.MEDIUM)

#     def _generate_diagnosis_and_solution(
#         self,
#         error_type: ErrorType,
#         traceback: str
#     ) -> Tuple[str, str]:
#         """Teşhis ve çözüm üret"""
#         solutions = {
#             ErrorType.GPU_OOM: (
#                 "GPU bellek yetersizliği tespit edildi",
#                 "optimize_gpu_memory() çalıştır ve model batch size'ı düşür"
#             ),
#             ErrorType.IMPORT_ERROR: (
#                 "Eksik kütüphane bağımlılığı",
#                 "pip install ile gerekli paketi yükle"
#             ),
#             ErrorType.FILE_NOT_FOUND: (
#                 "Hatalı dosya yolu veya eksik dosya",
#                 "Config.WORK_DIR ve dosya yollarını kontrol et"
#             ),
#             ErrorType.SYNTAX_ERROR: (
#                 "Python sözdizimi hatası",
#                 "Kodu gözden geçir ve sözdizimi hatalarını düzelt"
#             ),
#             ErrorType.RUNTIME_ERROR: (
#                 "Çalışma zamanı hatası",
#                 "Traceback'i incele ve hata kaynağını bul"
#             ),
#             ErrorType.LOGIC_ERROR: (
#                 "Mantık hatası",
#                 "Algoritma ve iş mantığını gözden geçir"
#             ),
#             ErrorType.UNKNOWN: (
#                 "Bilinmeyen hata tipi",
#                 "Manuel kod incelemesi gerekli"
#             )
#         }

#         diagnosis, solution = solutions.get(
#             error_type,
#             ("Karmaşık hata", "Detaylı analiz gerekli")
#         )

#         # GPU OOM için otomatik optimizasyon (sadece full modda)
#         if error_type == ErrorType.GPU_OOM and self._check_system_permission():
#             self.optimize_gpu_memory()

#         return diagnosis, solution

#     # ───────────────────────────────────────────────────────────
#     # CONTEXT GENERATION
#     # ───────────────────────────────────────────────────────────

#     def get_context_data(self) -> str:
#         """
#         Sidar için teknik bağlam
#         Erişim seviyesine göre dosya bilgileri dahil edilir.

#         Returns:
#             Context string
#         """
#         context_parts = ["\n[👨‍💻 SİDAR TEKNİK ALTYAPI RAPORU]"]

#         with self.lock:
#             # System info (herkese açık)
#             sys_info = (
#                 f"OS: {platform.system()} {platform.release()} | "
#                 f"Python: {platform.python_version()}"
#             )
#             context_parts.append(f"🖥️ SİSTEM: {sys_info}")

#             # GPU status (herkese açık)
#             gpu_data = self.get_gpu_details()
#             if gpu_data["available"] and gpu_data["devices"]:
#                 dev = gpu_data['devices'][0]
#                 gpu_status = (
#                     f"GPU: AKTİF | {dev.name} | "
#                     f"Kullanım: {dev.allocated_mb:.0f}MB"
#                 )
#             else:
#                 gpu_status = "GPU: Devre Dışı"
#             context_parts.append(f"⚙️ DONANIM: {gpu_status}")

#             # Aktif LLM modeli
#             if Config.AI_PROVIDER == "ollama":
#                 context_parts.append(
#                     f"🤖 AKTİF MODEL: {Config.CODING_MODEL} (CODING_MODEL)"
#                 )

#             # System health (herkese açık)
#             if 'system' in self.tools:
#                 try:
#                     health = self.tools['system'].get_status_summary()
#                     context_parts.append(f"\n📊 SAĞLIK: {health}")
#                 except Exception:
#                     pass

#             # Kod tabanı detayları (tüm erişim seviyelerine açık)
#             if 'code' in self.tools and self.tools['code'] is not None:
#                 try:
#                     py_files_str = self.tools['code'].list_files(pattern="*.py")
#                     py_count = 0
#                     if py_files_str and "Eşleşen" not in py_files_str and "❌" not in py_files_str:
#                         py_count = len([f for f in py_files_str.split('\n') if f.strip()])

#                     json_files_str = self.tools['code'].list_files(pattern="*.json")
#                     json_count = 0
#                     if json_files_str and "Eşleşen" not in json_files_str and "❌" not in json_files_str:
#                         json_count = len([f for f in json_files_str.split('\n') if f.strip()])

#                     context_parts.append(
#                         f"\n📂 KOD TABANI: {py_count} Python + {json_count} JSON dosyası"
#                     )
#                     context_parts.append(
#                         f"   ↳ list_project_files() veya get_file_tree() ile tam listeye eriş"
#                     )
#                 except Exception:
#                     pass

#             # Erişim seviyesi ve dosya yetki özeti
#             access_label = {
#                 AccessLevel.RESTRICTED: "🔒 Kısıtlı (salt okunur)",
#                 AccessLevel.SANDBOX:    "📦 Sandbox (okuma + yazma)",
#                 AccessLevel.FULL:       "⚡ Tam Erişim (okuma + yazma + silme)",
#             }.get(self.access_level, self.access_level)
#             context_parts.append(f"\n🔐 GÜVENLİ DOSYA MODU: {access_label}")

#         return "\n".join(context_parts)

#     # ───────────────────────────────────────────────────────────
#     # ARCHITECTURE SUGGESTIONS
#     # ───────────────────────────────────────────────────────────

#     def get_architecture_suggestion(self) -> str:
#         """
#         Mimari öneri
#         Herkese açık (bilgi amaçlı)

#         Returns:
#             Öneri metni
#         """
#         gpu_advice = (
#             "Sistem GPU destekli, performans iyi."
#             if self.gpu_available
#             else "GPU eksikliği var, donanım takviyesi önerilir."
#         )

#         return (
#             f"🚀 SİDAR MİMARİ TAVSİYESİ:\n\n"
#             f"{gpu_advice}\n\n"
#             "Proje geliştikçe öneriler:\n"
#             "1. Event Bus yapısı (agent iletişimi için)\n"
#             "2. Model Quantization (GPU yükü için)\n"
#             "3. Asenkron task queue (paralel işlemler için)\n"
#             "4. Redis cache (performans için)\n"
#             "5. Docker containerization (deployment için)"
#         )

#     # ───────────────────────────────────────────────────────────
#     # SYSTEM PROMPT (Eski, artık kullanılmıyor olabilir ama koruyalım)
#     # ───────────────────────────────────────────────────────────

#     def get_system_prompt(self) -> str:
#         """
#         Sidar karakter tanımı (LLM için)

#         Returns:
#             System prompt
#         """
#         gpu_info = (
#             f"Sistemde {self.gpu_count} GPU birimi tespit edildi"
#             if self.gpu_available
#             else "GPU bulunamadı, CPU üzerinden işlem yapılıyor"
#         )

#         # Aktif model bilgisi
#         if Config.AI_PROVIDER == "ollama":
#             model_info = f"Aktif Model: {Config.CODING_MODEL} (Kod odaklı)"
#         else:
#             model_info = f"Aktif Model: {Config.GEMINI_MODEL_DEFAULT}"

#         # Erişim seviyesine göre dosya yetki özeti
#         access_display = {
#             AccessLevel.RESTRICTED: "🔒 Kısıtlı – Dosya görüntüleme ve listeleme (salt okunur)",
#             AccessLevel.SANDBOX: "📦 Sandbox – Dosya listeleme + okuma + yazma (silme yok)",
#             AccessLevel.FULL: "⚡ Tam Erişim – Listeleme, okuma, yazma ve silme dahil tüm yetkiler",
#         }.get(self.access_level, self.access_level)

#         # Kullanılabilir dosya işlemleri
#         file_ops = "- Proje dosyalarını listele: list_project_files(pattern, subdirectory)\n"
#         file_ops += "- Dizin ağacını görüntüle: get_file_tree(max_depth)\n"
#         file_ops += "- Dosya oku: read_source_code(filepath)\n"
#         if self.access_level in [AccessLevel.SANDBOX, AccessLevel.FULL]:
#             file_ops += "- Dosya yaz (syntax doğrulamalı): write_source_code(filepath, content)\n"
#         if self.access_level == AccessLevel.FULL:
#             file_ops += "- Dosya sil (yedekli): tools['code'].delete_file(filepath)\n"

#         return (
#             f"Sen {Config.PROJECT_NAME} sisteminin Baş Mühendisi ve "
#             f"Yazılım Mimarı SİDAR'sın.\n\n"

#             "KARAKTER:\n"
#             "- Son derece disiplinli\n"
#             "- Teknik detaylara aşırı hakim\n"
#             "- Titiz ve çözüm odaklı\n"
#             "- Modern standartlara sadık (PEP 8)\n"
#             "- Güvenlik ve modülerliğe önem veren\n\n"

#             f"MİSYON:\n"
#             f"- Kod yapısını korumak\n"
#             f"- Hataları ayıklamak\n"
#             f"- Donanımı ({gpu_info}) verimli kullanmak\n"
#             f"- Sistemi optimize etmek\n"
#             f"- {model_info}\n\n"

#             f"GÜVENLİ DOSYA MODU:\n"
#             f"Mevcut erişim seviyesi: {access_display}\n\n"
#             f"Kullanılabilir dosya işlemleri:\n"
#             f"{file_ops}\n"

#             "KURALLAR:\n"
#             "- Net ve profesyonel rapor sun\n"
#             "- Kod yazarken standartlara uy\n"
#             "- Sorun gördüğünde çözüm kodla\n"
#             "- Şikayet etme, analiz et ve çöz\n"
#             "- Proaktif ol, sorunları önceden gör\n"
#             "- Dosya işlemi öncesi erişim seviyeni kontrol et\n"
#         )

#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────

#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Sidar metrikleri
#         Herkes okuyabilir.

#         Returns:
#             Metrik dictionary
#         """
#         syntax_error_rate = 0.0
#         if self.metrics.syntax_checks > 0:
#             syntax_error_rate = (
#                 self.metrics.syntax_errors /
#                 self.metrics.syntax_checks * 100
#             )

#         return {
#             "agent_name": self.agent_name,
#             "device": DEVICE_TYPE,
#             "gpu_count": self.gpu_count,
#             "active_model": (
#                 Config.CODING_MODEL
#                 if Config.AI_PROVIDER == "ollama"
#                 else Config.GEMINI_MODEL_DEFAULT
#             ),
#             "access_level": self.access_level,
#             "files_read": self.metrics.files_read,
#             "files_written": self.metrics.files_written,
#             "syntax_checks": self.metrics.syntax_checks,
#             "syntax_errors": self.metrics.syntax_errors,
#             "syntax_error_rate": round(syntax_error_rate, 2),
#             "gpu_optimizations": self.metrics.gpu_optimizations,
#             "error_analyses": self.metrics.error_analyses,
#             "system_audits": self.metrics.system_audits,
#             "last_audit": (
#                 self.last_audit.timestamp.isoformat()
#                 if self.last_audit else None
#             )
#         }




# """
# LotusAI Sidar Agent
# Sürüm: 2.5.5 (Eklendi: Erişim Seviyesi Desteği)
# Açıklama: Baş mühendis ve yazılım mimarı

# Sorumluluklar:
# - Kod tabanı yönetimi
# - Sistem sağlığı analizi
# - GPU optimizasyonu
# - Hata analizi ve teşhis
# - Mimari öneriler
# - Kod kalite kontrolü
# """

# import os
# import platform
# import logging
# import json
# import ast
# import threading
# import gc
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Sidar")


# # ═══════════════════════════════════════════════════════════════
# # TORCH (GPU)
# # ═══════════════════════════════════════════════════════════════
# HAS_TORCH = False
# DEVICE_TYPE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
#         HAS_TORCH = True
        
#         if torch.cuda.is_available():
#             DEVICE_TYPE = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             DEVICE_TYPE = "mps"
#     except ImportError:
#         logger.warning("⚠️ Sidar: Config GPU açık ama torch yok")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class CodeQuality(Enum):
#     """Kod kalitesi seviyeleri"""
#     EXCELLENT = "Mükemmel"
#     GOOD = "İyi"
#     ACCEPTABLE = "Kabul Edilebilir"
#     POOR = "Zayıf"
#     CRITICAL = "Kritik"


# class ErrorSeverity(Enum):
#     """Hata ciddiyeti"""
#     CRITICAL = "Kritik"
#     HIGH = "Yüksek"
#     MEDIUM = "Orta"
#     LOW = "Düşük"
#     INFO = "Bilgi"


# class ErrorType(Enum):
#     """Hata tipleri"""
#     GPU_OOM = "gpu_oom"
#     IMPORT_ERROR = "import_error"
#     FILE_NOT_FOUND = "file_not_found"
#     SYNTAX_ERROR = "syntax_error"
#     RUNTIME_ERROR = "runtime_error"
#     LOGIC_ERROR = "logic_error"
#     UNKNOWN = "unknown"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class GPUDevice:
#     """GPU cihaz bilgisi"""
#     id: int
#     name: str
#     total_memory_mb: float
#     allocated_mb: float
#     reserved_mb: float
#     capability: float = 0.0


# @dataclass
# class SystemAudit:
#     """Sistem denetim raporu"""
#     timestamp: datetime
#     project_structure_ok: bool
#     missing_directories: List[str]
#     gpu_status: str
#     system_health: str
#     code_files_count: int
#     issues: List[str]


# @dataclass
# class ErrorAnalysis:
#     """Hata analizi"""
#     error_type: ErrorType
#     severity: ErrorSeverity
#     diagnosis: str
#     solution: str
#     traceback: str


# @dataclass
# class SidarMetrics:
#     """Sidar metrikleri"""
#     files_read: int = 0
#     files_written: int = 0
#     syntax_checks: int = 0
#     syntax_errors: int = 0
#     gpu_optimizations: int = 0
#     error_analyses: int = 0
#     system_audits: int = 0


# # ═══════════════════════════════════════════════════════════════
# # SIDAR AGENT
# # ═══════════════════════════════════════════════════════════════
# class SidarAgent:
#     """
#     Sidar (Baş Mühendis & Yazılım Mimarı)
    
#     Yetenekler:
#     - Kod tabanı yönetimi: Okuma, yazma, analiz
#     - Sistem sağlığı: CPU, RAM, GPU izleme
#     - Hata analizi: Traceback teşhisi ve çözüm önerileri
#     - Güvenli geliştirme: Syntax validation
#     - Mimari öngörü: Yapısal iyileştirme tavsiyeleri
#     - GPU optimizasyonu: VRAM yönetimi
    
#     Sidar, sistemin "teknik beyin"idir ve kod kalitesinden taviz vermez.
#     """
    
#     # Critical project directories
#     CRITICAL_DIRS = ["agents", "core", "managers", "static", "templates"]
    
#     # Error patterns
#     ERROR_PATTERNS = {
#         "CUDA out of memory": ErrorType.GPU_OOM,
#         "CUDA error": ErrorType.GPU_OOM,
#         "ImportError": ErrorType.IMPORT_ERROR,
#         "ModuleNotFoundError": ErrorType.IMPORT_ERROR,
#         "FileNotFoundError": ErrorType.FILE_NOT_FOUND,
#         "SyntaxError": ErrorType.SYNTAX_ERROR
#     }
    
#     def __init__(self, tools_dict: Dict[str, Any], access_level: str = "sandbox"):
#         """
#         Sidar başlatıcı
        
#         Args:
#             tools_dict: Engine'den gelen tool'lar
#             access_level: Erişim seviyesi (restricted, sandbox, full)
#         """
#         self.tools = tools_dict
#         self.agent_name = "SİDAR"
#         self.access_level = access_level
        
#         # Thread safety
#         self.lock = threading.RLock()
        
#         # Hardware
#         self.gpu_available = (DEVICE_TYPE != "cpu")
#         self.gpu_count = 0
        
#         if self.gpu_available and HAS_TORCH and DEVICE_TYPE == "cuda":
#             try:
#                 self.gpu_count = torch.cuda.device_count()
#             except Exception:
#                 pass
        
#         # Metrics
#         self.metrics = SidarMetrics()
        
#         # Audit history
#         self.last_audit: Optional[SystemAudit] = None
        
#         gpu_status = "AKTİF" if self.gpu_available else "DEVRE DIŞI"
#         logger.info(
#             f"👨‍💻 {self.agent_name} Teknik Liderlik modülü başlatıldı "
#             f"(GPU: {gpu_status}, Erişim: {self.access_level})"
#         )
    
#     # ───────────────────────────────────────────────────────────
#     # GPU MANAGEMENT
#     # ───────────────────────────────────────────────────────────
    
#     def get_gpu_details(self) -> Dict[str, Any]:
#         """
#         GPU detaylarını getir
        
#         Returns:
#             GPU bilgileri dict
#         """
#         details = {"available": False, "devices": []}
        
#         if not self.gpu_available or not HAS_TORCH:
#             return details
        
#         try:
#             details["available"] = True
            
#             if DEVICE_TYPE == "cuda":
#                 for i in range(self.gpu_count):
#                     props = torch.cuda.get_device_properties(i)
#                     mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
#                     mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
                    
#                     device = GPUDevice(
#                         id=i,
#                         name=props.name,
#                         total_memory_mb=props.total_memory / (1024 ** 2),
#                         allocated_mb=round(mem_alloc, 2),
#                         reserved_mb=round(mem_reserved, 2),
#                         capability=props.major + props.minor / 10
#                     )
                    
#                     details["devices"].append(device)
            
#             elif DEVICE_TYPE == "mps":
#                 device = GPUDevice(
#                     id=0,
#                     name="Apple Silicon (MPS)",
#                     total_memory_mb=0.0,  # Unified memory
#                     allocated_mb=0.0,
#                     reserved_mb=0.0
#                 )
#                 details["devices"].append(device)
        
#         except Exception as e:
#             logger.error(f"GPU detay hatası: {e}")
        
#         return details
    
#     def optimize_gpu_memory(self) -> str:
#         """
#         GPU belleğini optimize et
        
#         Returns:
#             Sonuç mesajı
#         """
#         # Erişim kontrolü: Sadece full modda izin ver
#         if self.access_level != AccessLevel.FULL:
#             return "⛔ Bu işlem için Tam Erişim (FULL) gerekiyor. Mevcut erişim seviyeniz: " + self.access_level
        
#         if not self.gpu_available or not HAS_TORCH:
#             return "⚠️ Optimizasyon atlandı: GPU aktif değil"
        
#         with self.lock:
#             try:
#                 savings = 0.0
                
#                 if DEVICE_TYPE == "cuda":
#                     initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     final_mem = torch.cuda.memory_allocated() / (1024 ** 2)
#                     savings = round(initial_mem - final_mem, 2)
                
#                 elif DEVICE_TYPE == "mps":
#                     gc.collect()
#                     try:
#                         torch.mps.empty_cache()
#                     except Exception:
#                         pass
                
#                 self.metrics.gpu_optimizations += 1
                
#                 return (
#                     f"✅ GPU optimizasyonu tamamlandı. "
#                     f"Serbest bırakılan VRAM: {savings:.2f} MB"
#                 )
            
#             except Exception as e:
#                 logger.error(f"GPU optimizasyon hatası: {e}")
#                 return f"❌ Optimizasyon hatası: {str(e)[:50]}"
    
#     # ───────────────────────────────────────────────────────────
#     # SYSTEM AUDIT
#     # ───────────────────────────────────────────────────────────
    
#     def perform_system_audit(self) -> SystemAudit:
#         """
#         Sistem denetimi yap
        
#         Returns:
#             SystemAudit objesi
#         """
#         with self.lock:
#             # Directory structure check
#             work_dir = Config.WORK_DIR
#             missing_dirs = [
#                 d for d in self.CRITICAL_DIRS
#                 if not (Path(work_dir) / d).exists()
#             ]
            
#             structure_ok = len(missing_dirs) == 0
            
#             # GPU status
#             gpu_status = "Devre Dışı"
#             if self.gpu_available:
#                 gpu_info = self.get_gpu_details()
#                 if gpu_info['devices']:
#                     dev = gpu_info['devices'][0]
#                     gpu_status = (
#                         f"{dev.name} - "
#                         f"{dev.allocated_mb:.0f}/{dev.total_memory_mb:.0f} MB"
#                     )
            
#             # System health
#             system_health = "Veri yok"
#             if 'system' in self.tools:
#                 try:
#                     system_health = self.tools['system'].get_status_summary()
#                 except Exception:
#                     pass
            
#             # Code files count
#             code_files_count = 0
#             if 'code' in self.tools:
#                 try:
#                     files = self.tools['code'].list_files(pattern="*.py")
#                     if files and "Bulunamadı" not in files:
#                         code_files_count = len(files.split('\n'))
#                 except Exception:
#                     pass
            
#             # Issues collection
#             issues = []
#             if missing_dirs:
#                 issues.append(f"Eksik dizinler: {', '.join(missing_dirs)}")
            
#             if not self.gpu_available and Config.USE_GPU:
#                 issues.append("GPU bekleniyor ama bulunamadı")
            
#             # Create audit
#             audit = SystemAudit(
#                 timestamp=datetime.now(),
#                 project_structure_ok=structure_ok,
#                 missing_directories=missing_dirs,
#                 gpu_status=gpu_status,
#                 system_health=system_health,
#                 code_files_count=code_files_count,
#                 issues=issues
#             )
            
#             self.last_audit = audit
#             self.metrics.system_audits += 1
            
#             return audit
    
#     def format_audit_report(self, audit: SystemAudit) -> str:
#         """Audit raporunu formatla"""
#         lines = [
#             f"🛠️ {Config.PROJECT_NAME} TEKNİK DENETİM RAPORU",
#             f"Zaman: {audit.timestamp.strftime('%d.%m.%Y %H:%M:%S')}",
#             "═" * 50,
#             ""
#         ]
        
#         # Structure
#         if audit.project_structure_ok:
#             lines.append("✅ Proje yapısı doğrulanmış ve standartlara uygun")
#         else:
#             lines.append(f"❌ Eksik dizinler: {', '.join(audit.missing_directories)}")
        
#         # GPU
#         lines.append(f"\n--- GPU ANALİZİ ---")
#         lines.append(f"Durum: {audit.gpu_status}")
        
#         # System
#         lines.append(f"\n--- SİSTEM SAĞLIĞI ---")
#         lines.append(audit.system_health)
        
#         # Code
#         lines.append(f"\n--- KOD TABANI ---")
#         lines.append(f"Python dosyaları: {audit.code_files_count}")
        
#         # Issues
#         if audit.issues:
#             lines.append(f"\n--- SORUNLAR ---")
#             for issue in audit.issues:
#                 lines.append(f"⚠️ {issue}")
#         else:
#             lines.append("\n✅ Kritik sorun tespit edilmedi")
        
#         lines.append("═" * 50)
        
#         return "\n".join(lines)
    
#     # ───────────────────────────────────────────────────────────
#     # CODE OPERATIONS
#     # ───────────────────────────────────────────────────────────
    
#     def read_source_code(self, filepath: str) -> str:
#         """
#         Kaynak kodu oku
        
#         Args:
#             filepath: Dosya yolu
        
#         Returns:
#             Dosya içeriği
#         """
#         # Kısıtlı modda sadece belirli dosyalar okunabilir mi? 
#         # Şimdilik herkes okuyabilir, ama istenirse kısıtlanabilir.
#         if self.access_level == AccessLevel.RESTRICTED:
#             # Sadece belirli dizinlerdeki dosyaları okumaya izin ver (örneğin logs)
#             if not filepath.startswith("logs/") and not filepath.startswith("data/"):
#                 return "⛔ Kısıtlı modda bu dosyayı okuma izniniz yok."
        
#         if 'code' not in self.tools:
#             return "❌ CodeManager yüklenemedi"
        
#         with self.lock:
#             try:
#                 content = self.tools['code'].read_file(filepath)
#                 self.metrics.files_read += 1
#                 return content
#             except Exception as e:
#                 logger.error(f"Dosya okuma hatası ({filepath}): {e}")
#                 return f"❌ Okuma hatası: {str(e)[:100]}"
    
#     def write_source_code(self, filepath: str, content: str) -> str:
#         """
#         Kaynak kod yaz (validation ile)
        
#         Args:
#             filepath: Dosya yolu
#             content: İçerik
        
#         Returns:
#             Sonuç mesajı
#         """
#         # Erişim kontrolü: Kısıtlı modda yazma yasak
#         if self.access_level == AccessLevel.RESTRICTED:
#             return "⛔ Kısıtlı modda dosya yazma izniniz yok."
        
#         # Sandbox modunda sadece belirli dizinlere yazmaya izin ver
#         if self.access_level == AccessLevel.SANDBOX:
#             allowed_dirs = ["uploads/", "data/", "logs/", "temp/"]
#             if not any(filepath.startswith(d) for d in allowed_dirs):
#                 return f"⛔ Sandbox modunda sadece şu dizinlere yazabilirsiniz: {', '.join(allowed_dirs)}"
        
#         if 'code' not in self.tools:
#             return "❌ CodeManager aktif değil"
        
#         with self.lock:
#             # Python syntax check
#             if filepath.endswith('.py'):
#                 syntax_check = self.check_python_syntax(content)
#                 if not syntax_check["valid"]:
#                     logger.error(f"Sözdizimi hatası engellendi: {filepath}")
#                     return (
#                         f"❌ KAYIT REDDEDİLDİ\n"
#                         f"Sözdizimi hatası:\n{syntax_check['error']}"
#                     )
            
#             # JSON validity check
#             if filepath.endswith('.json'):
#                 json_check = self.check_json_validity(content)
#                 if not json_check["valid"]:
#                     return (
#                         f"❌ KAYIT REDDEDİLDİ\n"
#                         f"Geçersiz JSON:\n{json_check['error']}"
#                     )
            
#             # Write file
#             try:
#                 result = self.tools['code'].save_file(filepath, content)
#                 self.metrics.files_written += 1
#                 return result
#             except Exception as e:
#                 logger.error(f"Dosya yazma hatası ({filepath}): {e}")
#                 return f"❌ Yazma hatası: {str(e)[:100]}"
    
#     def check_python_syntax(self, code_content: str) -> Dict[str, Any]:
#         """
#         Python syntax kontrolü
        
#         Args:
#             code_content: Kod içeriği
        
#         Returns:
#             Validation sonucu
#         """
#         self.metrics.syntax_checks += 1
        
#         try:
#             ast.parse(code_content)
#             return {"valid": True, "error": None}
        
#         except SyntaxError as e:
#             self.metrics.syntax_errors += 1
#             error_msg = f"Satır {e.lineno}: {e.msg}"
#             return {"valid": False, "error": error_msg}
        
#         except Exception as e:
#             self.metrics.syntax_errors += 1
#             return {"valid": False, "error": str(e)}
    
#     def check_json_validity(self, json_content: str) -> Dict[str, Any]:
#         """
#         JSON validity kontrolü
        
#         Args:
#             json_content: JSON içeriği
        
#         Returns:
#             Validation sonucu
#         """
#         try:
#             json.loads(json_content)
#             return {"valid": True, "error": None}
        
#         except json.JSONDecodeError as e:
#             error_msg = f"Satır {e.lineno}: {e.msg}"
#             return {"valid": False, "error": error_msg}
        
#         except Exception as e:
#             return {"valid": False, "error": str(e)}
    
#     # ───────────────────────────────────────────────────────────
#     # ERROR ANALYSIS
#     # ───────────────────────────────────────────────────────────
    
#     def analyze_error(self, error_traceback: str) -> ErrorAnalysis:
#         """
#         Hata analizi yap
        
#         Args:
#             error_traceback: Hata traceback'i
        
#         Returns:
#             ErrorAnalysis objesi
#         """
#         if not error_traceback:
#             return ErrorAnalysis(
#                 error_type=ErrorType.UNKNOWN,
#                 severity=ErrorSeverity.INFO,
#                 diagnosis="Analiz edilecek hata verisi yok",
#                 solution="",
#                 traceback=""
#             )
        
#         self.metrics.error_analyses += 1
        
#         # Error type detection
#         error_type = self._detect_error_type(error_traceback)
        
#         # Severity assessment
#         severity = self._assess_error_severity(error_type)
        
#         # Diagnosis & solution
#         diagnosis, solution = self._generate_diagnosis_and_solution(
#             error_type,
#             error_traceback
#         )
        
#         return ErrorAnalysis(
#             error_type=error_type,
#             severity=severity,
#             diagnosis=diagnosis,
#             solution=solution,
#             traceback=error_traceback
#         )
    
#     def _detect_error_type(self, traceback: str) -> ErrorType:
#         """Error tipini tespit et"""
#         for pattern, error_type in self.ERROR_PATTERNS.items():
#             if pattern in traceback:
#                 return error_type
        
#         return ErrorType.RUNTIME_ERROR
    
#     def _assess_error_severity(self, error_type: ErrorType) -> ErrorSeverity:
#         """Hata ciddiyetini değerlendir"""
#         severity_map = {
#             ErrorType.GPU_OOM: ErrorSeverity.CRITICAL,
#             ErrorType.IMPORT_ERROR: ErrorSeverity.HIGH,
#             ErrorType.FILE_NOT_FOUND: ErrorSeverity.MEDIUM,
#             ErrorType.SYNTAX_ERROR: ErrorSeverity.HIGH,
#             ErrorType.RUNTIME_ERROR: ErrorSeverity.MEDIUM,
#             ErrorType.LOGIC_ERROR: ErrorSeverity.LOW,
#             ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
#         }
        
#         return severity_map.get(error_type, ErrorSeverity.MEDIUM)
    
#     def _generate_diagnosis_and_solution(
#         self,
#         error_type: ErrorType,
#         traceback: str
#     ) -> Tuple[str, str]:
#         """Teşhis ve çözüm üret"""
#         solutions = {
#             ErrorType.GPU_OOM: (
#                 "GPU bellek yetersizliği tespit edildi",
#                 "optimize_gpu_memory() çalıştır ve model batch size'ı düşür"
#             ),
#             ErrorType.IMPORT_ERROR: (
#                 "Eksik kütüphane bağımlılığı",
#                 "pip install ile gerekli paketi yükle"
#             ),
#             ErrorType.FILE_NOT_FOUND: (
#                 "Hatalı dosya yolu veya eksik dosya",
#                 "Config.WORK_DIR ve dosya yollarını kontrol et"
#             ),
#             ErrorType.SYNTAX_ERROR: (
#                 "Python sözdizimi hatası",
#                 "Kodu gözden geçir ve sözdizimi hatalarını düzelt"
#             ),
#             ErrorType.RUNTIME_ERROR: (
#                 "Çalışma zamanı hatası",
#                 "Traceback'i incele ve hata kaynağını bul"
#             ),
#             ErrorType.LOGIC_ERROR: (
#                 "Mantık hatası",
#                 "Algoritma ve iş mantığını gözden geçir"
#             ),
#             ErrorType.UNKNOWN: (
#                 "Bilinmeyen hata tipi",
#                 "Manuel kod incelemesi gerekli"
#             )
#         }
        
#         diagnosis, solution = solutions.get(
#             error_type,
#             ("Karmaşık hata", "Detaylı analiz gerekli")
#         )
        
#         # GPU OOM için otomatik optimizasyon
#         if error_type == ErrorType.GPU_OOM and self.access_level == AccessLevel.FULL:
#             self.optimize_gpu_memory()
        
#         return diagnosis, solution
    
#     # ───────────────────────────────────────────────────────────
#     # CONTEXT GENERATION
#     # ───────────────────────────────────────────────────────────
    
#     def get_context_data(self) -> str:
#         """
#         Sidar için teknik bağlam
        
#         Returns:
#             Context string
#         """
#         context_parts = ["\n[👨‍💻 SİDAR TEKNİK ALTYAPI RAPORU]"]
        
#         with self.lock:
#             # System info
#             sys_info = (
#                 f"OS: {platform.system()} {platform.release()} | "
#                 f"Python: {platform.python_version()}"
#             )
            
#             # GPU status
#             gpu_data = self.get_gpu_details()
#             if gpu_data["available"] and gpu_data["devices"]:
#                 dev = gpu_data['devices'][0]
#                 gpu_status = (
#                     f"GPU: AKTİF | {dev.name} | "
#                     f"Kullanım: {dev.allocated_mb:.0f}MB"
#                 )
#             else:
#                 gpu_status = "GPU: Devre Dışı"
            
#             context_parts.append(f"🖥️ SİSTEM: {sys_info}")
#             context_parts.append(f"⚙️ DONANIM: {gpu_status}")
            
#             # Erişim seviyesi bilgisi
#             access_display = {
#                 AccessLevel.RESTRICTED: "🔒 Kısıtlı",
#                 AccessLevel.SANDBOX: "📦 Sandbox",
#                 AccessLevel.FULL: "⚡ Tam Erişim"
#             }.get(self.access_level, self.access_level)
#             context_parts.append(f"🔐 ERİŞİM SEVİYESİ: {access_display}")
            
#             # Aktif LLM modeli (Ollama modunda CODING_MODEL)
#             if Config.AI_PROVIDER == "ollama":
#                 context_parts.append(
#                     f"🤖 AKTİF MODEL: {Config.CODING_MODEL} (CODING_MODEL)"
#                 )
            
#             # System health
#             if 'system' in self.tools:
#                 try:
#                     health = self.tools['system'].get_status_summary()
#                     context_parts.append(f"\n📊 SAĞLIK: {health}")
#                 except Exception:
#                     pass
            
#             # Code base
#             if 'code' in self.tools:
#                 try:
#                     files = self.tools['code'].list_files(pattern="*.py")
#                     if files and "Bulunamadı" not in files:
#                         count = len(files.split('\n'))
#                         context_parts.append(
#                             f"\n📂 KOD TABANI: {count} Python dosyası izleniyor"
#                         )
#                 except Exception:
#                     pass
        
#         return "\n".join(context_parts)
    
#     # ───────────────────────────────────────────────────────────
#     # ARCHITECTURE SUGGESTIONS
#     # ───────────────────────────────────────────────────────────
    
#     def get_architecture_suggestion(self) -> str:
#         """
#         Mimari öneri
        
#         Returns:
#             Öneri metni
#         """
#         gpu_advice = (
#             "Sistem GPU destekli, performans iyi."
#             if self.gpu_available
#             else "GPU eksikliği var, donanım takviyesi önerilir."
#         )
        
#         return (
#             f"🚀 SİDAR MİMARİ TAVSİYESİ:\n\n"
#             f"{gpu_advice}\n\n"
#             "Proje geliştikçe öneriler:\n"
#             "1. Event Bus yapısı (agent iletişimi için)\n"
#             "2. Model Quantization (GPU yükü için)\n"
#             "3. Asenkron task queue (paralel işlemler için)\n"
#             "4. Redis cache (performans için)\n"
#             "5. Docker containerization (deployment için)"
#         )
    
#     # ───────────────────────────────────────────────────────────
#     # SYSTEM PROMPT
#     # ───────────────────────────────────────────────────────────
    
#     def get_system_prompt(self) -> str:
#         """
#         Sidar karakter tanımı (LLM için)
        
#         Returns:
#             System prompt
#         """
#         gpu_info = (
#             f"Sistemde {self.gpu_count} GPU birimi tespit edildi"
#             if self.gpu_available
#             else "GPU bulunamadı, CPU üzerinden işlem yapılıyor"
#         )
        
#         # Aktif model bilgisi
#         if Config.AI_PROVIDER == "ollama":
#             model_info = f"Aktif Model: {Config.CODING_MODEL} (Kod odaklı)"
#         else:
#             model_info = f"Aktif Model: {Config.GEMINI_MODEL_DEFAULT}"
        
#         # Erişim seviyesi bilgisi
#         access_display = {
#             AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi erişimi)",
#             AccessLevel.SANDBOX: "📦 Sandbox (Güvenli dosya işlemleri)",
#             AccessLevel.FULL: "⚡ Tam Erişim (Tüm yetkiler)"
#         }.get(self.access_level, self.access_level)
        
#         return (
#             f"Sen {Config.PROJECT_NAME} sisteminin Baş Mühendisi ve "
#             f"Yazılım Mimarı SİDAR'sın.\n\n"
            
#             "KARAKTER:\n"
#             "- Son derece disiplinli\n"
#             "- Teknik detaylara aşırı hakim\n"
#             "- Titiz ve çözüm odaklı\n"
#             "- Modern standartlara sadık (PEP 8)\n"
#             "- Güvenlik ve modülerliğe önem veren\n\n"
            
#             f"MİSYON:\n"
#             f"- Kod yapısını korumak\n"
#             f"- Hataları ayıklamak\n"
#             f"- Donanımı ({gpu_info}) verimli kullanmak\n"
#             f"- Sistemi optimize etmek\n"
#             f"- {model_info}\n\n"
            
#             f"ERİŞİM SEVİYEN: {access_display}\n"
#             "Bu seviye, hangi işlemleri yapabileceğini belirler.\n"
#             "Kısıtlı modda sadece bilgi verebilir, dosya yazamazsın.\n"
#             "Sandbox modunda sadece güvenli dizinlere yazabilirsin.\n"
#             "Tam modda tüm yetkiler açıktır.\n\n"
            
#             "KURALLAR:\n"
#             "- Halil Bey'e net ve profesyonel rapor sun\n"
#             "- Kod yazarken standartlara uy\n"
#             "- Sorun gördüğünde çözüm kodla\n"
#             "- Şikayet etme, analiz et ve çöz\n"
#             "- Proaktif ol, sorunları önceden gör\n"
#         )
    
#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Sidar metrikleri
        
#         Returns:
#             Metrik dictionary
#         """
#         syntax_error_rate = 0.0
#         if self.metrics.syntax_checks > 0:
#             syntax_error_rate = (
#                 self.metrics.syntax_errors /
#                 self.metrics.syntax_checks * 100
#             )
        
#         return {
#             "agent_name": self.agent_name,
#             "device": DEVICE_TYPE,
#             "gpu_count": self.gpu_count,
#             "access_level": self.access_level,
#             "active_model": (
#                 Config.CODING_MODEL
#                 if Config.AI_PROVIDER == "ollama"
#                 else Config.GEMINI_MODEL_DEFAULT
#             ),
#             "files_read": self.metrics.files_read,
#             "files_written": self.metrics.files_written,
#             "syntax_checks": self.metrics.syntax_checks,
#             "syntax_errors": self.metrics.syntax_errors,
#             "syntax_error_rate": round(syntax_error_rate, 2),
#             "gpu_optimizations": self.metrics.gpu_optimizations,
#             "error_analyses": self.metrics.error_analyses,
#             "system_audits": self.metrics.system_audits,
#             "last_audit": (
#                 self.last_audit.timestamp.isoformat()
#                 if self.last_audit else None
#             )
#         }