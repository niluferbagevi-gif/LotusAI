"""
LotusAI Sidar Agent
Sürüm: 2.5.4
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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config

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
    
    def __init__(self, tools_dict: Dict[str, Any]):
        """
        Sidar başlatıcı
        
        Args:
            tools_dict: Engine'den gelen tool'lar
        """
        self.tools = tools_dict
        self.agent_name = "SİDAR"
        
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
        
        # Audit history
        self.last_audit: Optional[SystemAudit] = None
        
        gpu_status = "AKTİF" if self.gpu_available else "DEVRE DIŞI"
        logger.info(
            f"👨‍💻 {self.agent_name} Teknik Liderlik modülü başlatıldı "
            f"(GPU: {gpu_status})"
        )
    
    # ───────────────────────────────────────────────────────────
    # GPU MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def get_gpu_details(self) -> Dict[str, Any]:
        """
        GPU detaylarını getir
        
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
        
        Returns:
            Sonuç mesajı
        """
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
        
        Args:
            filepath: Dosya yolu
            content: İçerik
        
        Returns:
            Sonuç mesajı
        """
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
        
        # GPU OOM için otomatik optimizasyon
        if error_type == ErrorType.GPU_OOM:
            self.optimize_gpu_memory()
        
        return diagnosis, solution
    
    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def get_context_data(self) -> str:
        """
        Sidar için teknik bağlam
        
        Returns:
            Context string
        """
        context_parts = ["\n[👨‍💻 SİDAR TEKNİK ALTYAPI RAPORU]"]
        
        with self.lock:
            # System info
            sys_info = (
                f"OS: {platform.system()} {platform.release()} | "
                f"Python: {platform.python_version()}"
            )
            
            # GPU status
            gpu_data = self.get_gpu_details()
            if gpu_data["available"] and gpu_data["devices"]:
                dev = gpu_data['devices'][0]
                gpu_status = (
                    f"GPU: AKTİF | {dev.name} | "
                    f"Kullanım: {dev.allocated_mb:.0f}MB"
                )
            else:
                gpu_status = "GPU: Devre Dışı"
            
            context_parts.append(f"🖥️ SİSTEM: {sys_info}")
            context_parts.append(f"⚙️ DONANIM: {gpu_status}")
            
            # Aktif LLM modeli (Ollama modunda CODING_MODEL)
            if Config.AI_PROVIDER == "ollama":
                context_parts.append(
                    f"🤖 AKTİF MODEL: {Config.CODING_MODEL} (CODING_MODEL)"
                )
            
            # System health
            if 'system' in self.tools:
                try:
                    health = self.tools['system'].get_status_summary()
                    context_parts.append(f"\n📊 SAĞLIK: {health}")
                except Exception:
                    pass
            
            # Code base
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
        
        return "\n".join(context_parts)
    
    # ───────────────────────────────────────────────────────────
    # ARCHITECTURE SUGGESTIONS
    # ───────────────────────────────────────────────────────────
    
    def get_architecture_suggestion(self) -> str:
        """
        Mimari öneri
        
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