"""
LotusAI config.py - Merkezi Yapılandırma Modülü
Sürüm: 2.6.0 (OpenClaw & Hibrit Mod Tam Destek)
Açıklama: Sistem ayarları, API anahtarları, donanım tespiti, dizin yönetimi ve erişim seviyesi kontrolü.
"""

import os
import sys
import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════
# UYARI FİLTRELERİ
# ═══════════════════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ═══════════════════════════════════════════════════════════════
# TEMEL DİZİN VE .ENV YÜKLEMESİ (SIRA DÜZELTİLDİ)
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Ortam değişkenleri diğer her şeyden ÖNCE yüklenmeli
if not ENV_PATH.exists():
    print("⚠️ '.env' dosyası bulunamadı! Varsayılan ayarlar kullanılacak.")
else:
    load_dotenv(dotenv_path=ENV_PATH)

# ═══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════
def get_bool_env(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ["true", "1", "yes", "on"]

def get_int_env(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_list_env(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    if default is None:
        default = []
    value = os.getenv(key, "")
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]

# ═══════════════════════════════════════════════════════════════
# LOGLAMA SİSTEMİ (DİNAMİK)
# ═══════════════════════════════════════════════════════════════
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = BASE_DIR / os.getenv("LOG_FILE", "logs/lotus_system.log")
LOG_MAX_BYTES = get_int_env("LOG_MAX_BYTES", 10485760)  # Varsayılan 10MB
LOG_BACKUP_COUNT = get_int_env("LOG_BACKUP_COUNT", 5)

# Eğer logs/ klasörü belirtilmişse ancak yoksa ebeveyn klasörü de yarat
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("LotusAI.Config")

if ENV_PATH.exists():
    logger.info(f"✅ Ortam değişkenleri yüklendi: {ENV_PATH}")

# ═══════════════════════════════════════════════════════════════
# ERİŞİM SEVİYESİ TANIMLARI (OpenClaw stili)
# ═══════════════════════════════════════════════════════════════
class AccessLevel:
    RESTRICTED = "restricted"   # 0: Sadece bilgi alma (okuma)
    SANDBOX = "sandbox"         # 1: Güvenli dosya yazma (sınırlı yazma)
    FULL = "full"               # 2: Tam erişim (terminal komutları dahil)

# ═══════════════════════════════════════════════════════════════
# DONANIM TESPİTİ
# ═══════════════════════════════════════════════════════════════
@dataclass
class HardwareInfo:
    """Donanım bilgilerini tutar"""
    has_cuda: bool
    gpu_name: str
    gpu_count: int = 0
    cpu_count: int = 0

def check_hardware() -> HardwareInfo:
    info = HardwareInfo(has_cuda=False, gpu_name="N/A")

    if not get_bool_env("USE_GPU", True):
        logger.info("ℹ️ GPU kullanımı .env ayarları ile devre dışı bırakıldı.")
        info.gpu_name = "Devre Dışı (Kullanıcı)"
        return info

    try:
        import torch
        if torch.cuda.is_available():
            info.has_cuda = True
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_count = torch.cuda.device_count()
            logger.info(
                f"🚀 Donanım Hızlandırma Aktif: {info.gpu_name} "
                f"({info.gpu_count} GPU tespit edildi)"
            )
        else:
            logger.info("ℹ️ CUDA bulunamadı, sistem CPU modunda çalışacak.")
            info.gpu_name = "CUDA Bulunamadı"
    except ImportError:
        logger.warning("⚠️ PyTorch yüklü değil, GPU kontrolü atlanıyor.")
        info.gpu_name = "PyTorch Yok"
    except Exception as e:
        logger.warning(f"⚠️ Donanım kontrolü hatası: {e}")
        info.gpu_name = "Tespit Edilemedi"

    try:
        import multiprocessing
        info.cpu_count = multiprocessing.cpu_count()
    except:
        info.cpu_count = 1

    return info

# Global donanım bilgisi
HARDWARE = check_hardware()

# ═══════════════════════════════════════════════════════════════
# ANA YAPILANDIRMA SINIFI
# ═══════════════════════════════════════════════════════════════
class Config:
    """
    LotusAI Merkezi Yapılandırma Sınıfı
    Sürüm: 2.6.0
    """

    # ───────────────────────────────────────────────────────────
    # GENEL SİSTEM BİLGİLERİ
    # ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "LotusAI")
    VERSION: str = "2.6.0"
    DEBUG_MODE: bool = get_bool_env("DEBUG_MODE", False)
    WORK_DIR: Path = Path(os.getenv("WORK_DIR", ".")).resolve()
    LANGUAGE: str = os.getenv("LANGUAGE", "tr").lower()
    VOICE_ENABLED: bool = get_bool_env("VOICE_ENABLED", True)
    TARGET_SCREEN: int = get_int_env("TARGET_SCREEN", 0)

    # ───────────────────────────────────────────────────────────
    # DİZİN YAPILANDIRMASI
    # ───────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = WORK_DIR / "uploads"
    TEMPLATE_DIR: Path = WORK_DIR / "templates"
    STATIC_DIR: Path = WORK_DIR / "static"
    LOG_DIR: Path = WORK_DIR / "logs"
    VOICES_DIR: Path = WORK_DIR / "voices"
    FACES_DIR: Path = WORK_DIR / "faces"
    MODELS_DIR: Path = WORK_DIR / "models"
    DATA_DIR: Path = WORK_DIR / "core" / "data"
    SANDBOX_DIR: Path = WORK_DIR / "temp" 

    REQUIRED_DIRS: List[Path] = [
        UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR,
        FACES_DIR, MODELS_DIR, DATA_DIR, SANDBOX_DIR
    ]

    # ───────────────────────────────────────────────────────────
    # SİSTEM ZAMANLAMALARI & GELİŞMİŞ AYARLAR
    # ───────────────────────────────────────────────────────────
    CONVERSATION_TIMEOUT: int = get_int_env("CONVERSATION_TIMEOUT", 30)
    SYSTEM_CHECK_INTERVAL: int = get_int_env("SYSTEM_CHECK_INTERVAL", 300)
    MAX_CONCURRENT_AGENTS: int = get_int_env("MAX_CONCURRENT_AGENTS", 6)
    MEMORY_RETENTION_DAYS: int = get_int_env("MEMORY_RETENTION_DAYS", 30)
    AUTO_CLEANUP: bool = get_bool_env("AUTO_CLEANUP", True)
    BACKUP_INTERVAL: int = get_int_env("BACKUP_INTERVAL", 24)
    EXPERIMENTAL_MODE: bool = get_bool_env("EXPERIMENTAL_MODE", False)
    MULTI_GPU: bool = get_bool_env("MULTI_GPU", False)
    DISTRIBUTED_MODE: bool = get_bool_env("DISTRIBUTED_MODE", False)

    # ───────────────────────────────────────────────────────────
    # AI SAĞLAYICI AYARLARI
    # ───────────────────────────────────────────────────────────
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "ollama").lower()
    USE_GPU: bool = HARDWARE.has_cuda
    GPU_INFO: str = HARDWARE.gpu_name
    CPU_COUNT: int = HARDWARE.cpu_count

    # ───────────────────────────────────────────────────────────
    # ERİŞİM SEVİYESİ (OpenClaw stili)
    # ───────────────────────────────────────────────────────────
    ACCESS_LEVEL: str = os.getenv("ACCESS_LEVEL", AccessLevel.SANDBOX).lower()
    if ACCESS_LEVEL not in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
        ACCESS_LEVEL = AccessLevel.SANDBOX
        logger.warning(f"Geçersiz ACCESS_LEVEL, varsayılan {ACCESS_LEVEL} kullanılıyor.")

    # ───────────────────────────────────────────────────────────
    # GEMINI (GOOGLE) MODEL AYARLARI
    # ───────────────────────────────────────────────────────────
    GEMINI_MODEL_DEFAULT: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-flash-lite")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_MAX_TOKENS: int = get_int_env("GEMINI_MAX_TOKENS", 8192)

    # ───────────────────────────────────────────────────────────
    # API ANAHTAR YÖNETİMİ (AKILLI FALLBACK)
    # ───────────────────────────────────────────────────────────
    _MAIN_KEY: Optional[str] = None
    _USING_FALLBACK_KEY: bool = False

    _KEY_PRIORITY = [
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_ATLAS",
        "GEMINI_API_KEY_SIDAR",
        "GEMINI_API_KEY_KURT",
        "GEMINI_API_KEY_KERBEROS",
        "GEMINI_API_KEY_POYRAZ",
        "GEMINI_API_KEY_GAYA"
    ]

    for key_name in _KEY_PRIORITY:
        _MAIN_KEY = os.getenv(key_name)
        if _MAIN_KEY:
            if key_name != "GEMINI_API_KEY":
                _USING_FALLBACK_KEY = True
                if AI_PROVIDER == "gemini":
                    logger.info(f"ℹ️ Ana API anahtarı bulunamadı, {key_name} kullanılacak.")
            break

    # HuggingFace Token
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    # ───────────────────────────────────────────────────────────
    # OLLAMA (YEREL AI) MODEL AYARLARI
    # ───────────────────────────────────────────────────────────
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemma2:9b")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llama3.2-vision")
    CODING_MODEL: str = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
    LOCAL_VEK: str = os.getenv("LOCAL_VEK", "nomic-embed-text")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    OLLAMA_TIMEOUT: int = get_int_env("OLLAMA_TIMEOUT", 30)

    # ───────────────────────────────────────────────────────────
    # AJAN YAPILANDIRMASI
    # ───────────────────────────────────────────────────────────
    AGENT_CONFIGS: Dict[str, Dict[str, str]] = {
        "ATLAS": {
            "key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_PRO,
            "role": "Koordinatör"
        },
        "SIDAR": {
            "key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_DEFAULT,
            "ollama_model": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
            "role": "Yönetim"
        },
        "KURT": {
            "key": os.getenv("GEMINI_API_KEY_KURT", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_DEFAULT,
            "role": "Güvenlik"
        },
        "POYRAZ": {
            "key": os.getenv("GEMINI_API_KEY_POYRAZ", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_DEFAULT,
            "role": "Analiz"
        },
        "KERBEROS": {
            "key": os.getenv("GEMINI_API_KEY_KERBEROS", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_PRO,
            "role": "Güvenlik+"
        },
        "GAYA": {
            "key": os.getenv("GEMINI_API_KEY_GAYA", _MAIN_KEY or ""),
            "model": GEMINI_MODEL_DEFAULT,
            "role": "Asistan"
        }
    }

    # ───────────────────────────────────────────────────────────
    # GÜVENLİK & KAMERA (MANAGER) AYARLARI
    # ───────────────────────────────────────────────────────────
    FACE_REC_MODEL: str = os.getenv("FACE_REC_MODEL", "cnn" if USE_GPU else "hog")
    FACE_TOLERANCE: float = float(os.getenv("FACE_TOLERANCE", "0.45"))
    LIVE_VISUAL_CHECK: bool = get_bool_env("LIVE_VISUAL_CHECK", True)
    PATRON_IMAGE_PATH: Path = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")
    CAMERA_INDEX: int = get_int_env("CAMERA_INDEX", 2)

    # ───────────────────────────────────────────────────────────
    # SES (TTS/STT) AYARLARI
    # ───────────────────────────────────────────────────────────
    USE_XTTS: bool = get_bool_env("USE_XTTS", True)
    TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "tr")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "tr-TR")
    XTTS_MODEL: str = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3") # Yedek Motor
    VOICE_SPEED: int = get_int_env("VOICE_SPEED", 150)

    # ───────────────────────────────────────────────────────────
    # FİNANS MODÜLÜ & BINANCE
    # ───────────────────────────────────────────────────────────
    FINANCE_MODE: bool = get_bool_env("FINANCE_MODE", True)
    DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "TRY")
    SUPPORTED_CURRENCIES: List[str] = get_list_env(
        "SUPPORTED_CURRENCIES", ["TRY", "USD", "EUR", "GBP"]
    )
    BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

    # ───────────────────────────────────────────────────────────
    # SOSYAL MEDYA & META API ENTEGRASYONU
    # ───────────────────────────────────────────────────────────
    META_APP_ID: Optional[str] = os.getenv("META_APP_ID")
    META_ACCESS_TOKEN: Optional[str] = os.getenv("META_ACCESS_TOKEN")
    META_APP_SECRET: Optional[str] = os.getenv("META_APP_SECRET")
    META_VERIFY_TOKEN: str = os.getenv("META_VERIFY_TOKEN", "lotus_verify_token")
    
    WHATSAPP_PHONE_ID: Optional[str] = os.getenv("WHATSAPP_PHONE_ID")
    INSTAGRAM_ACCOUNT_ID: Optional[str] = os.getenv("INSTAGRAM_ACCOUNT_ID")
    FACEBOOK_PAGE_ID: Optional[str] = os.getenv("FACEBOOK_PAGE_ID")
    
    INSTAGRAM_LOGIN_USER: str = os.getenv("INSTAGRAM_LOGIN_USER", "lotusbagevi")
    INSTAGRAM_PASSWORD: Optional[str] = os.getenv("INSTAGRAM_PASSWORD")
    INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "lotusbagevi") # Hedef
    FACEBOOK_PAGE_NAME: str = os.getenv("FACEBOOK_PAGE_NAME", "niluferbagevi")
    COMPETITORS: List[str] = get_list_env("COMPETITORS", [])

    # ───────────────────────────────────────────────────────────
    # TESLİMAT PLATFORMLARI ENTEGRASYONU
    # ───────────────────────────────────────────────────────────
    YEMEKSEPETI_URL: Optional[str] = os.getenv("YEMEKSEPETI_URL")
    YEMEKSEPETI_USER: Optional[str] = os.getenv("YEMEKSEPETI_USER")
    YEMEKSEPETI_PASS: Optional[str] = os.getenv("YEMEKSEPETI_PASS")
    
    GETIR_URL: Optional[str] = os.getenv("GETIR_URL")
    GETIR_USER: Optional[str] = os.getenv("GETIR_USER")
    GETIR_PASS: Optional[str] = os.getenv("GETIR_PASS")
    
    TRENDYOL_URL: Optional[str] = os.getenv("TRENDYOL_URL")
    TRENDYOL_USER: Optional[str] = os.getenv("TRENDYOL_USER")
    TRENDYOL_PASS: Optional[str] = os.getenv("TRENDYOL_PASS")

    # ───────────────────────────────────────────────────────────
    # VERİTABANI & DEPOLAMA (CHROMADB)
    # ───────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "lotus_memory")

    # ───────────────────────────────────────────────────────────
    # WEB SUNUCUSU (FLASK)
    # ───────────────────────────────────────────────────────────
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = get_int_env("FLASK_PORT", 5000)
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "secret_key_change_me")

    # ───────────────────────────────────────────────────────────
    # GITHUB ENTEGRASYONU
    # ───────────────────────────────────────────────────────────
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "niluferbagevi-gif/LotusAI")
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN", None)

    # ───────────────────────────────────────────────────────────
    # GÜVENLİK (SİSTEM)
    # ───────────────────────────────────────────────────────────
    API_AUTH_ENABLED: bool = get_bool_env("API_AUTH_ENABLED", True)
    MAX_LOGIN_ATTEMPTS: int = get_int_env("MAX_LOGIN_ATTEMPTS", 3)
    SESSION_TIMEOUT: int = get_int_env("SESSION_TIMEOUT", 3600)

    # ───────────────────────────────────────────────────────────
    # METOTLAR
    # ───────────────────────────────────────────────────────────

    @classmethod
    def initialize_directories(cls) -> bool:
        success = True
        for folder in cls.REQUIRED_DIRS:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                logger.debug(f"✅ Dizin hazır: {folder.name}")
            except Exception as e:
                logger.error(f"❌ Dizin oluşturulamadı ({folder.name}): {e}")
                success = False
        return success

    @classmethod
    def get_agent_settings(cls, agent_name: str) -> Dict[str, str]:
        name_upper = agent_name.upper()

        if name_upper in cls.AGENT_CONFIGS:
            config = cls.AGENT_CONFIGS[name_upper].copy()

            if not config.get("key") and cls._MAIN_KEY:
                config["key"] = cls._MAIN_KEY
                logger.debug(f"Ajan {agent_name} için ana anahtar kullanılıyor")

            if cls.AI_PROVIDER == "ollama":
                if name_upper == "SIDAR":
                    config["active_model"] = config.get("ollama_model", cls.CODING_MODEL)
                else:
                    config["active_model"] = cls.TEXT_MODEL
            else:
                config["active_model"] = config.get("model", cls.GEMINI_MODEL_DEFAULT)

            return config

        logger.warning(f"⚠️ Bilinmeyen ajan: {agent_name}, varsayılan ayarlar kullanılıyor")
        return {
            "key": cls._MAIN_KEY or "",
            "model": cls.GEMINI_MODEL_DEFAULT,
            "active_model": cls.TEXT_MODEL if cls.AI_PROVIDER == "ollama" else cls.GEMINI_MODEL_DEFAULT,
            "role": "Bilinmiyor"
        }

    @classmethod
    def get_ollama_model_for(cls, agent_name: str) -> str:
        name_upper = agent_name.upper()
        if name_upper == "SIDAR":
            return cls.CODING_MODEL
        return cls.TEXT_MODEL

    @classmethod
    def set_provider_mode(cls, mode: str) -> None:
        mode_map = {
            "online": "gemini",
            "gemini": "gemini",
            "local": "ollama",
            "ollama": "ollama"
        }
        m_lower = mode.lower()
        if m_lower in mode_map:
            cls.AI_PROVIDER = mode_map[m_lower]
            logger.info(f"✅ AI Sağlayıcı modu güncellendi: {cls.AI_PROVIDER.upper()}")
        else:
            logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")
            logger.info(f"   Geçerli modlar: {', '.join(mode_map.keys())}")

    @classmethod
    def set_access_level(cls, level: str) -> None:
        level_lower = level.lower()
        if level_lower in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
            cls.ACCESS_LEVEL = level_lower
            logger.info(f"✅ Erişim seviyesi güncellendi: {cls.ACCESS_LEVEL.upper()}")
        else:
            logger.error(f"❌ Geçersiz erişim seviyesi: {level}, varsayılan sandbox kullanılacak")
            cls.ACCESS_LEVEL = AccessLevel.SANDBOX

    @classmethod
    def validate_critical_settings(cls) -> bool:
        is_valid = True

        if not cls.initialize_directories():
            logger.warning("⚠️ Bazı dizinler oluşturulamadı")
            is_valid = False

        if cls.AI_PROVIDER == "gemini":
            dummy_key = "BURAYA_GEMINI_API_KEY_YAZIN"
            if not cls._MAIN_KEY or cls._MAIN_KEY == dummy_key:
                logger.error(
                    "❌ KRİTİK HATA: Gemini modu seçili ama geçerli bir API anahtarı yok!\n"
                    "   Lütfen .env dosyasını kontrol edin."
                )
                is_valid = False
            else:
                if len(cls._MAIN_KEY) < 30:
                    logger.warning("⚠️ API anahtarı çok kısa görünüyor, geçersiz olabilir")

        if cls.LIVE_VISUAL_CHECK:
            if not cls.PATRON_IMAGE_PATH.exists():
                logger.warning(
                    f"⚠️ Patron resmi bulunamadı: {cls.PATRON_IMAGE_PATH}\n"
                    "   Yüz tanıma devre dışı bırakılabilir."
                )

        if cls.AI_PROVIDER == "ollama":
            try:
                import requests
                base_url = cls.OLLAMA_URL.rstrip('/')
                tags_url = f"{base_url}/tags" if base_url.endswith('api') else f"{base_url}/api/tags"
                
                response = requests.get(tags_url, timeout=2)
                if response.status_code != 200:
                    logger.warning(f"⚠️ Ollama servisi ({tags_url}) yanıt vermiyor. Durum: {response.status_code}")
                else:
                    logger.info(f"✅ Ollama bağlantısı başarılı")
            except Exception:
                logger.warning(
                    f"⚠️ Ollama servisine ulaşılamadı ({cls.OLLAMA_URL})\n"
                    "   Terminal'de 'ollama serve' komutunu çalıştırdığınızdan emin olun."
                )

        return is_valid

    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        return {
            "project": cls.PROJECT_NAME,
            "version": cls.VERSION,
            "provider": cls.AI_PROVIDER,
            "access_level": cls.ACCESS_LEVEL,
            "gpu_enabled": cls.USE_GPU,
            "gpu_info": cls.GPU_INFO,
            "cpu_count": cls.CPU_COUNT,
            "debug_mode": cls.DEBUG_MODE,
            "github_integration": bool(cls.GITHUB_TOKEN),
            "agents": list(cls.AGENT_CONFIGS.keys()),
            "ollama_models": {
                "text": cls.TEXT_MODEL,
                "vision": cls.VISION_MODEL,
                "coding": cls.CODING_MODEL,
                "embed": cls.LOCAL_VEK
            }
        }

    @classmethod
    def print_config_summary(cls) -> None:
        print("\n" + "═" * 60)
        print(f"  {cls.PROJECT_NAME} v{cls.VERSION} - Yapılandırma Özeti")
        print("═" * 60)
        print(f"  AI Sağlayıcı    : {cls.AI_PROVIDER.upper()}")
        print(f"  GPU Desteği     : {'✓ ' + cls.GPU_INFO if cls.USE_GPU else '✗ CPU Modu'}")
        print(f"  Erişim Seviyesi : {cls.ACCESS_LEVEL.upper()}")
        print(f"  CPU Çekirdek    : {cls.CPU_COUNT}")
        print(f"  Aktif Ajanlar   : {len(cls.AGENT_CONFIGS)}")
        print(f"  Debug Modu      : {'Açık' if cls.DEBUG_MODE else 'Kapalı'}")
        print(f"  GitHub Repo     : {cls.GITHUB_REPO}")
        if cls.AI_PROVIDER == "ollama":
            print("  ── Ollama Modelleri ──────────────────────────────")
            print(f"  TEXT Model      : {cls.TEXT_MODEL}")
            print(f"  VISION Model    : {cls.VISION_MODEL}")
            print(f"  CODING Model    : {cls.CODING_MODEL}  ← Sidar")
            print(f"  EMBED Model     : {cls.LOCAL_VEK}")
        print("═" * 60 + "\n")


logger.info(f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması yüklendi")

if Config.DEBUG_MODE:
    Config.print_config_summary()


# """
# LotusAI config.py - Merkezi Yapılandırma Modülü
# Sürüm: 2.6.0 (OpenClaw & Hibrit Mod Tam Destek)
# Açıklama: Sistem ayarları, API anahtarları, donanım tespiti, dizin yönetimi ve erişim seviyesi kontrolü.
# """

# import os
# import sys
# import logging
# import warnings
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# from dotenv import load_dotenv
# from typing import Dict, Any, Optional, List
# from dataclasses import dataclass

# # ═══════════════════════════════════════════════════════════════
# # UYARI FİLTRELERİ
# # ═══════════════════════════════════════════════════════════════
# warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
# warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# # ═══════════════════════════════════════════════════════════════
# # TEMEL DİZİN YAPILANDIRMASI
# # ═══════════════════════════════════════════════════════════════
# BASE_DIR = Path(__file__).resolve().parent
# LOG_DIR = BASE_DIR / "logs"
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# # ═══════════════════════════════════════════════════════════════
# # LOGLAMA SİSTEMİ
# # ═══════════════════════════════════════════════════════════════
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL, logging.INFO),
#     format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         RotatingFileHandler(
#             LOG_DIR / "lotus_system.log",
#             maxBytes=15 * 1024 * 1024,  # 15MB
#             backupCount=10,
#             encoding="utf-8"
#         )
#     ]
# )
# logger = logging.getLogger("LotusAI.Config")

# # ═══════════════════════════════════════════════════════════════
# # ORTAM DEĞİŞKENLERİ
# # ═══════════════════════════════════════════════════════════════
# ENV_PATH = BASE_DIR / ".env"
# if not ENV_PATH.exists():
#     logger.warning("⚠️ '.env' dosyası bulunamadı! Varsayılan ayarlar kullanılacak.")
# else:
#     load_dotenv(dotenv_path=ENV_PATH)
#     logger.info(f"✅ Ortam değişkenleri yüklendi: {ENV_PATH}")


# # ═══════════════════════════════════════════════════════════════
# # YARDIMCI FONKSİYONLAR
# # ═══════════════════════════════════════════════════════════════
# def get_bool_env(key: str, default: bool = False) -> bool:
#     val = os.getenv(key, str(default)).lower()
#     return val in ["true", "1", "yes", "on"]


# def get_int_env(key: str, default: int = 0) -> int:
#     try:
#         return int(os.getenv(key, default))
#     except (ValueError, TypeError):
#         logger.warning(f"⚠️ '{key}' geçersiz değer, varsayılan kullanılıyor: {default}")
#         return default


# def get_list_env(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
#     if default is None:
#         default = []
#     value = os.getenv(key, "")
#     if not value:
#         return default
#     return [item.strip() for item in value.split(separator) if item.strip()]


# # ═══════════════════════════════════════════════════════════════
# # ERİŞİM SEVİYESİ TANIMLARI (OpenClaw stili)
# # ═══════════════════════════════════════════════════════════════
# class AccessLevel:
#     RESTRICTED = "restricted"   # 0: Sadece bilgi alma (okuma)
#     SANDBOX = "sandbox"         # 1: Güvenli dosya yazma (sınırlı yazma)
#     FULL = "full"               # 2: Tam erişim (terminal komutları dahil)


# # ═══════════════════════════════════════════════════════════════
# # DONANIM TESPİTİ
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class HardwareInfo:
#     """Donanım bilgilerini tutar"""
#     has_cuda: bool
#     gpu_name: str
#     gpu_count: int = 0
#     cpu_count: int = 0


# def check_hardware() -> HardwareInfo:
#     info = HardwareInfo(has_cuda=False, gpu_name="N/A")

#     if not get_bool_env("USE_GPU", True):
#         logger.info("ℹ️ GPU kullanımı .env ayarları ile devre dışı bırakıldı.")
#         info.gpu_name = "Devre Dışı (Kullanıcı)"
#         return info

#     try:
#         import torch
#         if torch.cuda.is_available():
#             info.has_cuda = True
#             info.gpu_name = torch.cuda.get_device_name(0)
#             info.gpu_count = torch.cuda.device_count()
#             logger.info(
#                 f"🚀 Donanım Hızlandırma Aktif: {info.gpu_name} "
#                 f"({info.gpu_count} GPU tespit edildi)"
#             )
#         else:
#             logger.info("ℹ️ CUDA bulunamadı, sistem CPU modunda çalışacak.")
#             info.gpu_name = "CUDA Bulunamadı"
#     except ImportError:
#         logger.warning("⚠️ PyTorch yüklü değil, GPU kontrolü atlanıyor.")
#         info.gpu_name = "PyTorch Yok"
#     except Exception as e:
#         logger.warning(f"⚠️ Donanım kontrolü hatası: {e}")
#         info.gpu_name = "Tespit Edilemedi"

#     try:
#         import multiprocessing
#         info.cpu_count = multiprocessing.cpu_count()
#     except:
#         info.cpu_count = 1

#     return info


# # Global donanım bilgisi
# HARDWARE = check_hardware()


# # ═══════════════════════════════════════════════════════════════
# # ANA YAPILANDIRMA SINIFI
# # ═══════════════════════════════════════════════════════════════
# class Config:
#     """
#     LotusAI Merkezi Yapılandırma Sınıfı

#     Sürüm: 2.6.0
#     Özellikler:
#     - Çoklu API anahtarı yönetimi
#     - Ajan bazlı konfigürasyon
#     - Otomatik donanım tespiti
#     - Dizin yönetimi
#     - CODING_MODEL desteği (Sidar özel)
#     - Binance, Instagram ve GitHub API entegrasyonu
#     - Erişim seviyesi ve Hibrit Mod Yönetimi
#     """

#     # ───────────────────────────────────────────────────────────
#     # GENEL SİSTEM BİLGİLERİ
#     # ───────────────────────────────────────────────────────────
#     PROJECT_NAME: str = "LotusAI"
#     VERSION: str = "2.6.0"
#     DEBUG_MODE: bool = get_bool_env("DEBUG_MODE", False)
#     WORK_DIR: Path = Path(os.getenv("WORK_DIR", BASE_DIR))
#     LANGUAGE: str = os.getenv("LANGUAGE", "tr").lower()
#     VOICE_ENABLED: bool = get_bool_env("VOICE_ENABLED", True)

#     # ───────────────────────────────────────────────────────────
#     # DİZİN YAPILANDIRMASI
#     # ───────────────────────────────────────────────────────────
#     UPLOAD_DIR: Path = WORK_DIR / "uploads"
#     TEMPLATE_DIR: Path = WORK_DIR / "templates"
#     STATIC_DIR: Path = WORK_DIR / "static"
#     LOG_DIR: Path = WORK_DIR / "logs"
#     VOICES_DIR: Path = WORK_DIR / "voices"
#     FACES_DIR: Path = WORK_DIR / "faces"
#     MODELS_DIR: Path = WORK_DIR / "models"
#     DATA_DIR: Path = WORK_DIR / "core" / "data"
#     SANDBOX_DIR: Path = WORK_DIR / "temp" 

#     REQUIRED_DIRS: List[Path] = [
#         UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR,
#         FACES_DIR, MODELS_DIR, DATA_DIR
#     ]

#     # ───────────────────────────────────────────────────────────
#     # SİSTEM ZAMANLAMALARI
#     # ───────────────────────────────────────────────────────────
#     CONVERSATION_TIMEOUT: int = get_int_env("CONVERSATION_TIMEOUT", 60)
#     SYSTEM_CHECK_INTERVAL: int = get_int_env("SYSTEM_CHECK_INTERVAL", 300)

#     # ───────────────────────────────────────────────────────────
#     # AI SAĞLAYICI AYARLARI
#     # ───────────────────────────────────────────────────────────
#     # Bu değer artık set_provider_mode ile dinamik değişebilir
#     AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini").lower()

#     # Donanım bilgileri
#     USE_GPU: bool = HARDWARE.has_cuda
#     GPU_INFO: str = HARDWARE.gpu_name
#     CPU_COUNT: int = HARDWARE.cpu_count

#     # ───────────────────────────────────────────────────────────
#     # ERİŞİM SEVİYESİ (OpenClaw stili)
#     # ───────────────────────────────────────────────────────────
#     # Bu değer artık set_access_level ile dinamik değişebilir
#     ACCESS_LEVEL: str = os.getenv("ACCESS_LEVEL", AccessLevel.SANDBOX).lower()
    
#     # Başlangıç geçerlilik kontrolü
#     if ACCESS_LEVEL not in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
#         ACCESS_LEVEL = AccessLevel.SANDBOX
#         logger.warning(f"Geçersiz ACCESS_LEVEL, varsayılan {ACCESS_LEVEL} kullanılıyor.")

#     # ───────────────────────────────────────────────────────────
#     # GEMINI (GOOGLE) MODEL AYARLARI
#     # ───────────────────────────────────────────────────────────
#     GEMINI_MODEL_DEFAULT: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
#     GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
#     GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
#     GEMINI_MAX_TOKENS: int = get_int_env("GEMINI_MAX_TOKENS", 8192)

#     # ───────────────────────────────────────────────────────────
#     # API ANAHTAR YÖNETİMİ (AKILLI FALLBACK)
#     # ───────────────────────────────────────────────────────────
#     _MAIN_KEY: Optional[str] = None
#     _USING_FALLBACK_KEY: bool = False

#     _KEY_PRIORITY = [
#         "GEMINI_API_KEY",
#         "GEMINI_API_KEY_ATLAS",
#         "GEMINI_API_KEY_SIDAR",
#         "GEMINI_API_KEY_KURT",
#         "GEMINI_API_KEY_KERBEROS",
#         "GEMINI_API_KEY_POYRAZ",
#         "GEMINI_API_KEY_GAYA"
#     ]

#     for key_name in _KEY_PRIORITY:
#         _MAIN_KEY = os.getenv(key_name)
#         if _MAIN_KEY:
#             if key_name != "GEMINI_API_KEY":
#                 _USING_FALLBACK_KEY = True
#                 # Sadece Gemini modu aktifse bu uyarıyı ver, yoksa kafa karıştırma
#                 if AI_PROVIDER == "gemini":
#                     logger.info(f"ℹ️ Ana API anahtarı bulunamadı, {key_name} kullanılacak.")
#             break

#     # ───────────────────────────────────────────────────────────
#     # OLLAMA (YEREL AI) MODEL AYARLARI
#     # ───────────────────────────────────────────────────────────
#     # Genel metin modeli (gemma2:9b)
#     TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemma2:9b")
#     # Görsel analiz modeli (llama3.2-vision)
#     VISION_MODEL: str = os.getenv("VISION_MODEL", "llama3.2-vision")
#     # Sidar'a özel kodlama modeli (qwen2.5-coder:7b)
#     CODING_MODEL: str = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
#     # Vektörleme modeli
#     LOCAL_VEK: str = os.getenv("LOCAL_VEK", "nomic-embed-text")
#     # Bağlantı ayarları
#     OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
#     OLLAMA_TIMEOUT: int = get_int_env("OLLAMA_TIMEOUT", 30)

#     # ───────────────────────────────────────────────────────────
#     # AJAN YAPILANDIRMASI
#     # ───────────────────────────────────────────────────────────
#     AGENT_CONFIGS: Dict[str, Dict[str, str]] = {
#         "ATLAS": {
#             "key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_PRO,
#             "role": "Koordinatör"
#         },
#         "SIDAR": {
#             "key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_DEFAULT,
#             "ollama_model": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
#             "role": "Yönetim"
#         },
#         "KURT": {
#             "key": os.getenv("GEMINI_API_KEY_KURT", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_DEFAULT,
#             "role": "Güvenlik"
#         },
#         "POYRAZ": {
#             "key": os.getenv("GEMINI_API_KEY_POYRAZ", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_DEFAULT,
#             "role": "Analiz"
#         },
#         "KERBEROS": {
#             "key": os.getenv("GEMINI_API_KEY_KERBEROS", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_PRO,
#             "role": "Güvenlik+"
#         },
#         "GAYA": {
#             "key": os.getenv("GEMINI_API_KEY_GAYA", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_DEFAULT,
#             "role": "Asistan"
#         }
#     }

#     # ───────────────────────────────────────────────────────────
#     # MANAGER (YÖNETİCİ) ÖZEL AYARLARI
#     # ───────────────────────────────────────────────────────────
#     FACE_REC_MODEL: str = "cnn" if USE_GPU else "hog"
#     LIVE_VISUAL_CHECK: bool = get_bool_env("LIVE_VISUAL_CHECK", True)
#     PATRON_IMAGE_PATH: Path = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")

#     # Finans ayarları
#     FINANCE_MODE: bool = get_bool_env("FINANCE_MODE", True)
#     DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "TRY")
#     SUPPORTED_CURRENCIES: List[str] = get_list_env(
#         "SUPPORTED_CURRENCIES",
#         ["TRY", "USD", "EUR", "GBP"]
#     )
    
#     # Binance API Ayarları
#     BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
#     BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

#     # Instagram Hedef Hesap
#     INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "lotusbagevi")

#     # Ses ayarları
#     USE_XTTS: bool = get_bool_env("USE_XTTS", False)
#     TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
#     VOICE_SPEED: int = get_int_env("VOICE_SPEED", 150)

#     # ───────────────────────────────────────────────────────────
#     # GITHUB ENTEGRASYONU
#     # ───────────────────────────────────────────────────────────
#     GITHUB_REPO: str = os.getenv("GITHUB_REPO", "niluferbagevi-gif/LotusAI")
#     GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN", None)

#     # ───────────────────────────────────────────────────────────
#     # GÜVENLİK AYARLARI
#     # ───────────────────────────────────────────────────────────
#     API_AUTH_ENABLED: bool = get_bool_env("API_AUTH_ENABLED", True)
#     MAX_LOGIN_ATTEMPTS: int = get_int_env("MAX_LOGIN_ATTEMPTS", 3)
#     SESSION_TIMEOUT: int = get_int_env("SESSION_TIMEOUT", 3600)

#     # ───────────────────────────────────────────────────────────
#     # METOTLAR
#     # ───────────────────────────────────────────────────────────

#     @classmethod
#     def initialize_directories(cls) -> bool:
#         success = True
#         for folder in cls.REQUIRED_DIRS:
#             try:
#                 folder.mkdir(parents=True, exist_ok=True)
#                 logger.debug(f"✅ Dizin hazır: {folder.name}")
#             except Exception as e:
#                 logger.error(f"❌ Dizin oluşturulamadı ({folder.name}): {e}")
#                 success = False
#         return success

#     @classmethod
#     def get_agent_settings(cls, agent_name: str) -> Dict[str, str]:
#         """
#         Belirli bir ajan için ayarları döner.
#         Ollama modunda SIDAR için CODING_MODEL otomatik seçilir.
#         """
#         name_upper = agent_name.upper()

#         if name_upper in cls.AGENT_CONFIGS:
#             config = cls.AGENT_CONFIGS[name_upper].copy()

#             # Eğer ajan anahtarı yoksa ana anahtarı kullan
#             if not config.get("key") and cls._MAIN_KEY:
#                 config["key"] = cls._MAIN_KEY
#                 # Çok sık log basmaması için debug seviyesinde
#                 logger.debug(f"Ajan {agent_name} için ana anahtar kullanılıyor")

#             # Model Seçimi (Hibrit Yapı)
#             if cls.AI_PROVIDER == "ollama":
#                 # Sidar için özel kodlama modeli
#                 if name_upper == "SIDAR":
#                     config["active_model"] = config.get("ollama_model", cls.CODING_MODEL)
#                 else:
#                     config["active_model"] = cls.TEXT_MODEL
#             else:
#                 # Gemini Modu
#                 config["active_model"] = config.get("model", cls.GEMINI_MODEL_DEFAULT)

#             return config

#         logger.warning(f"⚠️ Bilinmeyen ajan: {agent_name}, varsayılan ayarlar kullanılıyor")
#         return {
#             "key": cls._MAIN_KEY or "",
#             "model": cls.GEMINI_MODEL_DEFAULT,
#             "active_model": cls.TEXT_MODEL if cls.AI_PROVIDER == "ollama" else cls.GEMINI_MODEL_DEFAULT,
#             "role": "Bilinmiyor"
#         }

#     @classmethod
#     def get_ollama_model_for(cls, agent_name: str) -> str:
#         """
#         Ollama modunda belirli bir ajan için kullanılacak modeli döner.
#         """
#         name_upper = agent_name.upper()
#         if name_upper == "SIDAR":
#             return cls.CODING_MODEL
#         return cls.TEXT_MODEL

#     @classmethod
#     def set_provider_mode(cls, mode: str) -> None:
#         """
#         AI sağlayıcı modunu ayarlar (Launcher'dan çağrılır).
#         Args:
#             mode: 'online', 'gemini', 'local' veya 'ollama'
#         """
#         mode_map = {
#             "online": "gemini",
#             "gemini": "gemini",
#             "local": "ollama",
#             "ollama": "ollama"
#         }
#         m_lower = mode.lower()
#         if m_lower in mode_map:
#             cls.AI_PROVIDER = mode_map[m_lower]
#             logger.info(f"✅ AI Sağlayıcı modu güncellendi: {cls.AI_PROVIDER.upper()}")
#         else:
#             logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")
#             logger.info(f"   Geçerli modlar: {', '.join(mode_map.keys())}")

#     @classmethod
#     def set_access_level(cls, level: str) -> None:
#         """
#         Erişim seviyesini ayarlar (Launcher'dan çağrılır).
#         Args:
#             level: "restricted", "sandbox" veya "full"
#         """
#         level_lower = level.lower()
#         if level_lower in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
#             cls.ACCESS_LEVEL = level_lower
#             logger.info(f"✅ Erişim seviyesi güncellendi: {cls.ACCESS_LEVEL.upper()}")
#         else:
#             logger.error(f"❌ Geçersiz erişim seviyesi: {level}, varsayılan sandbox kullanılacak")
#             cls.ACCESS_LEVEL = AccessLevel.SANDBOX

#     @classmethod
#     def validate_critical_settings(cls) -> bool:
#         """
#         Kritik sistem ayarlarını doğrular.
#         Seçilen AI Moduna göre dinamik kontrol yapar.
#         """
#         is_valid = True

#         # 1. Dizinleri oluştur
#         if not cls.initialize_directories():
#             logger.warning("⚠️ Bazı dizinler oluşturulamadı")
#             is_valid = False

#         # 2. API anahtarı kontrolü (SADECE GEMINI MODUNDAYSA)
#         if cls.AI_PROVIDER == "gemini":
#             dummy_key = "BURAYA_GEMINI_API_KEY_YAZIN"
#             if not cls._MAIN_KEY or cls._MAIN_KEY == dummy_key:
#                 logger.error(
#                     "❌ KRİTİK HATA: Gemini modu seçili ama geçerli bir API anahtarı yok!\n"
#                     "   Lütfen .env dosyasını kontrol edin."
#                 )
#                 is_valid = False
#             else:
#                 if len(cls._MAIN_KEY) < 30:
#                     logger.warning("⚠️ API anahtarı çok kısa görünüyor, geçersiz olabilir")

#         # 3. Patron resmi kontrolü
#         if cls.LIVE_VISUAL_CHECK:
#             if not cls.PATRON_IMAGE_PATH.exists():
#                 logger.warning(
#                     f"⚠️ Patron resmi bulunamadı: {cls.PATRON_IMAGE_PATH}\n"
#                     "   Yüz tanıma devre dışı bırakılabilir."
#                 )

#         # 4. Ollama kontrolü (SADECE OLLAMA MODUNDAYSA)
#         if cls.AI_PROVIDER == "ollama":
#             try:
#                 import requests
#                 # OLLAMA_URL dinamik olarak işleniyor
#                 base_url = cls.OLLAMA_URL.rstrip('/')
#                 tags_url = f"{base_url}/tags" if base_url.endswith('api') else f"{base_url}/api/tags"
                
#                 response = requests.get(tags_url, timeout=2)
#                 if response.status_code != 200:
#                     logger.warning(f"⚠️ Ollama servisi ({tags_url}) yanıt vermiyor. Durum: {response.status_code}")
#                 else:
#                     logger.info(f"✅ Ollama bağlantısı başarılı")
#             except Exception:
#                 logger.warning(
#                     f"⚠️ Ollama servisine ulaşılamadı ({cls.OLLAMA_URL})\n"
#                     "   Terminal'de 'ollama serve' komutunu çalıştırdığınızdan emin olun."
#                 )
#                 # Local modda bu hatayı vermek önemlidir ama programın çökmesini istemeyebiliriz.
#                 # Launcher zaten kontrol ettiği için burayı yumuşak geçiyoruz.

#         return is_valid

#     @classmethod
#     def get_system_info(cls) -> Dict[str, Any]:
#         """Sistem bilgilerini dictionary olarak döner."""
#         return {
#             "project": cls.PROJECT_NAME,
#             "version": cls.VERSION,
#             "provider": cls.AI_PROVIDER,
#             "access_level": cls.ACCESS_LEVEL,
#             "gpu_enabled": cls.USE_GPU,
#             "gpu_info": cls.GPU_INFO,
#             "cpu_count": cls.CPU_COUNT,
#             "debug_mode": cls.DEBUG_MODE,
#             "github_integration": bool(cls.GITHUB_TOKEN),
#             "agents": list(cls.AGENT_CONFIGS.keys()),
#             "ollama_models": {
#                 "text": cls.TEXT_MODEL,
#                 "vision": cls.VISION_MODEL,
#                 "coding": cls.CODING_MODEL,
#                 "embed": cls.LOCAL_VEK
#             }
#         }

#     @classmethod
#     def print_config_summary(cls) -> None:
#         """Yapılandırma özetini terminale yazdırır"""
#         print("\n" + "═" * 60)
#         print(f"  {cls.PROJECT_NAME} v{cls.VERSION} - Yapılandırma Özeti")
#         print("═" * 60)
#         print(f"  AI Sağlayıcı    : {cls.AI_PROVIDER.upper()}")
#         print(f"  GPU Desteği     : {'✓ ' + cls.GPU_INFO if cls.USE_GPU else '✗ CPU Modu'}")
#         print(f"  Erişim Seviyesi : {cls.ACCESS_LEVEL.upper()}")
#         print(f"  CPU Çekirdek    : {cls.CPU_COUNT}")
#         print(f"  Aktif Ajanlar   : {len(cls.AGENT_CONFIGS)}")
#         print(f"  Debug Modu      : {'Açık' if cls.DEBUG_MODE else 'Kapalı'}")
#         print(f"  GitHub Repo     : {cls.GITHUB_REPO}")
#         if cls.AI_PROVIDER == "ollama":
#             print("  ── Ollama Modelleri ──────────────────────────────")
#             print(f"  TEXT Model      : {cls.TEXT_MODEL}")
#             print(f"  VISION Model    : {cls.VISION_MODEL}")
#             print(f"  CODING Model    : {cls.CODING_MODEL}  ← Sidar")
#             print(f"  EMBED Model     : {cls.LOCAL_VEK}")
#         print("═" * 60 + "\n")

# # ═══════════════════════════════════════════════════════════════
# # BAŞLANGIÇ DOĞRULAMA
# # ═══════════════════════════════════════════════════════════════
# # Esas doğrulama Launcher tarafından LotusSystem.run() içinde yapılır.
# # Burada sadece modülün yüklendiğini bildiriyoruz.
# logger.info(
#     f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması yüklendi"
# )

# if Config.DEBUG_MODE:
#     Config.print_config_summary()

# # ═══════════════════════════════════════════════════════════════
# # BAŞLANGIÇ DOĞRULAMA
# # ═══════════════════════════════════════════════════════════════
# # Modül import edildiğinde varsayılan ayarlara göre bir kontrol yap
# # Ancak esas kontrol Launcher tarafından veya RuntimeContext başlatıldığında yapılmalıdır.
# if not Config.validate_critical_settings():
#     # Sadece varsayılan mod Gemini ise ve key yoksa uyar
#     if Config.AI_PROVIDER == "gemini":
#         logger.warning(
#             "🚨 Kritik ayar eksikliği tespit edildi. (Eğer Local mod kullanacaksanız bunu dikkate almayın)"
#         )
# else:
#     logger.info(
#         f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması yüklendi"
#     )

#     if Config.DEBUG_MODE:
#         Config.print_config_summary()