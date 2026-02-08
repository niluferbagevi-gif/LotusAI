import os
import sys
import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# --- UYARI FİLTRELEME ---
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
# Pynvml ve Torch uyarılarını bastırmak için filtreleme
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# --- LOGLAMA YAPILANDIRMASI ---
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log seviyesini .env'den alabilme özelliği
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "lotus_system.log",
            maxBytes=15 * 1024 * 1024, # 15MB
            backupCount=10,
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("LotusAI.Config")

# --- ORTAM DEĞİŞKENLERİ YÜKLEME ---
ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    logger.warning("⚠️ '.env' dosyası bulunamadı! Varsayılan ayarlar kullanılacak.")
else:
    load_dotenv(dotenv_path=ENV_PATH)

# --- YARDIMCI FONKSİYONLAR ---
def get_bool_env(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ["true", "1", "yes", "on"]

def get_int_env(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

# --- DONANIM HIZLANDIRMA (GPU) MERKEZİ KONTROLÜ ---
def check_hardware():
    """Donanım yeteneklerini kontrol eder ve detaylı bilgi döner."""
    has_cuda = False
    gpu_name = "N/A"
    
    # Kullanıcı .env üzerinden GPU'yu zorla kapattıysa hiç kontrol etme
    if not get_bool_env("USE_GPU", True):
        logger.info("ℹ️ GPU kullanımı .env ayarları ile devre dışı bırakıldı.")
        return False, "Disabled by User"

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"🚀 Donanım Hızlandırma Aktif: {gpu_name} ({gpu_count} GPU tespit edildi)")
        else:
            logger.info("ℹ️ GPU bulunamadı veya CUDA aktif değil, sistem CPU modunda çalışacak.")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch/CUDA hatası: {e}. Sistem CPU modunda devam edecek.")
        has_cuda = False
    
    return has_cuda, gpu_name

# Bu değişkenler global olarak bir kez hesaplanır ve diğer modüllerce kullanılır
HAS_CUDA, GPU_NAME = check_hardware()

class Config:
    """
    LotusAI Merkezi Yapılandırma Sınıfı.
    Sürüm 2.5.2 - Ajan Odaklı Anahtar Yönetimi
    """
    # --- GENEL SİSTEM BİLGİLERİ ---
    PROJECT_NAME = "LotusAI"
    VERSION = "2.5.2"
    DEBUG_MODE = get_bool_env("DEBUG_MODE", True)
    WORK_DIR = Path(os.getenv("WORK_DIR", BASE_DIR))

    # --- DİZİN YAPILANDIRMASI ---
    UPLOAD_DIR = WORK_DIR / "uploads"
    TEMPLATE_DIR = WORK_DIR / "templates"
    STATIC_DIR = WORK_DIR / "static"
    LOG_DIR = WORK_DIR / "logs"
    VOICES_DIR = WORK_DIR / "voices"
    FACES_DIR = WORK_DIR / "faces"
    MODELS_DIR = WORK_DIR / "models"
    DATA_DIR = WORK_DIR / "core" / "data"

    REQUIRED_DIRS = [UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR, FACES_DIR, MODELS_DIR, DATA_DIR]
    
    @classmethod
    def initialize_directories(cls):
        """Sistem için gerekli dizinleri oluşturur."""
        for folder in cls.REQUIRED_DIRS:
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Dizin hazırlama hatası ({folder.name}): {e}")

    # --- SİSTEM ZAMANLAMALARI ---
    CONVERSATION_TIMEOUT = get_int_env("CONVERSATION_TIMEOUT", 60)
    SYSTEM_CHECK_INTERVAL = get_int_env("SYSTEM_CHECK_INTERVAL", 300)

    # --- AI SAĞLAYICI AYARLARI ---
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
    
    # Global değişkeni kullan, tekrar kontrol etme
    USE_GPU = HAS_CUDA 
    GPU_INFO = GPU_NAME

    # --- GEMINI (GOOGLE) AYARLARI ---
    GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
    
    # --- AKILLI ANAHTAR YÖNETİMİ ---
    # 1. Önce doğrudan ana key'i kontrol et
    _MAIN_KEY = os.getenv("GEMINI_API_KEY")
    _USING_FALLBACK_KEY = False

    # 2. Eğer ana key yoksa, ajan keylerinden birini (Atlas) ana key yap
    if not _MAIN_KEY:
        _MAIN_KEY = os.getenv("GEMINI_API_KEY_ATLAS")
        if _MAIN_KEY:
            _USING_FALLBACK_KEY = True
            logger.info("ℹ️ Çoklu Ajan Modu: Genel işlemler için ATLAS anahtarı kullanılacak.")
    
    # 3. Hala yoksa diğerlerini dene
    if not _MAIN_KEY:
        _MAIN_KEY = os.getenv("GEMINI_API_KEY_SIDAR") or \
                    os.getenv("GEMINI_API_KEY_KURT") or \
                    os.getenv("GEMINI_API_KEY_KERBEROS")
        if _MAIN_KEY:
             _USING_FALLBACK_KEY = True

    HARDCODED_KEY = "" 
    if not _MAIN_KEY and HARDCODED_KEY:
        _MAIN_KEY = HARDCODED_KEY

    # Ajan Yapılandırması
    AGENT_CONFIGS: Dict[str, Any] = {
        "ATLAS": {"key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
        "SIDAR": {"key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
        "KURT": {"key": os.getenv("GEMINI_API_KEY_KURT", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
        "POYRAZ": {"key": os.getenv("GEMINI_API_KEY_POYRAZ", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
        "KERBEROS": {"key": os.getenv("GEMINI_API_KEY_KERBEROS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
        "GAYA": {"key": os.getenv("GEMINI_API_KEY_GAYA", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT}
    }

    # --- OLLAMA (YEREL AI) AYARLARI ---
    TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.1")
    VISION_MODEL = os.getenv("VISION_MODEL", "llava")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")

    # --- MANAGER (YÖNETİCİ) ÖZEL AYARLARI ---
    FACE_REC_MODEL = "cnn" if USE_GPU else "hog"
    LIVE_VISUAL_CHECK = get_bool_env("LIVE_VISUAL_CHECK", True)
    PATRON_IMAGE_PATH = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")

    FINANCE_MODE = get_bool_env("FINANCE_MODE", True)
    DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "TRY")

    USE_XTTS = get_bool_env("USE_XTTS", False)
    
    # --- GÜVENLİK ---
    API_AUTH_ENABLED = get_bool_env("API_AUTH_ENABLED", True)

    @classmethod
    def get_agent_settings(cls, agent_name: str) -> Dict[str, str]:
        """Ajan ayarlarını döner."""
        name_upper = agent_name.upper()
        if name_upper in cls.AGENT_CONFIGS:
            config = cls.AGENT_CONFIGS[name_upper].copy()
            if not config.get("key") and cls._MAIN_KEY:
                config["key"] = cls._MAIN_KEY
            return config
        
        return {"key": cls._MAIN_KEY, "model": cls.GEMINI_MODEL_DEFAULT}

    @classmethod
    def set_provider_mode(cls, mode: str):
        valid_modes = ["gemini", "ollama"]
        if mode.lower() in valid_modes:
            cls.AI_PROVIDER = mode.lower()
        else:
            logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")

    @classmethod
    def validate_critical_settings(cls) -> bool:
        """Hayati ayarların ve sistem bütünlüğünün kontrolü."""
        cls.initialize_directories()
        
        if cls.AI_PROVIDER == "gemini" and not cls._MAIN_KEY:
            logger.error("❌ KRİTİK HATA: Hiçbir GEMINI API Key bulunamadı!")
            return False 
            
        return True

# Başlangıç Doğrulaması
if not Config.validate_critical_settings():
    if Config.AI_PROVIDER == "gemini":
        logger.critical("🚨 Kritik API anahtarları eksik! Sistem çalışmayabilir.")
else:
    logger.info(f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması başarıyla tamamlandı.")

# import os
# import sys
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# from dotenv import load_dotenv
# from typing import Dict, Any, Optional

# # --- LOGLAMA YAPILANDIRMASI ---
# BASE_DIR = Path(__file__).resolve().parent
# LOG_DIR = BASE_DIR / "logs"
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# # Loglama formatını daha detaylı hale getirdik
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         RotatingFileHandler(
#             LOG_DIR / "lotus_system.log", 
#             maxBytes=10 * 1024 * 1024, # 10MB limit
#             backupCount=10, 
#             encoding="utf-8"
#         )
#     ]
# )
# logger = logging.getLogger("LotusAI.Config")

# # --- ORTAM DEĞİŞKENLERİ YÜKLEME ---
# ENV_PATH = BASE_DIR / ".env"
# if not ENV_PATH.exists():
#     logger.warning("⚠️ '.env' dosyası bulunamadı! Lütfen API anahtarlarını içeren bir .env dosyası oluşturun.")
# else:
#     load_dotenv(dotenv_path=ENV_PATH)

# # --- YARDIMCI FONKSİYONLAR ---
# def get_bool_env(key: str, default: bool = False) -> bool:
#     val = os.getenv(key, str(default)).lower()
#     return val in ["true", "1", "yes", "on"]

# def get_int_env(key: str, default: int = 0) -> int:
#     try:
#         return int(os.getenv(key, default))
#     except (ValueError, TypeError):
#         return default

# # --- DONANIM HIZLANDIRMA (GPU) KONTROLÜ ---
# def check_hardware():
#     has_cuda = False
#     gpu_name = "N/A"
#     try:
#         import torch
#         if torch.cuda.is_available():
#             has_cuda = True
#             gpu_name = torch.cuda.get_device_name(0)
#             logger.info(f"🚀 Donanım Hızlandırma Aktif: {gpu_name}")
#         else:
#             logger.info("ℹ️ GPU bulunamadı, sistem CPU modunda çalışacak.")
#     except ImportError:
#         logger.warning("⚠️ PyTorch bulunamadı. AI işlemleri için GPU desteği kontrol edilemedi.")
#     return has_cuda, gpu_name

# HAS_CUDA, GPU_NAME = check_hardware()

# class Config:
#     """
#     LotusAI Merkezi Yapılandırma Sınıfı.
#     Sürüm 2.4 - Profesyonel Donanım ve Ajan Yönetimi
#     """
#     # --- GENEL SİSTEM BİLGİLERİ ---
#     PROJECT_NAME = "LotusAI"
#     VERSION = "2.4"
#     DEBUG_MODE = get_bool_env("DEBUG_MODE", True)
#     WORK_DIR = Path(os.getenv("WORK_DIR", BASE_DIR))

#     # --- DİZİN YAPILANDIRMASI ---
#     UPLOAD_DIR = WORK_DIR / "uploads"
#     TEMPLATE_DIR = WORK_DIR / "templates"
#     STATIC_DIR = WORK_DIR / "static"
#     LOG_DIR = WORK_DIR / "logs"
#     VOICES_DIR = WORK_DIR / "voices"
#     FACES_DIR = WORK_DIR / "faces"
#     MODELS_DIR = WORK_DIR / "models" # Yerel modeller için yeni dizin

#     # Gerekli Dizinleri Otomatik Oluştur
#     for folder in [UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR, FACES_DIR, MODELS_DIR]:
#         try:
#             folder.mkdir(parents=True, exist_ok=True)
#         except Exception as e:
#             logger.error(f"❌ Dizin oluşturma hatası ({folder.name}): {e}")

#     # --- SİSTEM ZAMANLAMALARI ---
#     CONVERSATION_TIMEOUT = get_int_env("CONVERSATION_TIMEOUT", 60) # Saniye cinsinden
#     SYSTEM_CHECK_INTERVAL = get_int_env("SYSTEM_CHECK_INTERVAL", 300) # 5 Dakika

#     # --- AI SAĞLAYICI AYARLARI (MODÜLER) ---
#     AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
#     USE_GPU = get_bool_env("USE_GPU", True) and HAS_CUDA
#     GPU_INFO = GPU_NAME

#     # --- GEMINI (GOOGLE) AYARLARI ---
#     GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
#     GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
#     _MAIN_KEY = os.getenv("GEMINI_API_KEY", "")

#     # Ajanlara özel modeller ve anahtarlar (Dinamik erişim için temel sözlük)
#     # Yeni ajan eklendiğinde .env üzerinden otomatik tanınır.
#     AGENT_CONFIGS: Dict[str, Any] = {
#         "ATLAS": {"key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
#         "SIDAR": {"key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
#         "KURT": {"key": os.getenv("GEMINI_API_KEY_KURT", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
#         "POYRAZ": {"key": os.getenv("GEMINI_API_KEY_POYRAZ", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
#         "KERBEROS": {"key": os.getenv("GEMINI_API_KEY_KERBEROS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
#         "GAYA": {"key": os.getenv("GEMINI_API_KEY_GAYA", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT}
#     }

#     # --- OLLAMA (YEREL AI) AYARLARI ---
#     TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.1")
#     VISION_MODEL = os.getenv("VISION_MODEL", "llava")
#     OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")

#     # --- MANAGER (YÖNETİCİ) ÖZEL AYARLARI ---
#     # Camera Manager
#     CAMERA_INDEX = get_int_env("CAMERA_INDEX", 0)
#     FACE_REC_MODEL = "cnn" if USE_GPU else "hog"
#     LIVE_VISUAL_CHECK = get_bool_env("LIVE_VISUAL_CHECK", True)
#     PATRON_IMAGE_PATH = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")

#     # Finance & Accounting Manager
#     FINANCE_MODE = get_bool_env("FINANCE_MODE", True)
#     DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "TRY")

#     # Messaging & Media
#     USE_XTTS = get_bool_env("USE_XTTS", False)
#     META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
#     WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

#     @classmethod
#     def get_agent_settings(cls, agent_name: str) -> Dict[str, str]:
#         """
#         Belirtilen ajan için konfigürasyonu döner. 
#         Eğer listede yoksa varsayılan ayarları oluşturur.
#         """
#         name_upper = agent_name.upper()
#         if name_upper in cls.AGENT_CONFIGS:
#             return cls.AGENT_CONFIGS[name_upper]
        
#         # Dinamik olarak .env'den çekmeyi dene
#         dynamic_key = os.getenv(f"GEMINI_API_KEY_{name_upper}", cls._MAIN_KEY)
#         return {"key": dynamic_key, "model": cls.GEMINI_MODEL_DEFAULT}

#     @classmethod
#     def set_provider_mode(cls, mode: str):
#         """AI sağlayıcı modunu çalışma anında değiştirir."""
#         valid_modes = ["gemini", "ollama"]
#         if mode.lower() in valid_modes:
#             cls.AI_PROVIDER = mode.lower()
#             logger.info(f"🔄 AI Sağlayıcı Değiştirildi: {cls.AI_PROVIDER.upper()}")
#         else:
#             logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")

#     @classmethod
#     def validate_critical_settings(cls) -> bool:
#         """Sistemin çalışması için hayati olan ayarları kontrol eder."""
#         is_valid = True
#         if cls.AI_PROVIDER == "gemini" and not cls._MAIN_KEY:
#             logger.error("❌ HATA: Ana GEMINI_API_KEY eksik!")
#             is_valid = False
        
#         if cls.LIVE_VISUAL_CHECK and not cls.PATRON_IMAGE_PATH.exists():
#             logger.warning(f"⚠️ Görsel doğrulama aktif ancak {cls.PATRON_IMAGE_PATH.name} bulunamadı.")
#             # Bu kritik hata değil ama kullanıcıyı uyarır
            
#         return is_valid

# # Başlangıç doğrulaması
# if not Config.validate_critical_settings():
#     logger.warning("🚨 Bazı kritik ayarlar eksik. Sistem kısıtlı modda çalışabilir.")
# else:
#     logger.info("✅ Tüm kritik sistem ayarları doğrulandı.")