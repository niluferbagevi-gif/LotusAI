"""
LotusAI Merkezi Yapılandırma Modülü
Sürüm: 2.6.0 (Tekil sürüm kaynağına geçirildi)
Açıklama: Sistem ayarları, API anahtarları, donanım tespiti, dizin yönetimi ve erişim seviyesi
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
# TEMEL DİZİN YAPILANDIRMASI
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOGLAMA SİSTEMİ
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "lotus_system.log",
            maxBytes=15 * 1024 * 1024,  # 15MB
            backupCount=10,
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("LotusAI.Config")

# ═══════════════════════════════════════════════════════════════
# ORTAM DEĞİŞKENLERİ
# ═══════════════════════════════════════════════════════════════
ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    logger.warning("⚠️ '.env' dosyası bulunamadı! Varsayılan ayarlar kullanılacak.")
else:
    load_dotenv(dotenv_path=ENV_PATH)
    logger.info(f"✅ Ortam değişkenleri yüklendi: {ENV_PATH}")


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
        logger.warning(f"⚠️ '{key}' geçersiz değer, varsayılan kullanılıyor: {default}")
        return default


def get_list_env(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    if default is None:
        default = []
    value = os.getenv(key, "")
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


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
    Özellikler:
    - Çoklu API anahtarı yönetimi
    - Ajan bazlı konfigürasyon
    - Otomatik donanım tespiti
    - Dizin yönetimi
    - CODING_MODEL desteği (Sidar özel)
    - Binance ve Instagram API entegrasyonu
    - **Erişim seviyesi (OpenClaw stili)**
    """

    # ───────────────────────────────────────────────────────────
    # GENEL SİSTEM BİLGİLERİ
    # ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = "LotusAI"
    VERSION: str = "2.6.0"
    DEBUG_MODE: bool = get_bool_env("DEBUG_MODE", False)
    WORK_DIR: Path = Path(os.getenv("WORK_DIR", BASE_DIR))

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

    REQUIRED_DIRS: List[Path] = [
        UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR,
        FACES_DIR, MODELS_DIR, DATA_DIR
    ]

    # ───────────────────────────────────────────────────────────
    # SİSTEM ZAMANLAMALARI
    # ───────────────────────────────────────────────────────────
    CONVERSATION_TIMEOUT: int = get_int_env("CONVERSATION_TIMEOUT", 60)
    SYSTEM_CHECK_INTERVAL: int = get_int_env("SYSTEM_CHECK_INTERVAL", 300)

    # ───────────────────────────────────────────────────────────
    # AI SAĞLAYICI AYARLARI
    # ───────────────────────────────────────────────────────────
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini").lower()

    # Donanım bilgileri
    USE_GPU: bool = HARDWARE.has_cuda
    GPU_INFO: str = HARDWARE.gpu_name
    CPU_COUNT: int = HARDWARE.cpu_count

    # ───────────────────────────────────────────────────────────
    # ERİŞİM SEVİYESİ (OpenClaw stili)
    # ───────────────────────────────────────────────────────────
    ACCESS_LEVEL: str = os.getenv("ACCESS_LEVEL", AccessLevel.SANDBOX).lower()
    # Geçerlilik kontrolü
    if ACCESS_LEVEL not in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
        ACCESS_LEVEL = AccessLevel.SANDBOX
        logger.warning(f"Geçersiz ACCESS_LEVEL, varsayılan {ACCESS_LEVEL} kullanılıyor.")

    # ───────────────────────────────────────────────────────────
    # GEMINI (GOOGLE) MODEL AYARLARI
    # ───────────────────────────────────────────────────────────
    GEMINI_MODEL_DEFAULT: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
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
                logger.info(f"ℹ️ Ana API anahtarı bulunamadı, {key_name} kullanılacak.")
            break

    # ───────────────────────────────────────────────────────────
    # OLLAMA (YEREL AI) MODEL AYARLARI
    # ───────────────────────────────────────────────────────────
    # Genel metin modeli (gemma2:9b)
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemma2:9b")
    # Görsel analiz modeli (llama3.2-vision)
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llama3.2-vision")
    # Sidar'a özel kodlama modeli (qwen2.5-coder:7b)
    CODING_MODEL: str = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
    # Vektörleme modeli
    LOCAL_VEK: str = os.getenv("LOCAL_VEK", "nomic-embed-text")
    # Bağlantı ayarları
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
            # Ollama modunda CODING_MODEL kullanır
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
    # MANAGER (YÖNETİCİ) ÖZEL AYARLARI
    # ───────────────────────────────────────────────────────────
    FACE_REC_MODEL: str = "cnn" if USE_GPU else "hog"
    LIVE_VISUAL_CHECK: bool = get_bool_env("LIVE_VISUAL_CHECK", True)
    PATRON_IMAGE_PATH: Path = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")

    # Finans ayarları
    FINANCE_MODE: bool = get_bool_env("FINANCE_MODE", True)
    DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "TRY")
    SUPPORTED_CURRENCIES: List[str] = get_list_env(
        "SUPPORTED_CURRENCIES",
        ["TRY", "USD", "EUR", "GBP"]
    )
    
    # Binance API Ayarları
    BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

    # Instagram Hedef Hesap
    INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "lotusbagevi")

    # Ses ayarları
    USE_XTTS: bool = get_bool_env("USE_XTTS", False)
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
    VOICE_SPEED: int = get_int_env("VOICE_SPEED", 150)

    # ───────────────────────────────────────────────────────────
    # GÜVENLİK AYARLARI
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
        """
        Belirli bir ajan için ayarları döner.
        Ollama modunda SIDAR için CODING_MODEL otomatik seçilir.

        Args:
            agent_name: Ajan adı (büyük/küçük harf duyarsız)

        Returns:
            Ajan ayarları (key, model, role)
        """
        name_upper = agent_name.upper()

        if name_upper in cls.AGENT_CONFIGS:
            config = cls.AGENT_CONFIGS[name_upper].copy()

            # Eğer ajan anahtarı yoksa ana anahtarı kullan
            if not config.get("key") and cls._MAIN_KEY:
                config["key"] = cls._MAIN_KEY
                logger.debug(f"Ajan {agent_name} için ana anahtar kullanılıyor")

            # Ollama modunda Sidar'ın özel coding modelini etkinleştir
            if cls.AI_PROVIDER == "ollama" and name_upper == "SIDAR":
                config["active_model"] = config.get("ollama_model", cls.CODING_MODEL)
                logger.debug(
                    f"SIDAR Ollama modu: {config['active_model']} (CODING_MODEL)"
                )
            elif cls.AI_PROVIDER == "ollama":
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
        """
        Ollama modunda belirli bir ajan için kullanılacak modeli döner.

        Args:
            agent_name: Ajan adı

        Returns:
            Model adı string
        """
        name_upper = agent_name.upper()
        if name_upper == "SIDAR":
            return cls.CODING_MODEL
        # VISION gerektiren ajanlar için genişletilebilir
        return cls.TEXT_MODEL

    @classmethod
    def set_provider_mode(cls, mode: str) -> None:
        """
        AI sağlayıcı modunu ayarlar.

        Args:
            mode: 'online', 'gemini', 'local' veya 'ollama'
        """
        mode_map = {
            "online": "gemini",
            "local": "ollama",
            "ollama": "ollama",
            "gemini": "gemini"
        }
        m_lower = mode.lower()
        if m_lower in mode_map:
            cls.AI_PROVIDER = mode_map[m_lower]
            logger.info(f"✅ AI Sağlayıcı modu: {cls.AI_PROVIDER.upper()}")
        else:
            logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")
            logger.info(f"   Geçerli modlar: {', '.join(mode_map.keys())}")

    @classmethod
    def set_access_level(cls, level: str) -> None:
        """
        Erişim seviyesini ayarlar (launcher'dan çağrılır).
        
        Args:
            level: "restricted", "sandbox" veya "full"
        """
        level_lower = level.lower()
        if level_lower in [AccessLevel.RESTRICTED, AccessLevel.SANDBOX, AccessLevel.FULL]:
            cls.ACCESS_LEVEL = level_lower
            logger.info(f"✅ Erişim seviyesi: {cls.ACCESS_LEVEL}")
        else:
            logger.error(f"❌ Geçersiz erişim seviyesi: {level}, varsayılan sandbox kullanılacak")
            cls.ACCESS_LEVEL = AccessLevel.SANDBOX

    @classmethod
    def validate_critical_settings(cls) -> bool:
        """
        Kritik sistem ayarlarını doğrular.

        Returns:
            Tüm kritik ayarlar geçerliyse True
        """
        is_valid = True

        # 1. Dizinleri oluştur
        if not cls.initialize_directories():
            logger.warning("⚠️ Bazı dizinler oluşturulamadı")
            is_valid = False

        # 2. API anahtarı kontrolü (Gemini modu için)
        if cls.AI_PROVIDER == "gemini":
            if not cls._MAIN_KEY:
                logger.error(
                    "❌ KRİTİK HATA: Hiçbir GEMINI API anahtarı bulunamadı!\n"
                    "   .env dosyasına GEMINI_API_KEY ekleyin."
                )
                is_valid = False
            else:
                if len(cls._MAIN_KEY) < 30:
                    logger.warning("⚠️ API anahtarı çok kısa görünüyor, geçersiz olabilir")

        # 3. Patron resmi kontrolü
        if cls.LIVE_VISUAL_CHECK:
            if not cls.PATRON_IMAGE_PATH.exists():
                logger.warning(
                    f"⚠️ Patron resmi bulunamadı: {cls.PATRON_IMAGE_PATH}\n"
                    "   Yüz tanıma devre dışı bırakılabilir."
                )

        # 4. Ollama kontrolü (local mod için)
        if cls.AI_PROVIDER == "ollama":
            try:
                import requests
                response = requests.get(
                    "http://localhost:11434/api/tags",
                    timeout=2
                )
                if response.status_code != 200:
                    logger.warning("⚠️ Ollama servisi yanıt vermiyor")
                else:
                    logger.info(
                        f"✅ Ollama bağlantısı başarılı | "
                        f"TEXT: {cls.TEXT_MODEL} | "
                        f"VISION: {cls.VISION_MODEL} | "
                        f"CODING (Sidar): {cls.CODING_MODEL}"
                    )
            except Exception:
                logger.warning(
                    "⚠️ Ollama servisi kontrol edilemedi\n"
                    "   Terminal'de 'ollama serve' komutunu çalıştırın"
                )

        return is_valid

    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Sistem bilgilerini dictionary olarak döner."""
        return {
            "project": cls.PROJECT_NAME,
            "version": cls.VERSION,
            "provider": cls.AI_PROVIDER,
            "access_level": cls.ACCESS_LEVEL,
            "gpu_enabled": cls.USE_GPU,
            "gpu_info": cls.GPU_INFO,
            "cpu_count": cls.CPU_COUNT,
            "debug_mode": cls.DEBUG_MODE,
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
        """Yapılandırma özetini terminale yazdırır"""
        print("\n" + "═" * 60)
        print(f"  {cls.PROJECT_NAME} v{cls.VERSION} - Yapılandırma Özeti")
        print("═" * 60)
        print(f"  AI Sağlayıcı    : {cls.AI_PROVIDER.upper()}")
        print(f"  GPU Desteği     : {'✓ ' + cls.GPU_INFO if cls.USE_GPU else '✗ CPU Modu'}")
        print(f"  Erişim Seviyesi : {cls.ACCESS_LEVEL.upper()}")
        print(f"  CPU Çekirdek    : {cls.CPU_COUNT}")
        print(f"  Aktif Ajanlar   : {len(cls.AGENT_CONFIGS)}")
        print(f"  Debug Modu      : {'Açık' if cls.DEBUG_MODE else 'Kapalı'}")
        if cls.AI_PROVIDER == "ollama":
            print("  ── Ollama Modelleri ──────────────────────────────")
            print(f"  TEXT Model      : {cls.TEXT_MODEL}")
            print(f"  VISION Model    : {cls.VISION_MODEL}")
            print(f"  CODING Model    : {cls.CODING_MODEL}  ← Sidar")
            print(f"  EMBED Model     : {cls.LOCAL_VEK}")
        print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# BAŞLANGIÇ DOĞRULAMA
# ═══════════════════════════════════════════════════════════════
if not Config.validate_critical_settings():
    if Config.AI_PROVIDER == "gemini":
        logger.critical(
            "🚨 Kritik ayar eksik! Sistem düzgün çalışmayabilir.\n"
            "   Lütfen .env dosyasını kontrol edin."
        )
else:
    logger.info(
        f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması tamamlandı"
    )

    if Config.DEBUG_MODE:
        Config.print_config_summary()

# """
# LotusAI Merkezi Yapılandırma Modülü
# Sürüm: 2.5.3
# Açıklama: Sistem ayarları, API anahtarları, donanım tespiti ve dizin yönetimi
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
#     """
#     Environment değişkenini boolean'a çevirir.
    
#     Args:
#         key: Değişken adı
#         default: Varsayılan değer
    
#     Returns:
#         Boolean değer
#     """
#     val = os.getenv(key, str(default)).lower()
#     return val in ["true", "1", "yes", "on"]


# def get_int_env(key: str, default: int = 0) -> int:
#     """
#     Environment değişkenini integer'a çevirir.
    
#     Args:
#         key: Değişken adı
#         default: Varsayılan değer
    
#     Returns:
#         Integer değer
#     """
#     try:
#         return int(os.getenv(key, default))
#     except (ValueError, TypeError):
#         logger.warning(f"⚠️ '{key}' geçersiz değer, varsayılan kullanılıyor: {default}")
#         return default


# def get_list_env(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
#     """
#     Environment değişkenini liste'ye çevirir.
    
#     Args:
#         key: Değişken adı
#         default: Varsayılan liste
#         separator: Ayırıcı karakter
    
#     Returns:
#         String listesi
#     """
#     if default is None:
#         default = []
    
#     value = os.getenv(key, "")
#     if not value:
#         return default
    
#     return [item.strip() for item in value.split(separator) if item.strip()]


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
#     """
#     Sistem donanımını kontrol eder ve detaylı bilgi döner.
    
#     Returns:
#         HardwareInfo nesnesi
#     """
#     info = HardwareInfo(has_cuda=False, gpu_name="N/A")
    
#     # Kullanıcı GPU'yu manuel olarak devre dışı bıraktıysa
#     if not get_bool_env("USE_GPU", True):
#         logger.info("ℹ️ GPU kullanımı .env ayarları ile devre dışı bırakıldı.")
#         info.gpu_name = "Devre Dışı (Kullanıcı)"
#         return info
    
#     # PyTorch/CUDA kontrolü
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
    
#     # CPU bilgisi
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
    
#     Sürüm: 2.5.3
#     Özellikler:
#     - Çoklu API anahtarı yönetimi
#     - Ajan bazlı konfigürasyon
#     - Otomatik donanım tespiti
#     - Dizin yönetimi
#     """
    
#     # ───────────────────────────────────────────────────────────
#     # GENEL SİSTEM BİLGİLERİ
#     # ───────────────────────────────────────────────────────────
#     PROJECT_NAME: str = "LotusAI"
#     VERSION: str = "2.5.3"
#     DEBUG_MODE: bool = get_bool_env("DEBUG_MODE", False)
#     WORK_DIR: Path = Path(os.getenv("WORK_DIR", BASE_DIR))
    
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
#     AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini").lower()
    
#     # Donanım bilgileri
#     USE_GPU: bool = HARDWARE.has_cuda
#     GPU_INFO: str = HARDWARE.gpu_name
#     CPU_COUNT: int = HARDWARE.cpu_count
    
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
    
#     # Öncelik sırası: Ana key > Atlas > Diğer ajanlar
#     _KEY_PRIORITY = [
#         "GEMINI_API_KEY",
#         "GEMINI_API_KEY_ATLAS",
#         "GEMINI_API_KEY_SIDAR",
#         "GEMINI_API_KEY_KURT",
#         "GEMINI_API_KEY_KERBEROS",
#         "GEMINI_API_KEY_POYRAZ",
#         "GEMINI_API_KEY_GAYA"
#     ]
    
#     # Ana anahtarı bul
#     for key_name in _KEY_PRIORITY:
#         _MAIN_KEY = os.getenv(key_name)
#         if _MAIN_KEY:
#             if key_name != "GEMINI_API_KEY":
#                 _USING_FALLBACK_KEY = True
#                 logger.info(f"ℹ️ Ana API anahtarı bulunamadı, {key_name} kullanılacak.")
#             break
    
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
#     # OLLAMA (YEREL AI) AYARLARI
#     # ───────────────────────────────────────────────────────────
#     TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemma2:9b")
#     VISION_MODEL: str = os.getenv("VISION_MODEL", "llama3.2-vision")
#     CODING_MODEL: str = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
#     LOCAL_VEK: str = os.getenv("LOCAL_VEK", "nomic-embed-text")
#     OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
#     OLLAMA_TIMEOUT: int = get_int_env("OLLAMA_TIMEOUT", 30)
    
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
    
#     # Ses ayarları
#     USE_XTTS: bool = get_bool_env("USE_XTTS", False)
#     TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
#     VOICE_SPEED: int = get_int_env("VOICE_SPEED", 150)
    
#     # ───────────────────────────────────────────────────────────
#     # GÜVENLİK AYARLARI
#     # ───────────────────────────────────────────────────────────
#     API_AUTH_ENABLED: bool = get_bool_env("API_AUTH_ENABLED", True)
#     MAX_LOGIN_ATTEMPTS: int = get_int_env("MAX_LOGIN_ATTEMPTS", 3)
#     SESSION_TIMEOUT: int = get_int_env("SESSION_TIMEOUT", 3600)  # saniye
    
#     # ───────────────────────────────────────────────────────────
#     # METOTLAR
#     # ───────────────────────────────────────────────────────────
    
#     @classmethod
#     def initialize_directories(cls) -> bool:
#         """
#         Sistem için gerekli dizinleri oluşturur.
        
#         Returns:
#             Başarılı ise True
#         """
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
        
#         Args:
#             agent_name: Ajan adı (büyük/küçük harf duyarsız)
        
#         Returns:
#             Ajan ayarları (key, model, role)
#         """
#         name_upper = agent_name.upper()
        
#         if name_upper in cls.AGENT_CONFIGS:
#             config = cls.AGENT_CONFIGS[name_upper].copy()
            
#             # Eğer ajan anahtarı yoksa ana anahtarı kullan
#             if not config.get("key") and cls._MAIN_KEY:
#                 config["key"] = cls._MAIN_KEY
#                 logger.debug(f"Ajan {agent_name} için ana anahtar kullanılıyor")
            
#             return config
        
#         # Bilinmeyen ajan için varsayılan ayarlar
#         logger.warning(f"⚠️ Bilinmeyen ajan: {agent_name}, varsayılan ayarlar kullanılıyor")
#         return {
#             "key": cls._MAIN_KEY or "",
#             "model": cls.GEMINI_MODEL_DEFAULT,
#             "role": "Bilinmiyor"
#         }
    
#     @classmethod
#     def set_provider_mode(cls, mode: str) -> None:
#         """
#         AI sağlayıcı modunu ayarlar.
        
#         Args:
#             mode: 'online', 'gemini', 'local' veya 'ollama'
#         """
#         mode_map = {
#             "online": "gemini",
#             "local": "ollama",
#             "ollama": "ollama",
#             "gemini": "gemini"
#         }
        
#         m_lower = mode.lower()
        
#         if m_lower in mode_map:
#             cls.AI_PROVIDER = mode_map[m_lower]
#             cls.AI_PROVIDER = mode_map[m_lower]
#             logger.info(f"✅ AI Sağlayıcı modu: {cls.AI_PROVIDER.upper()}")
#         else:
#             logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")
#             logger.info(f"   Geçerli modlar: {', '.join(mode_map.keys())}")
    
#     @classmethod
#     def validate_critical_settings(cls) -> bool:
#         """
#         Kritik sistem ayarlarını doğrular.
        
#         Returns:
#             Tüm kritik ayarlar geçerliyse True
#         """
#         is_valid = True
        
#         # 1. Dizinleri oluştur
#         if not cls.initialize_directories():
#             logger.warning("⚠️ Bazı dizinler oluşturulamadı")
#             is_valid = False
        
#         # 2. API anahtarı kontrolü (Gemini modu için)
#         if cls.AI_PROVIDER == "gemini":
#             if not cls._MAIN_KEY:
#                 logger.error(
#                     "❌ KRİTİK HATA: Hiçbir GEMINI API anahtarı bulunamadı!\n"
#                     "   .env dosyasına GEMINI_API_KEY ekleyin."
#                 )
#                 is_valid = False
#             else:
#                 # Anahtar uzunluk kontrolü
#                 if len(cls._MAIN_KEY) < 30:
#                     logger.warning("⚠️ API anahtarı çok kısa görünüyor, geçersiz olabilir")
        
#         # 3. Patron resmi kontrolü (eğer yüz tanıma aktifse)
#         if cls.LIVE_VISUAL_CHECK:
#             if not cls.PATRON_IMAGE_PATH.exists():
#                 logger.warning(
#                     f"⚠️ Patron resmi bulunamadı: {cls.PATRON_IMAGE_PATH}\n"
#                     "   Yüz tanıma devre dışı bırakılabilir."
#                 )
        
#         # 4. Ollama kontrolü (local mod için)
#         if cls.AI_PROVIDER == "ollama":
#             try:
#                 import requests
#                 response = requests.get(
#                     "http://localhost:11434/api/tags",
#                     timeout=2
#                 )
#                 if response.status_code != 200:
#                     logger.warning("⚠️ Ollama servisi yanıt vermiyor")
#             except:
#                 logger.warning(
#                     "⚠️ Ollama servisi kontrol edilemedi\n"
#                     "   Terminal'de 'ollama serve' komutunu çalıştırın"
#                 )
        
#         return is_valid
    
#     @classmethod
#     def get_system_info(cls) -> Dict[str, Any]:
#         """
#         Sistem bilgilerini dictionary olarak döner.
        
#         Returns:
#             Sistem bilgileri
#         """
#         return {
#             "project": cls.PROJECT_NAME,
#             "version": cls.VERSION,
#             "provider": cls.AI_PROVIDER,
#             "gpu_enabled": cls.USE_GPU,
#             "gpu_info": cls.GPU_INFO,
#             "cpu_count": cls.CPU_COUNT,
#             "debug_mode": cls.DEBUG_MODE,
#             "agents": list(cls.AGENT_CONFIGS.keys())
#         }
    
#     @classmethod
#     def print_config_summary(cls) -> None:
#         """Yapılandırma özetini terminale yazdırır"""
#         print("\n" + "═" * 60)
#         print(f"  {cls.PROJECT_NAME} v{cls.VERSION} - Yapılandırma Özeti")
#         print("═" * 60)
#         print(f"  AI Sağlayıcı    : {cls.AI_PROVIDER.upper()}")
#         print(f"  GPU Desteği     : {'✓ ' + cls.GPU_INFO if cls.USE_GPU else '✗ CPU Modu'}")
#         print(f"  CPU Çekirdek    : {cls.CPU_COUNT}")
#         print(f"  Aktif Ajanlar   : {len(cls.AGENT_CONFIGS)}")
#         print(f"  Debug Modu      : {'Açık' if cls.DEBUG_MODE else 'Kapalı'}")
#         print("═" * 60 + "\n")


# # ═══════════════════════════════════════════════════════════════
# # BAŞLANGIÇ DOĞRULAMA
# # ═══════════════════════════════════════════════════════════════
# if not Config.validate_critical_settings():
#     if Config.AI_PROVIDER == "gemini":
#         logger.critical(
#             "🚨 Kritik ayar eksik! Sistem düzgün çalışmayabilir.\n"
#             "   Lütfen .env dosyasını kontrol edin."
#         )
# else:
#     logger.info(
#         f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması tamamlandı"
#     )
    
#     if Config.DEBUG_MODE:
#         Config.print_config_summary()


# """
# LotusAI Merkezi Yapılandırma Modülü
# Sürüm: 2.5.3
# Açıklama: Sistem ayarları, API anahtarları, donanım tespiti ve dizin yönetimi
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
#     """
#     Environment değişkenini boolean'a çevirir.
#     """
#     val = os.getenv(key, str(default)).lower()
#     return val in ["true", "1", "yes", "on"]


# def get_int_env(key: str, default: int = 0) -> int:
#     """
#     Environment değişkenini integer'a çevirir.
#     """
#     try:
#         return int(os.getenv(key, default))
#     except (ValueError, TypeError):
#         logger.warning(f"⚠️ '{key}' geçersiz değer, varsayılan kullanılıyor: {default}")
#         return default


# def get_list_env(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
#     """
#     Environment değişkenini liste'ye çevirir.
#     """
#     if default is None:
#         default = []
    
#     value = os.getenv(key, "")
#     if not value:
#         return default
    
#     return [item.strip() for item in value.split(separator) if item.strip()]


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
#     """
#     Sistem donanımını kontrol eder ve detaylı bilgi döner.
#     """
#     info = HardwareInfo(has_cuda=False, gpu_name="N/A")
    
#     # Kullanıcı GPU'yu manuel olarak devre dışı bıraktıysa
#     if not get_bool_env("USE_GPU", True):
#         logger.info("ℹ️ GPU kullanımı .env ayarları ile devre dışı bırakıldı.")
#         info.gpu_name = "Devre Dışı (Kullanıcı)"
#         return info
    
#     # PyTorch/CUDA kontrolü
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
    
#     # CPU bilgisi
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
    
#     Sürüm: 2.5.3 (Tr-Ollama)
#     Özellikler:
#     - Çoklu API anahtarı yönetimi
#     - Ajan bazlı konfigürasyon
#     - Otomatik donanım tespiti
#     - Dizin yönetimi
#     - Türkçe Dil Desteği
#     """
    
#     # ───────────────────────────────────────────────────────────
#     # GENEL SİSTEM BİLGİLERİ
#     # ───────────────────────────────────────────────────────────
#     PROJECT_NAME: str = "LotusAI"
#     VERSION: str = "2.5.3"
#     DEBUG_MODE: bool = get_bool_env("DEBUG_MODE", False)
#     WORK_DIR: Path = Path(os.getenv("WORK_DIR", BASE_DIR))
#     LANGUAGE: str = os.getenv("LANGUAGE", "tr").lower()  # Dil Ayarı
    
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
#     AI_PROVIDER: str = os.getenv("AI_PROVIDER", "ollama").lower()
    
#     # Donanım bilgileri
#     USE_GPU: bool = HARDWARE.has_cuda
#     GPU_INFO: str = HARDWARE.gpu_name
#     CPU_COUNT: int = HARDWARE.cpu_count
    
#     # ───────────────────────────────────────────────────────────
#     # GEMINI (GOOGLE) MODEL AYARLARI
#     # ───────────────────────────────────────────────────────────
#     GEMINI_MODEL_DEFAULT: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
#     GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
#     GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
#     GEMINI_MAX_TOKENS: int = get_int_env("GEMINI_MAX_TOKENS", 8192)
    
#     # ───────────────────────────────────────────────────────────
#     # OLLAMA (YEREL AI) AYARLARI
#     # ───────────────────────────────────────────────────────────
#     TEXT_MODEL: str = os.getenv("TEXT_MODEL", "llama3.1")
#     VISION_MODEL: str = os.getenv("VISION_MODEL", "llava")
#     EMBEDDING_MODEL: str = os.getenv("LOCAL_VEK", "nomic-embed-text")
#     OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
#     OLLAMA_TIMEOUT: int = get_int_env("OLLAMA_TIMEOUT", 60)

#     # ───────────────────────────────────────────────────────────
#     # API ANAHTAR YÖNETİMİ (AKILLI FALLBACK)
#     # ───────────────────────────────────────────────────────────
#     _MAIN_KEY: Optional[str] = None
#     _USING_FALLBACK_KEY: bool = False
    
#     # Öncelik sırası: Ana key > Atlas > Diğer ajanlar
#     _KEY_PRIORITY = [
#         "GEMINI_API_KEY",
#         "GEMINI_API_KEY_ATLAS",
#         "GEMINI_API_KEY_SIDAR",
#         "GEMINI_API_KEY_KURT",
#         "GEMINI_API_KEY_KERBEROS",
#         "GEMINI_API_KEY_POYRAZ",
#         "GEMINI_API_KEY_GAYA"
#     ]
    
#     # Ana anahtarı bul
#     for key_name in _KEY_PRIORITY:
#         _MAIN_KEY = os.getenv(key_name)
#         if _MAIN_KEY:
#             if key_name != "GEMINI_API_KEY":
#                 _USING_FALLBACK_KEY = True
#                 logger.info(f"ℹ️ Ana API anahtarı bulunamadı, {key_name} kullanılacak.")
#             break
    
#     # ───────────────────────────────────────────────────────────
#     # AJAN YAPILANDIRMASI
#     # ───────────────────────────────────────────────────────────
#     @classmethod
#     def get_agent_config(cls, agent_name: str) -> Dict[str, str]:
#         """
#         Sağlayıcıya (Ollama/Gemini) göre dinamik ajan konfigürasyonu
#         """
#         base_role = "Asistan"
        
#         # Eğer sağlayıcı Ollama ise, tüm ajanlar yerel modeli kullanır
#         if cls.AI_PROVIDER == "ollama":
#             return {
#                 "key": "", # Local modda anahtara gerek yok
#                 "model": cls.TEXT_MODEL,
#                 "role": base_role,
#                 "provider": "ollama"
#             }
        
#         # Gemini Modu
#         name_upper = agent_name.upper()
#         if name_upper in cls.AGENT_CONFIGS_GEMINI:
#             config = cls.AGENT_CONFIGS_GEMINI[name_upper].copy()
#             if not config.get("key") and cls._MAIN_KEY:
#                 config["key"] = cls._MAIN_KEY
#             config["provider"] = "gemini"
#             return config
            
#         return {
#             "key": cls._MAIN_KEY or "",
#             "model": cls.GEMINI_MODEL_DEFAULT,
#             "role": base_role,
#             "provider": "gemini"
#         }

#     AGENT_CONFIGS_GEMINI: Dict[str, Dict[str, str]] = {
#         "ATLAS": {
#             "key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_PRO,
#             "role": "Koordinatör"
#         },
#         "SIDAR": {
#             "key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY or ""),
#             "model": GEMINI_MODEL_DEFAULT,
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
    
#     # Geriye dönük uyumluluk için (Eski kodlar bu dict'i direkt çağırıyorsa)
#     AGENT_CONFIGS = AGENT_CONFIGS_GEMINI

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
    
#     # Ses ayarları
#     USE_XTTS: bool = get_bool_env("USE_XTTS", False)
#     TTS_ENGINE: str = os.getenv("TTS_ENGINE", "gtts") # Varsayılan gTTS (Türkçe için)
#     VOICE_SPEED: int = get_int_env("VOICE_SPEED", 150)
#     TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "tr")
    
#     # ───────────────────────────────────────────────────────────
#     # GÜVENLİK AYARLARI
#     # ───────────────────────────────────────────────────────────
#     API_AUTH_ENABLED: bool = get_bool_env("API_AUTH_ENABLED", True)
#     MAX_LOGIN_ATTEMPTS: int = get_int_env("MAX_LOGIN_ATTEMPTS", 3)
#     SESSION_TIMEOUT: int = get_int_env("SESSION_TIMEOUT", 3600)  # saniye
    
#     # ───────────────────────────────────────────────────────────
#     # METOTLAR
#     # ───────────────────────────────────────────────────────────
    
#     @classmethod
#     def initialize_directories(cls) -> bool:
#         """
#         Sistem için gerekli dizinleri oluşturur.
        
#         Returns:
#             Başarılı ise True
#         """
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
#         Belirli bir ajan için ayarları döner (Yeni Metot).
#         """
#         return cls.get_agent_config(agent_name)
    
#     @classmethod
#     def set_provider_mode(cls, mode: str) -> None:
#         """
#         AI sağlayıcı modunu ayarlar.
        
#         Args:
#             mode: 'online', 'gemini', 'local' veya 'ollama'
#         """
#         mode_map = {
#             "online": "gemini",
#             "local": "ollama",
#             "ollama": "ollama",
#             "gemini": "gemini"
#         }
        
#         m_lower = mode.lower()
        
#         if m_lower in mode_map:
#             cls.AI_PROVIDER = mode_map[m_lower]
#             logger.info(f"✅ AI Sağlayıcı modu: {cls.AI_PROVIDER.upper()}")
#         else:
#             logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")
#             logger.info(f"   Geçerli modlar: {', '.join(mode_map.keys())}")
    
#     @classmethod
#     def validate_critical_settings(cls) -> bool:
#         """
#         Kritik sistem ayarlarını doğrular.
        
#         Returns:
#             Tüm kritik ayarlar geçerliyse True
#         """
#         is_valid = True
        
#         # 1. Dizinleri oluştur
#         if not cls.initialize_directories():
#             logger.warning("⚠️ Bazı dizinler oluşturulamadı")
#             is_valid = False
        
#         # 2. API anahtarı kontrolü (Sadece Gemini modu için gerekli)
#         if cls.AI_PROVIDER == "gemini":
#             if not cls._MAIN_KEY:
#                 logger.error(
#                     "❌ KRİTİK HATA: Hiçbir GEMINI API anahtarı bulunamadı!\n"
#                     "   .env dosyasına GEMINI_API_KEY ekleyin veya OLLAMA modunu kullanın."
#                 )
#                 is_valid = False
#             else:
#                 # Anahtar uzunluk kontrolü
#                 if len(cls._MAIN_KEY) < 30:
#                     logger.warning("⚠️ API anahtarı çok kısa görünüyor, geçersiz olabilir")
        
#         # 3. Patron resmi kontrolü (eğer yüz tanıma aktifse)
#         if cls.LIVE_VISUAL_CHECK:
#             if not cls.PATRON_IMAGE_PATH.exists():
#                 logger.warning(
#                     f"⚠️ Patron resmi bulunamadı: {cls.PATRON_IMAGE_PATH}\n"
#                     "   Yüz tanıma devre dışı bırakılabilir."
#                 )
        
#         # 4. Ollama kontrolü (local mod için)
#         if cls.AI_PROVIDER == "ollama":
#             try:
#                 import requests
#                 # Ollama'nın çalıştığını ve modellerin yüklü olduğunu basitçe kontrol et
#                 base_url = cls.OLLAMA_URL.replace("/api", "") # Port kontrolü için base url
#                 response = requests.get(base_url, timeout=2)
#                 if response.status_code != 200:
#                     logger.warning("⚠️ Ollama servisi çalışıyor ancak durum kodu 200 değil.")
#             except:
#                 logger.warning(
#                     "⚠️ Ollama servisi kontrol edilemedi\n"
#                     "   Terminal'de 'ollama serve' komutunu çalıştırdığınızdan emin olun."
#                 )
        
#         return is_valid
    
#     @classmethod
#     def get_system_info(cls) -> Dict[str, Any]:
#         """
#         Sistem bilgilerini dictionary olarak döner.
#         """
#         return {
#             "project": cls.PROJECT_NAME,
#             "version": cls.VERSION,
#             "provider": cls.AI_PROVIDER,
#             "language": cls.LANGUAGE,
#             "gpu_enabled": cls.USE_GPU,
#             "gpu_info": cls.GPU_INFO,
#             "cpu_count": cls.CPU_COUNT,
#             "debug_mode": cls.DEBUG_MODE,
#             "agents": list(cls.AGENT_CONFIGS.keys()),
#             "models": {
#                 "text": cls.TEXT_MODEL,
#                 "vision": cls.VISION_MODEL,
#                 "embedding": cls.EMBEDDING_MODEL
#             }
#         }
    
#     @classmethod
#     def print_config_summary(cls) -> None:
#         """Yapılandırma özetini terminale yazdırır"""
#         print("\n" + "═" * 60)
#         print(f"  {cls.PROJECT_NAME} v{cls.VERSION} - Yapılandırma Özeti")
#         print("═" * 60)
#         print(f"  Dil (Lang)      : {cls.LANGUAGE.upper()}")
#         print(f"  AI Sağlayıcı    : {cls.AI_PROVIDER.upper()}")
#         if cls.AI_PROVIDER == "ollama":
#             print(f"  - Model         : {cls.TEXT_MODEL}")
#             print(f"  - Vision        : {cls.VISION_MODEL}")
#             print(f"  - Embedding     : {cls.EMBEDDING_MODEL}")
#         print(f"  GPU Desteği     : {'✓ ' + cls.GPU_INFO if cls.USE_GPU else '✗ CPU Modu'}")
#         print(f"  CPU Çekirdek    : {cls.CPU_COUNT}")
#         print(f"  Debug Modu      : {'Açık' if cls.DEBUG_MODE else 'Kapalı'}")
#         print("═" * 60 + "\n")


# # ═══════════════════════════════════════════════════════════════
# # BAŞLANGIÇ DOĞRULAMA
# # ═══════════════════════════════════════════════════════════════
# if not Config.validate_critical_settings():
#     if Config.AI_PROVIDER == "gemini":
#         logger.critical(
#             "🚨 Kritik ayar eksik! Sistem düzgün çalışmayabilir.\n"
#             "   Lütfen .env dosyasını kontrol edin."
#         )
# else:
#     logger.info(
#         f"✅ {Config.PROJECT_NAME} v{Config.VERSION} yapılandırması tamamlandı"
#     )
    
#     if Config.DEBUG_MODE:
#         Config.print_config_summary()

# # import os
# # import sys
# # import logging
# # from logging.handlers import RotatingFileHandler
# # from pathlib import Path
# # from dotenv import load_dotenv
# # from typing import Dict, Any, Optional

# # # --- LOGLAMA YAPILANDIRMASI ---
# # BASE_DIR = Path(__file__).resolve().parent
# # LOG_DIR = BASE_DIR / "logs"
# # LOG_DIR.mkdir(parents=True, exist_ok=True)

# # # Loglama formatını daha detaylı hale getirdik
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
# #     handlers=[
# #         logging.StreamHandler(sys.stdout),
# #         RotatingFileHandler(
# #             LOG_DIR / "lotus_system.log", 
# #             maxBytes=10 * 1024 * 1024, # 10MB limit
# #             backupCount=10, 
# #             encoding="utf-8"
# #         )
# #     ]
# # )
# # logger = logging.getLogger("LotusAI.Config")

# # # --- ORTAM DEĞİŞKENLERİ YÜKLEME ---
# # ENV_PATH = BASE_DIR / ".env"
# # if not ENV_PATH.exists():
# #     logger.warning("⚠️ '.env' dosyası bulunamadı! Lütfen API anahtarlarını içeren bir .env dosyası oluşturun.")
# # else:
# #     load_dotenv(dotenv_path=ENV_PATH)

# # # --- YARDIMCI FONKSİYONLAR ---
# # def get_bool_env(key: str, default: bool = False) -> bool:
# #     val = os.getenv(key, str(default)).lower()
# #     return val in ["true", "1", "yes", "on"]

# # def get_int_env(key: str, default: int = 0) -> int:
# #     try:
# #         return int(os.getenv(key, default))
# #     except (ValueError, TypeError):
# #         return default

# # # --- DONANIM HIZLANDIRMA (GPU) KONTROLÜ ---
# # def check_hardware():
# #     has_cuda = False
# #     gpu_name = "N/A"
# #     try:
# #         import torch
# #         if torch.cuda.is_available():
# #             has_cuda = True
# #             gpu_name = torch.cuda.get_device_name(0)
# #             logger.info(f"🚀 Donanım Hızlandırma Aktif: {gpu_name}")
# #         else:
# #             logger.info("ℹ️ GPU bulunamadı, sistem CPU modunda çalışacak.")
# #     except ImportError:
# #         logger.warning("⚠️ PyTorch bulunamadı. AI işlemleri için GPU desteği kontrol edilemedi.")
# #     return has_cuda, gpu_name

# # HAS_CUDA, GPU_NAME = check_hardware()

# # class Config:
# #     """
# #     LotusAI Merkezi Yapılandırma Sınıfı.
# #     Sürüm 2.4 - Profesyonel Donanım ve Ajan Yönetimi
# #     """
# #     # --- GENEL SİSTEM BİLGİLERİ ---
# #     PROJECT_NAME = "LotusAI"
# #     VERSION = "2.4"
# #     DEBUG_MODE = get_bool_env("DEBUG_MODE", True)
# #     WORK_DIR = Path(os.getenv("WORK_DIR", BASE_DIR))

# #     # --- DİZİN YAPILANDIRMASI ---
# #     UPLOAD_DIR = WORK_DIR / "uploads"
# #     TEMPLATE_DIR = WORK_DIR / "templates"
# #     STATIC_DIR = WORK_DIR / "static"
# #     LOG_DIR = WORK_DIR / "logs"
# #     VOICES_DIR = WORK_DIR / "voices"
# #     FACES_DIR = WORK_DIR / "faces"
# #     MODELS_DIR = WORK_DIR / "models" # Yerel modeller için yeni dizin

# #     # Gerekli Dizinleri Otomatik Oluştur
# #     for folder in [UPLOAD_DIR, LOG_DIR, VOICES_DIR, STATIC_DIR, FACES_DIR, MODELS_DIR]:
# #         try:
# #             folder.mkdir(parents=True, exist_ok=True)
# #         except Exception as e:
# #             logger.error(f"❌ Dizin oluşturma hatası ({folder.name}): {e}")

# #     # --- SİSTEM ZAMANLAMALARI ---
# #     CONVERSATION_TIMEOUT = get_int_env("CONVERSATION_TIMEOUT", 60) # Saniye cinsinden
# #     SYSTEM_CHECK_INTERVAL = get_int_env("SYSTEM_CHECK_INTERVAL", 300) # 5 Dakika

# #     # --- AI SAĞLAYICI AYARLARI (MODÜLER) ---
# #     AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
# #     USE_GPU = get_bool_env("USE_GPU", True) and HAS_CUDA
# #     GPU_INFO = GPU_NAME

# #     # --- GEMINI (GOOGLE) AYARLARI ---
# #     GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# #     GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro")
# #     _MAIN_KEY = os.getenv("GEMINI_API_KEY", "")

# #     # Ajanlara özel modeller ve anahtarlar (Dinamik erişim için temel sözlük)
# #     # Yeni ajan eklendiğinde .env üzerinden otomatik tanınır.
# #     AGENT_CONFIGS: Dict[str, Any] = {
# #         "ATLAS": {"key": os.getenv("GEMINI_API_KEY_ATLAS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
# #         "SIDAR": {"key": os.getenv("GEMINI_API_KEY_SIDAR", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
# #         "KURT": {"key": os.getenv("GEMINI_API_KEY_KURT", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
# #         "POYRAZ": {"key": os.getenv("GEMINI_API_KEY_POYRAZ", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT},
# #         "KERBEROS": {"key": os.getenv("GEMINI_API_KEY_KERBEROS", _MAIN_KEY), "model": GEMINI_MODEL_PRO},
# #         "GAYA": {"key": os.getenv("GEMINI_API_KEY_GAYA", _MAIN_KEY), "model": GEMINI_MODEL_DEFAULT}
# #     }

# #     # --- OLLAMA (YEREL AI) AYARLARI ---
# #     TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.1")
# #     VISION_MODEL = os.getenv("VISION_MODEL", "llava")
# #     OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")

# #     # --- MANAGER (YÖNETİCİ) ÖZEL AYARLARI ---
# #     # Camera Manager
# #     CAMERA_INDEX = get_int_env("CAMERA_INDEX", 0)
# #     FACE_REC_MODEL = "cnn" if USE_GPU else "hog"
# #     LIVE_VISUAL_CHECK = get_bool_env("LIVE_VISUAL_CHECK", True)
# #     PATRON_IMAGE_PATH = FACES_DIR / os.getenv("PATRON_IMAGE_PATH", "patron.jpg")

# #     # Finance & Accounting Manager
# #     FINANCE_MODE = get_bool_env("FINANCE_MODE", True)
# #     DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "TRY")

# #     # Messaging & Media
# #     USE_XTTS = get_bool_env("USE_XTTS", False)
# #     META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
# #     WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

# #     @classmethod
# #     def get_agent_settings(cls, agent_name: str) -> Dict[str, str]:
# #         """
# #         Belirtilen ajan için konfigürasyonu döner. 
# #         Eğer listede yoksa varsayılan ayarları oluşturur.
# #         """
# #         name_upper = agent_name.upper()
# #         if name_upper in cls.AGENT_CONFIGS:
# #             return cls.AGENT_CONFIGS[name_upper]
        
# #         # Dinamik olarak .env'den çekmeyi dene
# #         dynamic_key = os.getenv(f"GEMINI_API_KEY_{name_upper}", cls._MAIN_KEY)
# #         return {"key": dynamic_key, "model": cls.GEMINI_MODEL_DEFAULT}

# #     @classmethod
# #     def set_provider_mode(cls, mode: str):
# #         """AI sağlayıcı modunu çalışma anında değiştirir."""
# #         valid_modes = ["gemini", "ollama"]
# #         if mode.lower() in valid_modes:
# #             cls.AI_PROVIDER = mode.lower()
# #             logger.info(f"🔄 AI Sağlayıcı Değiştirildi: {cls.AI_PROVIDER.upper()}")
# #         else:
# #             logger.error(f"❌ Geçersiz sağlayıcı modu: {mode}")

# #     @classmethod
# #     def validate_critical_settings(cls) -> bool:
# #         """Sistemin çalışması için hayati olan ayarları kontrol eder."""
# #         is_valid = True
# #         if cls.AI_PROVIDER == "gemini" and not cls._MAIN_KEY:
# #             logger.error("❌ HATA: Ana GEMINI_API_KEY eksik!")
# #             is_valid = False
        
# #         if cls.LIVE_VISUAL_CHECK and not cls.PATRON_IMAGE_PATH.exists():
# #             logger.warning(f"⚠️ Görsel doğrulama aktif ancak {cls.PATRON_IMAGE_PATH.name} bulunamadı.")
# #             # Bu kritik hata değil ama kullanıcıyı uyarır
            
# #         return is_valid

# # # Başlangıç doğrulaması
# # if not Config.validate_critical_settings():
# #     logger.warning("🚨 Bazı kritik ayarlar eksik. Sistem kısıtlı modda çalışabilir.")
# # else:
# #     logger.info("✅ Tüm kritik sistem ayarları doğrulandı.")