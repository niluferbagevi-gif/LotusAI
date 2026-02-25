"""
LotusAI core/utils.py - Utility Functions
Sürüm: 2.6.0 (Merkezi Config Log Senkronu & ALSA Log Silencer)
Açıklama: Yardımcı fonksiyonlar, 3. parti log bastırma ve sistem yamaları (patches)

Özellikler:
- Third-party logging suppression
- Library patches (Transformers, Pygame)
- stderr/stdout suppression
- Environment setup
- Decorators (Retry, Silent)
"""

import os
import sys
import contextlib
import logging
import functools
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Generator
from io import StringIO

# ═══════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════
logger = logging.getLogger("LotusAI.Utils")


# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════
def setup_logging(suppress_third_party: bool = True) -> None:
    """
    Ek Logging yapılandırması (Ana yapılandırma config.py'de yapılmaktadır)
    Buradaki temel amaç 3. parti kütüphanelerin gereksiz loglarını susturmaktır.
    
    Args:
        suppress_third_party: 3. parti kütüphane loglarını bastır
    """
    # 3. parti kütüphane loglarını bastır
    if suppress_third_party:
        suppress_libraries = [
            "httpx",
            "httpcore",
            "urllib3",
            "huggingface_hub",
            "transformers",
            "requests",
            "asyncio",
            "werkzeug"  # Flask'in gereksiz trafik loglarını susturur
        ]
        
        for lib in suppress_libraries:
            logging.getLogger(lib).setLevel(logging.WARNING)
        
        # Daha kritik seviyede bastırılacaklar
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    
    # OpenCV ve ALSA loglarını kapat
    _suppress_opencv_logs()
    _suppress_alsa_logs()
    
    logger.debug("✅ 3. parti kütüphane log susturucuları aktif edildi")


def _suppress_opencv_logs() -> None:
    """OpenCV loglarını bastır"""
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    
    # Ek OpenCV environment variables
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

def _suppress_alsa_logs() -> None:
    """ALSA (Linux Ses Çekirdeği) C-seviyesi loglarını kökünden susturur"""
    if sys.platform != "linux":
        return  # Windows/Mac'te ALSA olmaz, atla.
        
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
        
        # C hata yakalayıcı fonksiyon şablonu
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        
        # Boş (hiçbir şey yapmayan) Python fonksiyonu
        def py_error_handler(filename, line, function, err, fmt):
            pass
            
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        # ALSA kütüphanesini bul ve hata yöneticimizi ata
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
        
        # Garbage collection'ı önlemek için fonksiyona bağla
        _suppress_alsa_logs.c_error_handler = c_error_handler
    except Exception as e:
        logger.debug(f"ALSA log susturucu başlatılamadı (normaldir): {e}")


# ═══════════════════════════════════════════════════════════════
# LIBRARY PATCHES
# ═══════════════════════════════════════════════════════════════
def patch_transformers() -> bool:
    """
    Transformers kütüphanesine MPS hatası için monkey patch uygula
    
    Returns:
        Başarılı ise True
    """
    try:
        import transformers.pytorch_utils
        
        # isin_mps_friendly fonksiyonu yoksa ekle
        if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
            def _isin_mps_friendly(*args, **kwargs) -> bool:
                """MPS uyumluluğu kontrolü (her zaman False döner)"""
                return False
            
            transformers.pytorch_utils.isin_mps_friendly = _isin_mps_friendly
            logger.debug("✅ Transformers MPS patch uygulandı")
            return True
        
        return True
    
    except ImportError:
        logger.debug("ℹ️ Transformers yüklü değil, patch atlanıyor")
        return False
    
    except Exception as e:
        logger.warning(f"⚠️ Transformers patch hatası: {e}")
        return False


def patch_pygame() -> bool:
    """
    Pygame'e patch uygula (gerekirse)
    
    Returns:
        Başarılı ise True
    """
    try:
        import pygame
        
        # Pygame mixer için environment variables
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        
        logger.debug("✅ Pygame patch uygulandı")
        return True
    
    except ImportError:
        logger.debug("ℹ️ Pygame yüklü değil, patch atlanıyor")
        return False
    
    except Exception as e:
        logger.warning(f"⚠️ Pygame patch hatası: {e}")
        return False


def apply_all_patches() -> Dict[str, bool]:
    """
    Tüm library patch'lerini uygula
    
    Returns:
        Patch sonuçları dict
    """
    results = {
        "transformers": patch_transformers(),
        "pygame": patch_pygame()
    }
    
    successful = sum(results.values())
    total = len(results)
    
    logger.info(f"📦 Library patches: {successful}/{total} başarılı")
    
    return results


# ═══════════════════════════════════════════════════════════════
# STDERR/STDOUT SUPPRESSION
# ═══════════════════════════════════════════════════════════════
@contextlib.contextmanager
def ignore_stderr() -> Generator[None, None, None]:
    """
    Stderr çıktısını geçici olarak bastır
    
    ALSA, JACK ve OpenCV hatalarını susturmak için kullanılır.
    
    Usage:
        with ignore_stderr():
            # Noisy code here
            pass
    
    Yields:
        None
    """
    devnull = None
    old_stderr = None
    
    try:
        # /dev/null aç
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Mevcut stderr'i sakla
        old_stderr = os.dup(sys.stderr.fileno())
        
        # stderr'i /dev/null'a yönlendir
        os.dup2(devnull, sys.stderr.fileno())
        
        yield
    
    except Exception as e:
        logger.debug(f"stderr suppression hatası: {e}")
        yield
    
    finally:
        # stderr'i geri yükle
        if old_stderr is not None:
            try:
                os.dup2(old_stderr, sys.stderr.fileno())
                os.close(old_stderr)
            except Exception:
                pass
        
        # devnull'u kapat
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:
                pass


@contextlib.contextmanager
def ignore_stdout() -> Generator[None, None, None]:
    """
    Stdout çıktısını geçici olarak bastır
    
    Usage:
        with ignore_stdout():
            print("This won't be printed")
    
    Yields:
        None
    """
    devnull = None
    old_stdout = None
    
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(sys.stdout.fileno())
        os.dup2(devnull, sys.stdout.fileno())
        
        yield
    
    except Exception as e:
        logger.debug(f"stdout suppression hatası: {e}")
        yield
    
    finally:
        if old_stdout is not None:
            try:
                os.dup2(old_stdout, sys.stdout.fileno())
                os.close(old_stdout)
            except Exception:
                pass
        
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:
                pass


@contextlib.contextmanager
def suppress_output() -> Generator[None, None, None]:
    """
    Hem stdout hem stderr'i bastır
    
    Usage:
        with suppress_output():
            # Completely silent code
            pass
    
    Yields:
        None
    """
    with ignore_stdout():
        with ignore_stderr():
            yield


@contextlib.contextmanager
def capture_output() -> Generator[StringIO, None, None]:
    """
    Stdout çıktısını yakala
    
    Usage:
        with capture_output() as output:
            print("Hello")
        
        print(output.getvalue())  # "Hello\n"
    
    Yields:
        StringIO objesi
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout


# ═══════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════
def suppress_errors(
    default_return: Any = None,
    log_errors: bool = True
) -> Callable:
    """
    Hataları bastıran decorator
    
    Args:
        default_return: Hata durumunda döndürülecek değer
        log_errors: Hataları logla
    
    Usage:
        @suppress_errors(default_return=False)
        def risky_function():
            return 1 / 0
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} hatası: {e}")
                return default_return
        return wrapper
    return decorator


def silent_execution(func: Callable) -> Callable:
    """
    Fonksiyonu sessizce çalıştıran decorator
    
    Usage:
        @silent_execution
        def noisy_function():
            print("This won't be printed")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_output():
            return func(*args, **kwargs)
    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Hata durumunda yeniden deneyen decorator
    
    Args:
        max_attempts: Maksimum deneme sayısı
        delay: Denemeler arası bekleme (saniye)
        exceptions: Yakalanacak exception türleri
    
    Usage:
        @retry(max_attempts=3, delay=2.0)
        def unstable_function():
            # Unstable code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    logger.warning(
                        f"{func.__name__} deneme {attempt + 1}/{max_attempts} "
                        f"başarısız: {e}"
                    )
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# PLATFORM UTILITIES
# ═══════════════════════════════════════════════════════════════
def get_platform_info() -> Dict[str, str]:
    """
    Platform bilgilerini döndür
    
    Returns:
        Platform bilgileri dict
    """
    import platform
    
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }


def is_windows() -> bool:
    """Windows platformu mu"""
    return sys.platform.startswith('win')


def is_linux() -> bool:
    """Linux platformu mu"""
    return sys.platform.startswith('linux')


def is_macos() -> bool:
    """macOS platformu mu"""
    return sys.platform == 'darwin'


# ═══════════════════════════════════════════════════════════════
# PATH UTILITIES
# ═══════════════════════════════════════════════════════════════
def ensure_dir(path: Path) -> Path:
    """
    Dizinin var olduğundan emin ol, yoksa oluştur
    
    Args:
        path: Dizin yolu
    
    Returns:
        Path objesi
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(path: Path) -> float:
    """
    Dosya boyutunu MB cinsinden döndür
    
    Args:
        path: Dosya yolu
    
    Returns:
        Boyut (MB)
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def clean_filename(filename: str, replacement: str = "_") -> str:
    """
    Dosya adını temizle (geçersiz karakterleri kaldır)
    
    Args:
        filename: Orijinal dosya adı
        replacement: Geçersiz karakterler için yedek
    
    Returns:
        Temiz dosya adı
    """
    import re
    
    # Geçersiz karakterler
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    
    # Temizle
    clean = re.sub(invalid_chars, replacement, filename)
    
    # Boşlukları da değiştir (opsiyonel)
    clean = clean.replace(' ', replacement)
    
    # Birden fazla replacement'ı tek karaktere düşür
    clean = re.sub(f'{re.escape(replacement)}+', replacement, clean)
    
    # Başta/sonda replacement varsa kaldır
    clean = clean.strip(replacement)
    
    return clean


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════
def setup_environment() -> None:
    """
    Tüm environment ayarlarını yap
    
    - Logging setup
    - Library patches
    - Environment variables
    """
    # Logging
    setup_logging()
    
    # Patches
    apply_all_patches()
    
    # Environment variables
    _setup_env_variables()
    
    logger.info("✅ Environment setup tamamlandı")


def _setup_env_variables() -> None:
    """Gerekli environment variable'ları ayarla"""
    env_vars = {
        # TensorFlow
        "TF_CPP_MIN_LOG_LEVEL": "2",
        
        # CUDA
        "CUDA_LAUNCH_BLOCKING": "0",
        
        # OpenMP
        "OMP_NUM_THREADS": "1",
        
        # Tokenizers
        "TOKENIZERS_PARALLELISM": "false",
        
        # Misc
        "PYTHONIOENCODING": "utf-8"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


# ═══════════════════════════════════════════════════════════════
# VALIDATION UTILITIES
# ═══════════════════════════════════════════════════════════════
def validate_file_exists(path: Path, raise_error: bool = True) -> bool:
    """
    Dosyanın var olduğunu kontrol et
    
    Args:
        path: Dosya yolu
        raise_error: Hata durumunda exception fırlat
    
    Returns:
        Dosya varsa True
    
    Raises:
        FileNotFoundError: raise_error=True ve dosya yoksa
    """
    path = Path(path)
    
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        return False
    
    return True


def validate_directory_writable(path: Path, raise_error: bool = True) -> bool:
    """
    Dizine yazma yetkisi olduğunu kontrol et
    
    Args:
        path: Dizin yolu
        raise_error: Hata durumunda exception fırlat
    
    Returns:
        Yazılabilirse True
    
    Raises:
        PermissionError: raise_error=True ve yazılamıyorsa
    """
    path = Path(path)
    
    if not os.access(path, os.W_OK):
        if raise_error:
            raise PermissionError(f"Dizine yazma yetkisi yok: {path}")
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════
# STRING UTILITIES
# ═══════════════════════════════════════════════════════════════
def truncate_string(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """
    Metni kısalt
    
    Args:
        text: Orijinal metin
        max_length: Maksimum uzunluk
        suffix: Kısaltma soneki
    
    Returns:
        Kısaltılmış metin
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_string(text: str) -> str:
    """
    Metni temizle (güvenli hale getir)
    
    Args:
        text: Ham metin
    
    Returns:
        Temizlenmiş metin
    """
    import html
    
    # HTML escape
    clean = html.escape(text)
    
    # Whitespace normalize
    clean = ' '.join(clean.split())
    
    return clean


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════
# Otomatik environment setup (import sırasında)
try:
    _suppress_opencv_logs()
    _suppress_alsa_logs()
except Exception as e:
    logger.debug(f"Log suppression hatası: {e}")