# core/utils.py
import os
import sys
import contextlib
import logging
import transformers.pytorch_utils

def setup_logging():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    
    # OpenCV Loglarını Kapat
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def patch_transformers():
    """MPS hatasını önlemek için monkey patch."""
    try:
        if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
            def _isin_mps_friendly():
                return False
            transformers.pytorch_utils.isin_mps_friendly = _isin_mps_friendly
    except ImportError:
        pass

@contextlib.contextmanager
def ignore_stderr():
    """ALSA, JACK ve OpenCV hatalarını susturur."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)
    except Exception:
        yield
    finally:
        os.close(devnull)