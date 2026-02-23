"""
LotusAI Camera Manager
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Kamera görüntü yönetimi (WSL/Linux + IR Sensör Uyumlu)

Özellikler:
- CUDA destekli görüntü işleme
- Dinamik kamera portu tarama (V4L2 + MJPG Desteği)
- IR / Boş Frame bypass
- Snapshot kaydetme (erişim seviyesine göre kısıtlı)
- Base64 dönüştürme
- Görüntü ön işleme
- Thread-safe operasyonlar
"""

import cv2
import logging
import threading
import base64
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Camera")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class CameraStatus(Enum):
    """Kamera durumları"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class ImageFormat(Enum):
    """Görüntü formatları"""
    NUMPY = "numpy"
    BASE64 = "base64"
    FILE = "file"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class CameraInfo:
    """Kamera bilgisi"""
    index: int
    width: int
    height: int
    fps: float
    backend: str
    cuda_available: bool


@dataclass
class CameraMetrics:
    """Kamera metrikleri"""
    frames_captured: int = 0
    snapshots_saved: int = 0
    errors_encountered: int = 0
    preprocessing_count: int = 0
    cuda_operations: int = 0


# ═══════════════════════════════════════════════════════════════
# CAMERA MANAGER
# ═══════════════════════════════════════════════════════════════
class CameraManager:
    """
    LotusAI Kamera Görüntü Yöneticisi
    
    Yetenekler:
    - CUDA destekli görüntü işleme
    - Dinamik port tarama
    - Snapshot kaydetme (sandbox ve full modda)
    - Base64 dönüştürme
    - Görüntü ön işleme (netleştirme)
    - Thread-safe operasyonlar
    - Erişim seviyesi kontrolleri
    
    OpenCV CUDA kullanarak GPU üzerinde görüntü işleme yapabilir.
    """
    
    # Camera settings
    DEFAULT_RESOLUTION = (640, 480)
    DEFAULT_FPS = 30
    WARMUP_FRAMES = 5  # WSL ve IR kameralar için ısınma süresi artırıldı
    JPEG_QUALITY = 80
    
    # Port scanning (WSL'de kamera video2 vb. olabildiği için genişletildi)
    MAX_PORT_SCAN = 10 
    
    def __init__(self, access_level: Optional[str] = None):
        """
        Camera manager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        # Değişiklik: Eğer parametre girilmezse doğrudan Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL
        
        # Thread safety
        self.lock = threading.RLock()
        
        # WSL/Linux Uyumlu Backend Seçimi
        self.backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY

        # Status
        self.status = CameraStatus.IDLE
        self._active_cap: Optional[cv2.VideoCapture] = None
        
        # CUDA detection
        self.cuda_available = self._detect_cuda()
        if self.cuda_available:
            self._init_cuda_filter()
        
        # Paths
        self.work_dir = Config.WORK_DIR
        self.snapshot_dir = self.work_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings
        self.camera_index = getattr(Config, "CAMERA_INDEX", 0)
        self.resolution = self.DEFAULT_RESOLUTION
        self.flip_horizontal = True
        
        # Metrics
        self.metrics = CameraMetrics()
        
        # Info
        self.camera_info: Optional[CameraInfo] = None
    
    def _detect_cuda(self) -> bool:
        """CUDA tespiti"""
        try:
            if hasattr(cv2, 'cuda'):
                cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
                if cuda_count > 0:
                    logger.info("🚀 OpenCV CUDA aktif (GPU görüntü işleme)")
                    return True
        except Exception:
            pass
        
        logger.info("ℹ️ Kamera CPU modunda")
        return False
    
    def _init_cuda_filter(self) -> None:
        """CUDA filtresi başlat"""
        try:
            self.gpu_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC3,
                cv2.CV_8UC3,
                (0, 0),
                2.0
            )
        except Exception as e:
            logger.error(f"CUDA filter başlatma hatası: {e}")
            self.cuda_available = False
    
    # ───────────────────────────────────────────────────────────
    # INITIALIZATION
    # ───────────────────────────────────────────────────────────
    
    def start(self) -> bool:
        """
        Kamera donanımını başlat
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            # Test default port
            if self._test_hardware(self.camera_index):
                self._update_camera_info()
                logger.info(f"✅ Kamera hazır (Port: {self.camera_index}, Backend: {self.backend})")
                return True
            
            # Scan for active cameras
            logger.warning(
                f"⚠️ Kamera {self.camera_index} erişilemiyor veya boş/IR verisi dönüyor. "
                "Diğer aktif portlar taranıyor..."
            )
            
            active_ports = self.list_cameras()
            
            if active_ports:
                self.camera_index = active_ports[0]
                self._update_camera_info()
                logger.info(f"✅ Gerçek Kamera bulundu (Port: {self.camera_index})")
                return True
            
            logger.error("❌ Erişilebilir kamera bulunamadı! (Lütfen 'sudo chmod 777 /dev/video*' komutunu çalıştırdığınızdan emin olun)")
            self.status = CameraStatus.DISCONNECTED
            return False
    
    def _test_hardware(self, index: int) -> bool:
        """
        Kamera portu testi (IR ve boş çerçeve bypass)
        
        Args:
            index: Kamera port index'i
        
        Returns:
            Çalışıyorsa True
        """
        try:
            cap = cv2.VideoCapture(index, self.backend)
            
            if not cap.isOpened():
                return False
            
            # WSL için MJPG formatını zorla
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Isınma (Warmup): IR portları veya uykudaki kameralar ilk kareleri boş atar
            for _ in range(self.WARMUP_FRAMES):
                cap.grab()
            
            # Görüntü test
            ret, frame = cap.read()
            cap.release()
            
            # Görüntü başarıyla alındıysa ve piksel matrisi içeriyorsa (IR metadata portlarını elemek için)
            return ret and frame is not None and frame.size > 0
        
        except Exception:
            return False
    
    def _update_camera_info(self) -> None:
        """Kamera bilgilerini güncelle"""
        try:
            cap = cv2.VideoCapture(self.camera_index, self.backend)
            
            if cap.isOpened():
                self.camera_info = CameraInfo(
                    index=self.camera_index,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS),
                    backend=cap.getBackendName(),
                    cuda_available=self.cuda_available
                )
                cap.release()
        
        except Exception as e:
            logger.error(f"Kamera bilgi güncelleme hatası: {e}")
    
    def list_cameras(self) -> List[int]:
        """
        Aktif kamera portlarını listele
        
        Returns:
            Port index listesi
        """
        active_ports = []
        
        for i in range(self.MAX_PORT_SCAN):
            if self._test_hardware(i):
                active_ports.append(i)
                
        return active_ports
    
    # ───────────────────────────────────────────────────────────
    # FRAME CAPTURE
    # ───────────────────────────────────────────────────────────
    
    def get_frame(
        self,
        raw: bool = True,
        preprocess: bool = False
    ) -> Optional[Union[np.ndarray, str]]:
        """
        Kameradan frame yakala (Tüm erişim seviyelerine açık)
        
        Args:
            raw: True ise numpy array, False ise base64
            preprocess: Ön işleme uygula
        
        Returns:
            Frame (numpy veya base64) veya None
        """
        if self.status == CameraStatus.BUSY:
            return None
        
        with self.lock:
            self.status = CameraStatus.BUSY
            frame = None
            cap = None
            
            try:
                cap = cv2.VideoCapture(self.camera_index, self.backend)
                
                if not cap.isOpened():
                    logger.error(f"❌ Kamera bağlantısı koptu! ({self.camera_index})")
                    self.status = CameraStatus.ERROR
                    return None
                
                # Settings & MJPG format for WSL
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Warmup (auto exposure)
                for _ in range(self.WARMUP_FRAMES):
                    cap.grab()
                
                # Capture
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    logger.warning("🚫 Boş görüntü veya zaman aşımı")
                    frame = None
                else:
                    # Flip horizontal
                    if self.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    
                    # Preprocess
                    if preprocess:
                        frame = self._preprocess_frame(frame)
                        self.metrics.preprocessing_count += 1
                    
                    self.metrics.frames_captured += 1
            
            except Exception as e:
                logger.error(f"❌ Frame yakalama hatası: {e}")
                self.metrics.errors_encountered += 1
                self.status = CameraStatus.ERROR
            
            finally:
                if cap:
                    cap.release()
                self.status = CameraStatus.IDLE
            
            # Return format
            if frame is not None:
                if not raw:
                    return self._convert_to_base64(frame)
                return frame
            
            return None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Görüntü ön işleme (Unsharp Masking)
        
        Args:
            frame: Ham görüntü
        
        Returns:
            İşlenmiş görüntü
        """
        # GPU acceleration
        if self.cuda_available:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                gpu_blur = self.gpu_filter.apply(gpu_frame)
                
                # Sharpening: Original * 1.5 - Blur * 0.5
                res_gpu = cv2.cuda.addWeighted(
                    gpu_frame, 1.5,
                    gpu_blur, -0.5,
                    0
                )
                
                self.metrics.cuda_operations += 1
                return res_gpu.download()
            
            except Exception as e:
                logger.debug(f"CUDA preprocessing hatası: {e}")
        
        # CPU fallback
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    
    def _convert_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """
        Base64'e dönüştür
        
        Args:
            frame: Numpy array
        
        Returns:
            Base64 string veya None
        """
        try:
            # JPEG encode
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY]
            )
            
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{jpg_as_text}"
        
        except Exception as e:
            logger.error(f"Base64 dönüşüm hatası: {e}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # SNAPSHOT
    # ───────────────────────────────────────────────────────────
    
    def save_snapshot(self, prefix: str = "guvenlik") -> Optional[str]:
        """
        Snapshot kaydet (Sandbox veya full modda)
        
        Args:
            prefix: Dosya adı öneki
        
        Returns:
            Dosya yolu veya None
        """
        # İzin kontrolü
        if self.access_level == AccessLevel.RESTRICTED:
            logger.warning("🚫 Kısıtlı modda snapshot kaydedilemez")
            return None
        
        frame = self.get_frame(raw=True, preprocess=True)
        
        if frame is None:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            save_path = self.snapshot_dir / filename
            
            cv2.imwrite(str(save_path), frame)
            
            self.metrics.snapshots_saved += 1
            logger.info(f"📸 Snapshot kaydedildi: {filename}")
            
            return str(save_path)
        
        except Exception as e:
            logger.error(f"Snapshot kayıt hatası: {e}")
            self.metrics.errors_encountered += 1
            return None
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_info(self) -> Optional[CameraInfo]:
        """
        Kamera bilgilerini getir
        
        Returns:
            CameraInfo veya None
        """
        return self.camera_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Kamera metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "frames_captured": self.metrics.frames_captured,
            "snapshots_saved": self.metrics.snapshots_saved,
            "errors_encountered": self.metrics.errors_encountered,
            "preprocessing_count": self.metrics.preprocessing_count,
            "cuda_operations": self.metrics.cuda_operations,
            "cuda_available": self.cuda_available,
            "status": self.status.value,
            "camera_index": self.camera_index,
            "access_level": self.access_level
        }
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Çözünürlük ayarla (Tüm erişim seviyelerine açık)
        
        Args:
            width: Genişlik
            height: Yükseklik
        
        Returns:
            Başarılı ise True
        """
        try:
            self.resolution = (width, height)
            logger.info(f"📐 Çözünürlük: {width}x{height}")
            return True
        except Exception:
            return False
    
    def stop(self) -> None:
        """
        Kamera servisini durdur (Sadece full modda)
        """
        # İzin kontrolü: Kamera durdurma sistem kaynağına müdahale, full mod gerekli
        if self.access_level != AccessLevel.FULL:
            logger.warning("🚫 Kamera durdurma için full erişim gerekli")
            return
        
        with self.lock:
            if self._active_cap:
                self._active_cap.release()
                self._active_cap = None
            
            self.status = CameraStatus.IDLE
            logger.info("🔌 Kamera servisi durduruldu")



# """
# LotusAI Camera Manager
# Sürüm: 2.5.6 (Eklendi: Erişim Seviyesi Desteği)
# Açıklama: Kamera görüntü yönetimi (WSL/Linux + IR Sensör Uyumlu)

# Özellikler:
# - CUDA destekli görüntü işleme
# - Dinamik kamera portu tarama (V4L2 + MJPG Desteği)
# - IR / Boş Frame bypass
# - Snapshot kaydetme
# - Base64 dönüştürme
# - Görüntü ön işleme
# - Thread-safe operasyonlar
# - Erişim seviyesi kontrolleri
# """

# import cv2
# import logging
# import threading
# import base64
# import sys
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# from typing import Optional, Union, List, Tuple, Dict, Any
# from dataclasses import dataclass
# from enum import Enum

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Camera")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class CameraStatus(Enum):
#     """Kamera durumları"""
#     IDLE = "idle"
#     BUSY = "busy"
#     ERROR = "error"
#     DISCONNECTED = "disconnected"


# class ImageFormat(Enum):
#     """Görüntü formatları"""
#     NUMPY = "numpy"
#     BASE64 = "base64"
#     FILE = "file"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class CameraInfo:
#     """Kamera bilgisi"""
#     index: int
#     width: int
#     height: int
#     fps: float
#     backend: str
#     cuda_available: bool


# @dataclass
# class CameraMetrics:
#     """Kamera metrikleri"""
#     frames_captured: int = 0
#     snapshots_saved: int = 0
#     errors_encountered: int = 0
#     preprocessing_count: int = 0
#     cuda_operations: int = 0


# # ═══════════════════════════════════════════════════════════════
# # CAMERA MANAGER
# # ═══════════════════════════════════════════════════════════════
# class CameraManager:
#     """
#     LotusAI Kamera Görüntü Yöneticisi
    
#     Yetenekler:
#     - CUDA destekli görüntü işleme
#     - Dinamik port tarama
#     - Snapshot kaydetme
#     - Base64 dönüştürme
#     - Görüntü ön işleme (netleştirme)
#     - Thread-safe operasyonlar
#     - Erişim seviyesine göre kısıtlama
    
#     OpenCV CUDA kullanarak GPU üzerinde görüntü işleme yapabilir.
#     """
    
#     # Camera settings
#     DEFAULT_RESOLUTION = (640, 480)
#     DEFAULT_FPS = 30
#     WARMUP_FRAMES = 5  # WSL ve IR kameralar için ısınma süresi artırıldı
#     JPEG_QUALITY = 80
    
#     # Port scanning (WSL'de kamera video2 vb. olabildiği için genişletildi)
#     MAX_PORT_SCAN = 10 
    
#     def __init__(self, access_level: str = "sandbox"):
#         """
#         Camera manager başlatıcı
        
#         Args:
#             access_level: Erişim seviyesi (restricted, sandbox, full)
#         """
#         self.access_level = access_level
        
#         # Thread safety
#         self.lock = threading.RLock()
        
#         # WSL/Linux Uyumlu Backend Seçimi
#         self.backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY

#         # Status
#         self.status = CameraStatus.IDLE
#         self._active_cap: Optional[cv2.VideoCapture] = None
        
#         # CUDA detection
#         self.cuda_available = self._detect_cuda()
#         if self.cuda_available:
#             self._init_cuda_filter()
        
#         # Paths
#         self.work_dir = Config.WORK_DIR
#         self.snapshot_dir = self.work_dir / "snapshots"
#         self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
#         # Settings
#         self.camera_index = getattr(Config, "CAMERA_INDEX", 0)
#         self.resolution = self.DEFAULT_RESOLUTION
#         self.flip_horizontal = True
        
#         # Metrics
#         self.metrics = CameraMetrics()
        
#         # Info
#         self.camera_info: Optional[CameraInfo] = None
        
#         # Erişim seviyesi logla
#         logger.info(f"📷 Camera Manager başlatıldı (Erişim: {self.access_level})")
    
#     def _detect_cuda(self) -> bool:
#         """CUDA tespiti"""
#         try:
#             if hasattr(cv2, 'cuda'):
#                 cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
#                 if cuda_count > 0:
#                     logger.info("🚀 OpenCV CUDA aktif (GPU görüntü işleme)")
#                     return True
#         except Exception:
#             pass
        
#         logger.info("ℹ️ Kamera CPU modunda")
#         return False
    
#     def _init_cuda_filter(self) -> None:
#         """CUDA filtresi başlat"""
#         try:
#             self.gpu_filter = cv2.cuda.createGaussianFilter(
#                 cv2.CV_8UC3,
#                 cv2.CV_8UC3,
#                 (0, 0),
#                 2.0
#             )
#         except Exception as e:
#             logger.error(f"CUDA filter başlatma hatası: {e}")
#             self.cuda_available = False
    
#     # ───────────────────────────────────────────────────────────
#     # INITIALIZATION
#     # ───────────────────────────────────────────────────────────
    
#     def start(self) -> bool:
#         """
#         Kamera donanımını başlat
        
#         Returns:
#             Başarılı ise True
#         """
#         with self.lock:
#             # Test default port
#             if self._test_hardware(self.camera_index):
#                 self._update_camera_info()
#                 logger.info(f"✅ Kamera hazır (Port: {self.camera_index}, Backend: {self.backend})")
#                 return True
            
#             # Scan for active cameras
#             logger.warning(
#                 f"⚠️ Kamera {self.camera_index} erişilemiyor veya boş/IR verisi dönüyor. "
#                 "Diğer aktif portlar taranıyor..."
#             )
            
#             active_ports = self.list_cameras()
            
#             if active_ports:
#                 self.camera_index = active_ports[0]
#                 self._update_camera_info()
#                 logger.info(f"✅ Gerçek Kamera bulundu (Port: {self.camera_index})")
#                 return True
            
#             logger.error("❌ Erişilebilir kamera bulunamadı! (Lütfen 'sudo chmod 777 /dev/video*' komutunu çalıştırdığınızdan emin olun)")
#             self.status = CameraStatus.DISCONNECTED
#             return False
    
#     def _test_hardware(self, index: int) -> bool:
#         """
#         Kamera portu testi (IR ve boş çerçeve bypass)
        
#         Args:
#             index: Kamera port index'i
        
#         Returns:
#             Çalışıyorsa True
#         """
#         try:
#             cap = cv2.VideoCapture(index, self.backend)
            
#             if not cap.isOpened():
#                 return False
            
#             # WSL için MJPG formatını zorla
#             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#             # Isınma (Warmup): IR portları veya uykudaki kameralar ilk kareleri boş atar
#             for _ in range(self.WARMUP_FRAMES):
#                 cap.grab()
            
#             # Görüntü test
#             ret, frame = cap.read()
#             cap.release()
            
#             # Görüntü başarıyla alındıysa ve piksel matrisi içeriyorsa (IR metadata portlarını elemek için)
#             return ret and frame is not None and frame.size > 0
        
#         except Exception:
#             return False
    
#     def _update_camera_info(self) -> None:
#         """Kamera bilgilerini güncelle"""
#         try:
#             cap = cv2.VideoCapture(self.camera_index, self.backend)
            
#             if cap.isOpened():
#                 self.camera_info = CameraInfo(
#                     index=self.camera_index,
#                     width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                     fps=cap.get(cv2.CAP_PROP_FPS),
#                     backend=cap.getBackendName(),
#                     cuda_available=self.cuda_available
#                 )
#                 cap.release()
        
#         except Exception as e:
#             logger.error(f"Kamera bilgi güncelleme hatası: {e}")
    
#     def list_cameras(self) -> List[int]:
#         """
#         Aktif kamera portlarını listele (Tüm kullanıcılara açık)
        
#         Returns:
#             Port index listesi
#         """
#         active_ports = []
        
#         for i in range(self.MAX_PORT_SCAN):
#             if self._test_hardware(i):
#                 active_ports.append(i)
                
#         return active_ports
    
#     # ───────────────────────────────────────────────────────────
#     # FRAME CAPTURE (Erişim kontrollü)
#     # ───────────────────────────────────────────────────────────
    
#     def get_frame(
#         self,
#         raw: bool = True,
#         preprocess: bool = False
#     ) -> Optional[Union[np.ndarray, str]]:
#         """
#         Kameradan frame yakala (Kısıtlı modda çalışmaz)
        
#         Args:
#             raw: True ise numpy array, False ise base64
#             preprocess: Ön işleme uygula
        
#         Returns:
#             Frame (numpy veya base64) veya None
#         """
#         # Erişim kontrolü: Kısıtlı modda kamera kullanımı yasak
#         if self.access_level == AccessLevel.RESTRICTED:
#             logger.warning("🚫 Kısıtlı modda kamera kullanımı engellendi")
#             return None
        
#         if self.status == CameraStatus.BUSY:
#             return None
        
#         with self.lock:
#             self.status = CameraStatus.BUSY
#             frame = None
#             cap = None
            
#             try:
#                 cap = cv2.VideoCapture(self.camera_index, self.backend)
                
#                 if not cap.isOpened():
#                     logger.error(f"❌ Kamera bağlantısı koptu! ({self.camera_index})")
#                     self.status = CameraStatus.ERROR
#                     return None
                
#                 # Settings & MJPG format for WSL
#                 cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
#                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
#                 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
#                 # Warmup (auto exposure)
#                 for _ in range(self.WARMUP_FRAMES):
#                     cap.grab()
                
#                 # Capture
#                 ret, frame = cap.read()
                
#                 if not ret or frame is None or frame.size == 0:
#                     logger.warning("🚫 Boş görüntü veya zaman aşımı")
#                     frame = None
#                 else:
#                     # Flip horizontal
#                     if self.flip_horizontal:
#                         frame = cv2.flip(frame, 1)
                    
#                     # Preprocess
#                     if preprocess:
#                         frame = self._preprocess_frame(frame)
#                         self.metrics.preprocessing_count += 1
                    
#                     self.metrics.frames_captured += 1
            
#             except Exception as e:
#                 logger.error(f"❌ Frame yakalama hatası: {e}")
#                 self.metrics.errors_encountered += 1
#                 self.status = CameraStatus.ERROR
            
#             finally:
#                 if cap:
#                     cap.release()
#                 self.status = CameraStatus.IDLE
            
#             # Return format
#             if frame is not None:
#                 if not raw:
#                     return self._convert_to_base64(frame)
#                 return frame
            
#             return None
    
#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Görüntü ön işleme (Unsharp Masking)
        
#         Args:
#             frame: Ham görüntü
        
#         Returns:
#             İşlenmiş görüntü
#         """
#         # GPU acceleration
#         if self.cuda_available:
#             try:
#                 gpu_frame = cv2.cuda_GpuMat()
#                 gpu_frame.upload(frame)
                
#                 gpu_blur = self.gpu_filter.apply(gpu_frame)
                
#                 # Sharpening: Original * 1.5 - Blur * 0.5
#                 res_gpu = cv2.cuda.addWeighted(
#                     gpu_frame, 1.5,
#                     gpu_blur, -0.5,
#                     0
#                 )
                
#                 self.metrics.cuda_operations += 1
#                 return res_gpu.download()
            
#             except Exception as e:
#                 logger.debug(f"CUDA preprocessing hatası: {e}")
        
#         # CPU fallback
#         gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
#         return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    
#     def _convert_to_base64(self, frame: np.ndarray) -> Optional[str]:
#         """
#         Base64'e dönüştür
        
#         Args:
#             frame: Numpy array
        
#         Returns:
#             Base64 string veya None
#         """
#         try:
#             # JPEG encode
#             _, buffer = cv2.imencode(
#                 '.jpg',
#                 frame,
#                 [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY]
#             )
            
#             jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#             return f"data:image/jpeg;base64,{jpg_as_text}"
        
#         except Exception as e:
#             logger.error(f"Base64 dönüşüm hatası: {e}")
#             return None
    
#     # ───────────────────────────────────────────────────────────
#     # SNAPSHOT (Erişim kontrollü)
#     # ───────────────────────────────────────────────────────────
    
#     def save_snapshot(self, prefix: str = "guvenlik") -> Optional[str]:
#         """
#         Snapshot kaydet (Kısıtlı modda çalışmaz)
        
#         Args:
#             prefix: Dosya adı öneki
        
#         Returns:
#             Dosya yolu veya None
#         """
#         # Erişim kontrolü: Kısıtlı modda snapshot kaydedilemez
#         if self.access_level == AccessLevel.RESTRICTED:
#             logger.warning("🚫 Kısıtlı modda snapshot kaydetme engellendi")
#             return None
        
#         frame = self.get_frame(raw=True, preprocess=True)
        
#         if frame is None:
#             return None
        
#         try:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"{prefix}_{timestamp}.jpg"
#             save_path = self.snapshot_dir / filename
            
#             cv2.imwrite(str(save_path), frame)
            
#             self.metrics.snapshots_saved += 1
#             logger.info(f"📸 Snapshot kaydedildi: {filename}")
            
#             return str(save_path)
        
#         except Exception as e:
#             logger.error(f"Snapshot kayıt hatası: {e}")
#             self.metrics.errors_encountered += 1
#             return None
    
#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────
    
#     def get_info(self) -> Optional[CameraInfo]:
#         """
#         Kamera bilgilerini getir (Tüm kullanıcılara açık)
        
#         Returns:
#             CameraInfo veya None
#         """
#         return self.camera_info
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Kamera metrikleri
        
#         Returns:
#             Metrik dictionary
#         """
#         return {
#             "frames_captured": self.metrics.frames_captured,
#             "snapshots_saved": self.metrics.snapshots_saved,
#             "errors_encountered": self.metrics.errors_encountered,
#             "preprocessing_count": self.metrics.preprocessing_count,
#             "cuda_operations": self.metrics.cuda_operations,
#             "cuda_available": self.cuda_available,
#             "status": self.status.value,
#             "camera_index": self.camera_index,
#             "access_level": self.access_level
#         }
    
#     def set_resolution(self, width: int, height: int) -> bool:
#         """
#         Çözünürlük ayarla (Sadece sandbox ve full modda)
        
#         Args:
#             width: Genişlik
#             height: Yükseklik
        
#         Returns:
#             Başarılı ise True
#         """
#         if self.access_level == AccessLevel.RESTRICTED:
#             logger.warning("🚫 Kısıtlı modda çözünürlük değiştirilemez")
#             return False
        
#         try:
#             self.resolution = (width, height)
#             logger.info(f"📐 Çözünürlük: {width}x{height}")
#             return True
#         except Exception:
#             return False
    
#     def stop(self) -> None:
#         """Kamera servisini durdur"""
#         with self.lock:
#             if self._active_cap:
#                 self._active_cap.release()
#                 self._active_cap = None
            
#             self.status = CameraStatus.IDLE
#             logger.info("🔌 Kamera servisi durduruldu")




