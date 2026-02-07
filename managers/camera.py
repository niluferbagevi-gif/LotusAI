import cv2
import logging
import threading
import time
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List

# Proje içi modüller
try:
    from config import Config
except ImportError:
    class Config:
        WORK_DIR = Path.cwd()
        CAMERA_INDEX = 0
        DEBUG_MODE = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Camera")

# OpenCV ve Donanım Kontrolü
CV2_AVAILABLE = True
CUDA_AVAILABLE = False

try:
    _test_cv2 = cv2.__version__
    # CUDA desteği kontrolü
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        logger.info(f"🚀 GPU/CUDA Desteği Aktif: {cv2.cuda.getDevice() if hasattr(cv2.cuda, 'getDevice') else 'Tespit Edildi'}")
    else:
        logger.warning("⚠️ CUDA uyumlu GPU bulunamadı. CPU modunda devam ediliyor.")
except Exception as e:
    CV2_AVAILABLE = False
    logger.error(f"❌ OpenCV hatası: {e}")

class CameraManager:
    """
    LotusAI Kamera Görüntü Yöneticisi (GPU Optimize Edilmiş).
    
    Yetenekler:
    - GPU Hızlandırma: Görüntü işleme filtreleri CUDA çekirdeklerinde çalışır.
    - Akıllı Yakalama: Işık dengesini koruyan ısınma döngülü kare yakalama.
    - Çoklu Format: İşleme için RAW (Numpy), Web UI için Base64 çıktı.
    - Kaynak Yönetimi: Donanım kilitlenmelerini önleyen güvenli (RLock) erişim.
    """
    
    def __init__(self):
        # Reentrant Lock: Aynı ipliğin kendi kilidini tekrar alabilmesini sağlar.
        self.lock = threading.RLock()
        
        # Durum Değişkenleri
        self.last_seen_person = None 
        self.is_busy = False
        self._active_cap = None
        
        # GPU Nesneleri (Sadece CUDA varsa oluşturulur)
        self.gpu_filter = None
        if CUDA_AVAILABLE:
            # Keskinleştirme için önceden tanımlanmış GPU filtresi (Hız için)
            # Parametreler: (src_type, ksize, sigmaX)
            self.gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (0, 0), 2.0)
        
        # Dizin Yapılandırması
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.snapshot_dir = self.work_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Ayarlar
        self.camera_index = getattr(Config, "CAMERA_INDEX", 0)
        self.resolution = (640, 480) 
        self.flip_horizontal = True  

    def start(self):
        """Kamera servisinin donanım hazırlığını kontrol eder."""
        if not CV2_AVAILABLE:
            return
            
        with self.lock:
            if self._test_hardware():
                logger.info(f"✅ Kamera servisi hazır. (GPU: {'Aktif' if CUDA_AVAILABLE else 'Pasif'}, Port: {self.camera_index})")
            else:
                logger.warning(f"⚠️ Kamera (ID:{self.camera_index}) algılandı ancak erişim kısıtlı olabilir.")

    def _test_hardware(self) -> bool:
        """Kamera donanımının erişilebilir olup olmadığını test eder."""
        cap = cv2.VideoCapture(self.camera_index)
        available = cap.isOpened()
        if available:
            cap.release()
        return available

    def get_frame(self, raw: bool = True, preprocess: bool = False) -> Optional[Union[np.ndarray, str]]:
        """
        Kameradan anlık bir kare yakalar.
        
        Args:
            raw: True ise OpenCV matrisi, False ise Base64 string döner.
            preprocess: Görüntü üzerinde GPU/CPU iyileştirmeleri yapar.
        """
        if not CV2_AVAILABLE:
            return None
        
        with self.lock:
            self.is_busy = True
            frame = None
            cap = None
            
            try:
                # Backend seçimi
                backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') and self.work_dir.drive else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.camera_index, backend)
                
                if not cap.isOpened():
                    logger.error(f"❌ Kamera donanımına erişilemedi! İndeks: {self.camera_index}")
                    return None

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Isınma Döngüsü
                for _ in range(5):
                    cap.grab()

                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.error("🚫 Kameradan veri okunamadı.")
                    frame = None
                else:
                    # Görüntü Çevirme
                    if self.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    
                    # Ön İşleme (GPU Desteği ile)
                    if preprocess:
                        frame = self._preprocess_frame(frame)

            except Exception as e:
                logger.error(f"❌ Kamera yakalama hatası: {e}")
            
            finally:
                if cap:
                    cap.release()
                self.is_busy = False

            # Çıktı Formatı
            if frame is not None:
                if not raw:
                    return self._convert_to_base64(frame)
                return frame
            
            return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Görüntü kalitesini artırmak için filtre uygular (CUDA varsa GPU kullanır)."""
        if CUDA_AVAILABLE:
            try:
                # 1. Görüntüyü GPU'ya Yükle
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)

                # 2. GPU üzerinde Gaussian Blur uygula
                gpu_blur = self.gpu_filter.apply(gpu_frame)

                # 3. Keskinleştirme (Unsharp Masking: frame * 1.5 - blur * 0.5)
                # Not: addWeighted doğrudan GPU'da her sürümde stabil olmayabilir, 
                # bu yüzden işlem bittikten sonra CPU'da veya CUDA aritmetiği ile yapılır.
                res_gpu = cv2.cuda.addWeighted(gpu_frame, 1.5, gpu_blur, -0.5, 0)

                # 4. Görüntüyü CPU'ya geri çek
                return res_gpu.download()
            except Exception as e:
                logger.warning(f"⚠️ GPU ön işleme hatası (CPU'ya dönülüyor): {e}")
                # Hata durumunda CPU fallback
                pass

        # CPU Modu (Fallback veya CUDA yoksa)
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)

    def _convert_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """OpenCV karesini Web UI için Base64 formatına dönüştürür."""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{jpg_as_text}"
        except Exception as e:
            logger.error(f"Base64 dönüşüm hatası: {e}")
            return None

    def save_snapshot(self, prefix: str = "security") -> Optional[str]:
        """Kritik anlarda o anki kareyi diske kaydeder."""
        frame = self.get_frame(raw=True)
        if frame is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}.jpg"
                save_path = self.snapshot_dir / filename
                
                cv2.imwrite(str(save_path), frame)
                logger.info(f"📸 Snapshot kaydedildi: {filename}")
                return str(save_path)
            except Exception as e:
                logger.error(f"Snapshot kayıt hatası: {e}")
        return None

    def list_cameras(self) -> List[int]:
        """Sistemdeki aktif tüm kamera portlarını tarar."""
        active_ports = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    active_ports.append(i)
                cap.release()
        return active_ports
    
    def stop(self):
        """Servis kapanırken donanım temizliği yapar."""
        with self.lock:
            if self._active_cap:
                self._active_cap.release()
            logger.info("🔌 Kamera servisi kapatıldı.")