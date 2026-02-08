import cv2
import logging
import threading
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

class CameraManager:
    """
    LotusAI Kamera Görüntü Yöneticisi.
    
    Not: OpenCV'nin standart pip sürümü CUDA desteklemez. 
    Kamera yakalama (I/O) işlemleri CPU tabanlıdır ve bu en kararlı yöntemdir.
    GPU, sadece çok ağır görüntü işleme algoritmalarında (DNN vb.) gereklidir.
    """
    
    def __init__(self):
        # Thread Safety
        self.lock = threading.RLock()
        
        # Durum Değişkenleri
        self.is_busy = False
        self._active_cap = None
        
        # OpenCV CUDA Kontrolü (Sadece bilgilendirme amaçlı)
        self.cuda_available = False
        try:
            # OpenCV'nin CUDA modülü var mı ve cihaz sayısı > 0 mı?
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.cuda_available = True
                self.gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (0, 0), 2.0)
                logger.info("🚀 OpenCV CUDA Desteği Aktif (Görüntü işleme GPU'da yapılacak)")
            else:
                # Bu bir hata değildir, standart davranıştır.
                logger.info("ℹ️ Kamera servisi CPU modunda başlatılıyor (Standart OpenCV).")
        except Exception:
            logger.info("ℹ️ Kamera servisi CPU modunda başlatılıyor.")

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
        with self.lock:
            if self._test_hardware():
                logger.info(f"✅ Kamera servisi hazır. (Port: {self.camera_index})")
            else:
                logger.warning(f"⚠️ Kamera (ID:{self.camera_index}) algılandı ancak erişim sağlanamıyor.")

    def _test_hardware(self) -> bool:
        """Kamera donanımının erişilebilir olup olmadığını test eder."""
        try:
            backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') and self.work_dir.drive else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            available = cap.isOpened()
            if available:
                cap.release()
            return available
        except Exception as e:
            logger.error(f"Donanım testi hatası: {e}")
            return False

    def get_frame(self, raw: bool = True, preprocess: bool = False) -> Optional[Union[np.ndarray, str]]:
        """
        Kameradan anlık bir kare yakalar.
        """
        with self.lock:
            self.is_busy = True
            frame = None
            cap = None
            
            try:
                # Backend seçimi (Windows için DSHOW tercih edilir)
                backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') and self.work_dir.drive else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.camera_index, backend)
                
                if not cap.isOpened():
                    logger.error(f"❌ Kamera donanımına erişilemedi! İndeks: {self.camera_index}")
                    return None

                # Ayarları uygula
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Gecikmeyi önlemek için buffer 1

                # Isınma Döngüsü (Karanlık görüntüyü önler)
                # 2 kare yeterlidir, 5 kare çok zaman kaybettirir.
                for _ in range(2):
                    cap.grab()

                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning("🚫 Kameradan boş veri döndü.")
                    frame = None
                else:
                    if self.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    
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
        """Görüntü netleştirme (Unsharp Mask)."""
        # Eğer özel derlenmiş OpenCV varsa GPU kullan
        if self.cuda_available:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_blur = self.gpu_filter.apply(gpu_frame)
                # GPU üzerinde ağırlıklı toplama
                res_gpu = cv2.cuda.addWeighted(gpu_frame, 1.5, gpu_blur, -0.5, 0)
                return res_gpu.download()
            except Exception:
                # GPU hatası olursa CPU'ya düş
                pass

        # CPU Modu (Standart ve Hızlı)
        # GaussianBlur CPU üzerinde oldukça hızlıdır.
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)

    def _convert_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Web UI için Base64 dönüşümü."""
        try:
            # Sıkıştırma kalitesini 85'ten 80'e çekerek hız kazanabiliriz
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{jpg_as_text}"
        except Exception as e:
            logger.error(f"Base64 dönüşüm hatası: {e}")
            return None

    def save_snapshot(self, prefix: str = "security") -> Optional[str]:
        """Anlık görüntüyü diske kaydeder."""
        frame = self.get_frame(raw=True, preprocess=True) # Snapshotlarda kalite için preprocess=True
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
        """Sistemdeki aktif kamera portlarını tarar."""
        active_ports = []
        # İlk 3 port genellikle yeterlidir, taramayı hızlandırmak için 5'ten 3'e düşürüldü
        for i in range(3):
            # Linux/Mac'te backend belirtmek taramayı hızlandırabilir
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    active_ports.append(i)
                cap.release()
        return active_ports
    
    def stop(self):
        """Servis kapanış işlemi."""
        with self.lock:
            if self._active_cap:
                self._active_cap.release()
            logger.info("🔌 Kamera servisi kapatıldı.")