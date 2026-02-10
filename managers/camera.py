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

# --- LOGLAMA YAPILANDIRMASI ---
logger = logging.getLogger("LotusAI.Camera")

class CameraManager:
    """
    LotusAI Kamera Görüntü Yöneticisi.
    v2.6.5 - Dinamik Port Tarama ve CUDA Destekli Keskinleştirme.
    """
    
    def __init__(self):
        # İş parçacığı güvenliği için kilit
        self.lock = threading.RLock()
        
        # Durum Değişkenleri
        self.is_busy = False
        self._active_cap = None
        
        # OpenCV CUDA Kontrolü (Görüntü işleme hızlandırması için)
        self.cuda_available = False
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.cuda_available = True
                # Önbelleğe alınmış Gaussian filtresi
                self.gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (0, 0), 2.0)
                logger.info("🚀 Kamera: OpenCV CUDA Desteği Aktif (Görüntü işleme GPU üzerinde yapılacak)")
            else:
                logger.info("ℹ️ Kamera servisi CPU modunda başlatılıyor (Standart OpenCV).")
        except Exception:
            logger.info("ℹ️ Kamera servisi CPU modunda başlatılıyor.")

        # Dizin Yapılandırması
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.snapshot_dir = self.work_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Temel Ayarlar
        self.camera_index = getattr(Config, "CAMERA_INDEX", 0)
        self.resolution = (640, 480) 
        self.flip_horizontal = True  

    def start(self):
        """Kamera donanımını hazırlar. Hata durumunda alternatif portları tarar."""
        with self.lock:
            # Önce yapılandırmadaki varsayılan portu dene
            if self._test_hardware(self.camera_index):
                logger.info(f"✅ Kamera servisi hazır. (Port: {self.camera_index})")
            else:
                logger.warning(f"⚠️ Kamera (ID:{self.camera_index}) erişilemiyor. Aktif cihazlar taranıyor...")
                active_ports = self.list_cameras()
                
                if active_ports:
                    self.camera_index = active_ports[0]
                    logger.info(f"✅ Çalışan kamera bulundu ve seçildi: Port {self.camera_index}")
                else:
                    logger.error("❌ Sistemde erişilebilir hiçbir kamera bulunamadı!")

    def _test_hardware(self, index: int) -> bool:
        """Belirli bir porttaki kameranın görüntü verip vermediğini test eder."""
        try:
            # Linux sistemlerde V4L2 backend'i bazen daha kararlıdır
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if not cap.isOpened():
                return False
            
            # Kameranın gerçekten görüntü döndürdüğünü doğrula
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception:
            return False

    def list_cameras(self) -> List[int]:
        """Sistemdeki aktif kamera portlarını (0-4 arası) tarar."""
        active_ports = []
        # Modern sistemlerde genellikle 0-2 arası portlar kullanılır
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    active_ports.append(i)
                cap.release()
        return active_ports

    def get_frame(self, raw: bool = True, preprocess: bool = False) -> Optional[Union[np.ndarray, str]]:
        """
        Kameradan anlık bir kare yakalar ve opsiyonel olarak ön işlemeden geçirir.
        """
        if self.is_busy:
            return None

        with self.lock:
            self.is_busy = True
            frame = None
            cap = None
            
            try:
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    logger.error(f"❌ Kamera bağlantısı koptu! İndeks: {self.camera_index}")
                    return None

                # Donanım ayarlarını uygula
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Isınma döngüsü (Otomatik pozlamanın dengelenmesi için)
                for _ in range(2):
                    cap.grab()

                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning("🚫 Kameradan boş görüntü döndü.")
                    frame = None
                else:
                    # Görüntü yönünü düzelt (Ayna modu)
                    if self.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    
                    # Netleştirme ve iyileştirme
                    if preprocess:
                        frame = self._preprocess_frame(frame)

            except Exception as e:
                logger.error(f"❌ Kamera yakalama hatası: {e}")
            
            finally:
                if cap:
                    cap.release()
                self.is_busy = False

            # Dönüş formatını belirle
            if frame is not None:
                if not raw:
                    return self._convert_to_base64(frame)
                return frame
            
            return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Görüntü netleştirme işlemi (Unsharp Masking)."""
        if self.cuda_available:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_blur = self.gpu_filter.apply(gpu_frame)
                # Keskinlik artırma: Orijinal * 1.5 - Bulanık * 0.5
                res_gpu = cv2.cuda.addWeighted(gpu_frame, 1.5, gpu_blur, -0.5, 0)
                return res_gpu.download()
            except Exception:
                pass

        # CPU tabanlı hızlı netleştirme
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)

    def _convert_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Web arayüzünde gösterim için görüntüyü Base64 formatına çevirir."""
        try:
            # JPG sıkıştırma (Kalite: 80)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{jpg_as_text}"
        except Exception as e:
            logger.error(f"Base64 dönüşüm hatası: {e}")
            return None

    def save_snapshot(self, prefix: str = "guvenlik") -> Optional[str]:
        """Anlık görüntüyü snapshot dizinine kaydeder."""
        frame = self.get_frame(raw=True, preprocess=True)
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

    def stop(self):
        """Kamera servisini güvenli bir şekilde kapatır."""
        with self.lock:
            if self._active_cap:
                self._active_cap.release()
            logger.info("🔌 Kamera servisi durduruldu.")