"""
LotusAI Security & Authentication Manager
Sürüm: 2.5.3
Açıklama: Biyometrik kimlik doğrulama ve güvenlik yönetimi

Özellikler:
- Yüz tanıma (face_recognition + dlib)
- Ses tanıma (placeholder)
- GPU hızlandırma (CNN model)
- Kullanıcı kaydı
- Güvenlik durumu takibi
- Thread-safe operations
"""

import cv2
import logging
import numpy as np
import threading
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config
from core.user_manager import UserManager

logger = logging.getLogger("LotusAI.Security")


# ═══════════════════════════════════════════════════════════════
# EXTERNAL LIBRARIES
# ═══════════════════════════════════════════════════════════════
# Face Recognition
try:
    import face_recognition
    import dlib
    FACE_REC_AVAILABLE = True
    
    # GPU kontrolü
    GPU_AVAILABLE = Config.USE_GPU and dlib.DLIB_USE_CUDA
    
    if Config.USE_GPU:
        if dlib.DLIB_USE_CUDA:
            logger.info("🚀 Yüz Tanıma: GPU (CUDA) Modu Aktif")
        else:
            logger.warning("⚠️ Config GPU açık ama Dlib CUDA görmüyor, CPU kullanılacak")
    
except ImportError:
    FACE_REC_AVAILABLE = False
    GPU_AVAILABLE = False
    logger.warning("⚠️ face_recognition/dlib yüklü değil, yüz tanıma devre dışı")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class SecurityStatus(Enum):
    """Güvenlik durumları"""
    APPROVED = "ONAYLI"
    VOICE_APPROVED = "SES_ONAYLI"
    QUESTIONING = "SORGULAMA"
    WAITING = "BEKLEME"
    INITIALIZING = "BAŞLATILIYOR"
    
    @property
    def is_authenticated(self) -> bool:
        """Kimlik doğrulandı mı"""
        return self in {SecurityStatus.APPROVED, SecurityStatus.VOICE_APPROVED}


class SecuritySubStatus(Enum):
    """Güvenlik alt durumları"""
    VISUAL_VERIFICATION_CNN = "GÖRSEL_DOGRULAMA_CNN"
    VISUAL_VERIFICATION_HOG = "GÖRSEL_DOGRULAMA_HOG"
    VOICE_VERIFICATION = "SESLI_KIMLIK_DOGRULAMA"
    INTRO_MODE = "TANIŞMA_MODU"
    CAMERA_EMPTY = "KAMERA_BOŞ"
    SAFE_MODE = "GÜVENLİK_MODU"
    INITIALIZING = "BAŞLATILIYOR"


class RecognitionModel(Enum):
    """Yüz tanıma modelleri"""
    CNN = "cnn"  # GPU gerektirir, daha doğru
    HOG = "hog"  # CPU, hızlı


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
class SecurityConfig:
    """Security manager konfigürasyonu"""
    # Recognition tolerances
    FACE_TOLERANCE = 0.45  # Düşük = sıkı kontrol
    VOICE_CONFIDENCE_THRESHOLD = 0.70
    
    # Stability tracking
    MAX_INSTABILITY_COUNT = 12
    
    # Processing intervals (saniye)
    PROCESS_INTERVAL_CNN = 0.2  # GPU hızlı
    PROCESS_INTERVAL_HOG = 0.4  # CPU daha yavaş
    
    # Frame scaling
    FRAME_SCALE_HOG = 0.5
    FRAME_SCALE_CNN = 0.8
    FRAME_SCALE_DETECTION = 0.4
    
    # Encoding
    NUM_JITTERS = 1  # Daha fazla = daha doğru ama yavaş
    
    # Cache
    ENABLE_FRAME_CACHE = True
    CACHE_DURATION_SECONDS = 1.0


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class SecurityResult:
    """Güvenlik analizi sonucu"""
    status: SecurityStatus
    user: Optional[Dict[str, Any]]
    sub_status: SecuritySubStatus
    confidence: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_tuple(self) -> Tuple[str, Any, str]:
        """Geriye uyumluluk için tuple formatı"""
        return (self.status.value, self.user, self.sub_status.value)


@dataclass
class FaceEncoding:
    """Yüz encoding bilgisi"""
    user_id: str
    encoding: np.ndarray
    registered_date: datetime


@dataclass
class SecurityMetrics:
    """Güvenlik metrikleri"""
    total_recognitions: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    stranger_detections: int = 0
    average_recognition_time: float = 0.0


# ═══════════════════════════════════════════════════════════════
# SECURITY MANAGER
# ═══════════════════════════════════════════════════════════════
class SecurityManager:
    """
    LotusAI Güvenlik ve Kimlik Doğrulama Yöneticisi
    
    Sorumluluklar:
    - Yüz tanıma (face recognition)
    - Ses tanıma (voice recognition)
    - Kullanıcı kaydı
    - Güvenlik durumu takibi
    - Biyometrik veri yönetimi
    
    Thread-safe design ile concurrent access desteklenir.
    """
    
    def __init__(
        self,
        camera_manager: Any,
        memory_manager: Optional[Any] = None
    ):
        """
        Security manager başlatıcı
        
        Args:
            camera_manager: Kamera yöneticisi
            memory_manager: Hafıza yöneticisi (opsiyonel)
        """
        self.camera_manager = camera_manager
        self.memory = memory_manager
        self.user_manager = UserManager()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Face encodings
        self.face_encodings: List[FaceEncoding] = []
        self.voice_profiles: Dict[str, str] = {}
        
        # Stability tracking
        self.instability_counter = 0
        self.last_known_state = SecurityResult(
            status=SecurityStatus.WAITING,
            user=None,
            sub_status=SecuritySubStatus.INITIALIZING
        )
        
        # Performance tracking
        self.last_process_time = 0.0
        self.metrics = SecurityMetrics()
        
        # Frame cache
        self.cached_frame = None
        self.cache_timestamp = None
        
        # Directories
        self.work_dir = Config.WORK_DIR
        self.faces_dir = self.work_dir / "faces"
        self.voices_dir = self.work_dir / "voices"
        
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self._configure_model()
        
        # Load identities
        if FACE_REC_AVAILABLE:
            self.reload_identities()
        
        logger.info("✅ SecurityManager başlatıldı")
    
    def _configure_model(self) -> None:
        """Model yapılandırması"""
        # GPU status
        actual_gpu_status = Config.USE_GPU and GPU_AVAILABLE
        
        # Model type
        if hasattr(Config, 'FACE_REC_MODEL'):
            self.model_type = RecognitionModel(Config.FACE_REC_MODEL)
        else:
            self.model_type = (
                RecognitionModel.CNN if actual_gpu_status
                else RecognitionModel.HOG
            )
        
        # GPU yoksa CNN kullanılamaz
        if not actual_gpu_status and self.model_type == RecognitionModel.CNN:
            logger.warning("⚠️ GPU yok, CNN yerine HOG kullanılıyor")
            self.model_type = RecognitionModel.HOG
        
        # Processing interval
        self.process_interval = (
            SecurityConfig.PROCESS_INTERVAL_CNN
            if self.model_type == RecognitionModel.CNN
            else SecurityConfig.PROCESS_INTERVAL_HOG
        )
        
        # Tolerance
        self.recognition_tolerance = getattr(
            Config,
            'FACE_TOLERANCE',
            SecurityConfig.FACE_TOLERANCE
        )
        
        logger.info(
            f"🛡️ Security Model: {self.model_type.value.upper()} "
            f"(Interval: {self.process_interval}s)"
        )
    
    # ───────────────────────────────────────────────────────────
    # IDENTITY MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def reload_identities(self) -> None:
        """Kullanıcı veritabanından biyometrik verileri yükle"""
        with self.lock:
            self.face_encodings.clear()
            self.voice_profiles.clear()
            
            logger.info("👤 Kimlik veritabanı güncelleniyor...")
            
            loaded_faces = 0
            loaded_voices = 0
            
            for user_id, user_data in self.user_manager.users.items():
                # Face encoding
                face_file = user_data.get("face_file")
                if face_file:
                    if self._load_face_encoding(user_id, face_file):
                        loaded_faces += 1
                
                # Voice profile
                voice_file = user_data.get("voice_file")
                if voice_file:
                    self.voice_profiles[user_id] = voice_file
                    loaded_voices += 1
            
            logger.info(
                f"✅ Yüklendi: {loaded_faces} yüz, {loaded_voices} ses profili"
            )
    
    def _load_face_encoding(self, user_id: str, face_file: str) -> bool:
        """
        Tek bir yüz encoding'i yükle
        
        Args:
            user_id: Kullanıcı ID
            face_file: Yüz dosya yolu
        
        Returns:
            Başarılı ise True
        """
        img_path = self.work_dir / face_file
        
        if not img_path.exists():
            logger.warning(f"⚠️ Dosya bulunamadı: {img_path}")
            return False
        
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(
                image,
                num_jitters=SecurityConfig.NUM_JITTERS
            )
            
            if encodings:
                self.face_encodings.append(
                    FaceEncoding(
                        user_id=user_id,
                        encoding=encodings[0],
                        registered_date=datetime.now()
                    )
                )
                return True
            else:
                logger.warning(f"⚠️ Yüz bulunamadı: {user_id}")
        
        except Exception as e:
            logger.error(f"❌ Encoding hatası ({user_id}): {e}")
        
        return False
    
    def register_new_visitor(
        self,
        name: str,
        audio_data: Optional[Any] = None
    ) -> Tuple[bool, str]:
        """
        Yeni kullanıcı kaydı
        
        Args:
            name: Kullanıcı adı
            audio_data: Ses verisi (opsiyonel)
        
        Returns:
            Tuple[başarı durumu, mesaj]
        """
        if not FACE_REC_AVAILABLE:
            return False, "❌ Yüz tanıma modülü yüklü değil"
        
        # İsim temizleme
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', name.lower().replace(" ", "_"))
        
        # Duplicate check
        if any(
            u.get('name', '').lower() == name.lower()
            for u in self.user_manager.users.values()
        ):
            return False, f"❌ '{name}' zaten kayıtlı"
        
        # Frame al
        frame = self._get_current_frame()
        if frame is None:
            return False, "❌ Kamera görüntüsü alınamadı"
        
        # Yüz tespiti
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=self.model_type.value
            )
        except Exception as e:
            logger.error(f"Yüz tespiti hatası: {e}")
            return False, "❌ Yüz tespiti başarısız"
        
        if not face_locations:
            return False, "❌ Yüz tespit edilemedi, kameraya daha net bakın"
        
        # Yüz kaydet
        timestamp = int(time.time())
        face_filename = f"{clean_name}_{timestamp}.jpg"
        face_path = self.faces_dir / face_filename
        
        cv2.imwrite(str(face_path), frame)
        
        # Ses kaydet (opsiyonel)
        voice_rel_path = None
        if audio_data:
            voice_rel_path = self._save_voice_sample(clean_name, audio_data, timestamp)
        
        # Kullanıcı oluştur
        success = self.user_manager.create_new_user(
            name=name,
            level=2,
            face_file=f"faces/{face_filename}",
            voice_file=voice_rel_path
        )
        
        if success:
            # Identities'i yeniden yükle
            self.reload_identities()
            
            # Memory'ye kaydet
            if self.memory:
                self.memory.save(
                    "SECURITY",
                    "system",
                    f"Yeni kullanıcı: {name}"
                )
            
            logger.info(f"✅ Kullanıcı kaydedildi: {name}")
            return True, f"✅ Seni sisteme kaydettim {name}"
        
        return False, "❌ Veritabanı yazma hatası"
    
    def _save_voice_sample(
        self,
        clean_name: str,
        audio_data: Any,
        timestamp: int
    ) -> Optional[str]:
        """
        Ses örneği kaydet
        
        Args:
            clean_name: Temizlenmiş isim
            audio_data: Ses verisi
            timestamp: Zaman damgası
        
        Returns:
            Relative path veya None
        """
        try:
            voice_filename = f"{clean_name}_{timestamp}.wav"
            voice_path = self.voices_dir / voice_filename
            
            with open(voice_path, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            return f"voices/{voice_filename}"
        
        except Exception as e:
            logger.error(f"Ses kayıt hatası: {e}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # RECOGNITION
    # ───────────────────────────────────────────────────────────
    
    def check_static_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Frame'de kayıtlı yüz ara
        
        Args:
            frame: Video frame
        
        Returns:
            Kullanıcı bilgisi veya None
        """
        if (not FACE_REC_AVAILABLE or
            frame is None or
            not self.face_encodings):
            return None
        
        start_time = time.time()
        
        try:
            # Frame scaling
            scale = (
                SecurityConfig.FRAME_SCALE_CNN
                if self.model_type == RecognitionModel.CNN
                else SecurityConfig.FRAME_SCALE_HOG
            )
            
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Face locations
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=self.model_type.value
            )
            
            if not face_locations:
                return None
            
            # Face encodings
            face_encodings_in_frame = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )
            
            # Compare
            for face_encoding in face_encodings_in_frame:
                match = self._match_face_encoding(face_encoding)
                
                if match:
                    # Metrics
                    recognition_time = time.time() - start_time
                    self._update_metrics(True, recognition_time)
                    
                    return match
            
            # No match
            self._update_metrics(False, time.time() - start_time)
        
        except Exception as e:
            logger.error(f"Frame check hatası: {e}")
        
        return None
    
    def _match_face_encoding(
        self,
        face_encoding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Encoding'i bilinen yüzlerle karşılaştır
        
        Args:
            face_encoding: Yüz encoding'i
        
        Returns:
            Kullanıcı bilgisi veya None
        """
        # Extract known encodings
        known_encodings = [fe.encoding for fe in self.face_encodings]
        
        # Compare
        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=self.recognition_tolerance
        )
        
        if True not in matches:
            return None
        
        # Find best match
        face_distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            user_id = self.face_encodings[best_match_index].user_id
            return self.user_manager.users.get(user_id)
        
        return None
    
    def recognize_speaker(
        self,
        audio_data: Any
    ) -> Tuple[Optional[str], float]:
        """
        Konuşmacıyı tanı (Placeholder)
        
        Args:
            audio_data: Ses verisi
        
        Returns:
            Tuple[user_id, confidence]
        """
        if not audio_data or not self.voice_profiles:
            return None, 0.0
        
        # TODO: Gerçek ses tanıma implementasyonu
        # Şu an için basit placeholder
        if hasattr(audio_data, 'frame_data') and len(audio_data.frame_data) > 4000:
            # İlk kullanıcıyı döndür (geçici)
            if self.voice_profiles:
                user_id = list(self.voice_profiles.keys())[0]
                return user_id, SecurityConfig.VOICE_CONFIDENCE_THRESHOLD
        
        return None, 0.0
    
    # ───────────────────────────────────────────────────────────
    # SITUATION ANALYSIS
    # ───────────────────────────────────────────────────────────
    
    def analyze_situation(
        self,
        audio_data: Optional[Any] = None
    ) -> Tuple[str, Any, str]:
        """
        Güvenlik durumu analizi (geriye uyumluluk için tuple döndürür)
        
        Args:
            audio_data: Ses verisi (opsiyonel)
        
        Returns:
            Tuple[status, user, sub_status]
        """
        result = self.analyze_situation_detailed(audio_data)
        return result.to_tuple()
    
    def analyze_situation_detailed(
        self,
        audio_data: Optional[Any] = None
    ) -> SecurityResult:
        """
        Detaylı güvenlik analizi
        
        Args:
            audio_data: Ses verisi (opsiyonel)
        
        Returns:
            SecurityResult objesi
        """
        current_time = time.time()
        
        # Rate limiting
        if (current_time - self.last_process_time) < self.process_interval:
            return self.last_known_state
        
        self.last_process_time = current_time
        
        # Face recognition yoksa safe mode
        if not FACE_REC_AVAILABLE:
            admin = self.user_manager.get_user_by_level(5) or {
                "name": "Admin",
                "level": 5
            }
            return SecurityResult(
                status=SecurityStatus.APPROVED,
                user=admin,
                sub_status=SecuritySubStatus.SAFE_MODE
            )
        
        # 1. Visual recognition
        frame = self._get_current_frame()
        identified_user = self.check_static_frame(frame)
        
        if identified_user:
            self.instability_counter = 0
            self.camera_manager.last_seen_person = identified_user.get('name')
            
            # Log first recognition
            if self.last_known_state.user != identified_user:
                logger.info(
                    f"🛡️ Tanındı: {identified_user.get('name')} "
                    f"({self.model_type.value})"
                )
                
                if self.memory:
                    self.memory.save(
                        "SECURITY",
                        "system",
                        f"{identified_user.get('name')} algılandı"
                    )
            
            sub_status = (
                SecuritySubStatus.VISUAL_VERIFICATION_CNN
                if self.model_type == RecognitionModel.CNN
                else SecuritySubStatus.VISUAL_VERIFICATION_HOG
            )
            
            result = SecurityResult(
                status=SecurityStatus.APPROVED,
                user=identified_user,
                sub_status=sub_status,
                confidence=1.0
            )
            
            self.last_known_state = result
            return result
        
        # 2. Stranger detection
        if frame is not None and self._detect_stranger(frame):
            self.instability_counter = 0
            self.camera_manager.last_seen_person = "Yabancı"
            
            result = SecurityResult(
                status=SecurityStatus.QUESTIONING,
                user={"name": "Yabancı", "level": 0},
                sub_status=SecuritySubStatus.INTRO_MODE
            )
            
            self.last_known_state = result
            self.metrics.stranger_detections += 1
            return result
        
        # 3. Voice recognition
        if audio_data:
            user_id, confidence = self.recognize_speaker(audio_data)
            
            if user_id and confidence >= SecurityConfig.VOICE_CONFIDENCE_THRESHOLD:
                user = self.user_manager.users.get(user_id)
                
                if user:
                    self.instability_counter = 0
                    
                    result = SecurityResult(
                        status=SecurityStatus.VOICE_APPROVED,
                        user=user,
                        sub_status=SecuritySubStatus.VOICE_VERIFICATION,
                        confidence=confidence
                    )
                    
                    self.last_known_state = result
                    return result
        
        # 4. Stability tracking
        if self.instability_counter < SecurityConfig.MAX_INSTABILITY_COUNT:
            self.instability_counter += 1
            
            # Keep previous authenticated state
            if self.last_known_state.status.is_authenticated:
                return self.last_known_state
        
        # 5. Default: waiting
        self.camera_manager.last_seen_person = None
        
        result = SecurityResult(
            status=SecurityStatus.WAITING,
            user=None,
            sub_status=SecuritySubStatus.CAMERA_EMPTY
        )
        
        self.last_known_state = result
        return result
    
    def _detect_stranger(self, frame: np.ndarray) -> bool:
        """
        Frame'de yabancı yüz var mı tespit et
        
        Args:
            frame: Video frame
        
        Returns:
            Yabancı tespit edildiyse True
        """
        try:
            small = cv2.resize(
                frame,
                (0, 0),
                fx=SecurityConfig.FRAME_SCALE_DETECTION,
                fy=SecurityConfig.FRAME_SCALE_DETECTION
            )
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(
                rgb,
                model=self.model_type.value
            )
            
            return len(face_locations) > 0
        
        except Exception as e:
            logger.error(f"Stranger detection hatası: {e}")
            return False
    
    # ───────────────────────────────────────────────────────────
    # FRAME MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def _get_current_frame(self) -> Optional[np.ndarray]:
        """
        Mevcut frame'i al (cache ile)
        
        Returns:
            Video frame veya None
        """
        if not SecurityConfig.ENABLE_FRAME_CACHE:
            return self.camera_manager.get_frame()
        
        now = time.time()
        
        # Cache geçerliyse kullan
        if (self.cached_frame is not None and
            self.cache_timestamp is not None and
            (now - self.cache_timestamp) < SecurityConfig.CACHE_DURATION_SECONDS):
            return self.cached_frame
        
        # Yeni frame al
        frame = self.camera_manager.get_frame()
        
        if frame is not None:
            self.cached_frame = frame
            self.cache_timestamp = now
        
        return frame
    
    # ───────────────────────────────────────────────────────────
    # METRICS
    # ───────────────────────────────────────────────────────────
    
    def _update_metrics(self, success: bool, recognition_time: float) -> None:
        """
        Metrikleri güncelle
        
        Args:
            success: Tanıma başarılı mı
            recognition_time: Tanıma süresi (saniye)
        """
        self.metrics.total_recognitions += 1
        
        if success:
            self.metrics.successful_recognitions += 1
        else:
            self.metrics.failed_recognitions += 1
        
        # Moving average
        n = self.metrics.total_recognitions
        self.metrics.average_recognition_time = (
            (self.metrics.average_recognition_time * (n - 1) + recognition_time) / n
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Güvenlik metriklerini döndür
        
        Returns:
            Metrik dictionary'si
        """
        success_rate = 0.0
        if self.metrics.total_recognitions > 0:
            success_rate = (
                self.metrics.successful_recognitions /
                self.metrics.total_recognitions * 100
            )
        
        return {
            "model": self.model_type.value,
            "gpu_enabled": GPU_AVAILABLE,
            "total_recognitions": self.metrics.total_recognitions,
            "successful_recognitions": self.metrics.successful_recognitions,
            "failed_recognitions": self.metrics.failed_recognitions,
            "stranger_detections": self.metrics.stranger_detections,
            "success_rate": round(success_rate, 2),
            "avg_recognition_time": round(
                self.metrics.average_recognition_time * 1000,
                2
            ),  # ms
            "loaded_faces": len(self.face_encodings),
            "loaded_voices": len(self.voice_profiles)
        }
    
    # ───────────────────────────────────────────────────────────
    # CLEANUP
    # ───────────────────────────────────────────────────────────
    
    def shutdown(self) -> None:
        """Security manager'ı kapat"""
        logger.info("SecurityManager kapatılıyor...")
        
        with self.lock:
            # Cache temizle
            self.cached_frame = None
            self.cache_timestamp = None
            
            # Encodings temizle
            self.face_encodings.clear()
            self.voice_profiles.clear()
        
        logger.info("✅ SecurityManager kapatıldı")