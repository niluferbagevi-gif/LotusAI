import cv2
import logging
import numpy as np
import threading
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Proje içi modüller
from config import Config
from core.user_manager import UserManager

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Security")

# face_recognition ve dlib kontrolü
FACE_REC_AVAILABLE = False
GPU_AVAILABLE = False

try:
    import face_recognition
    import dlib
    FACE_REC_AVAILABLE = True
    # dlib'in CUDA ile derlenip derlenmediğini kontrol et
    GPU_AVAILABLE = dlib.DLIB_USE_CUDA
    if GPU_AVAILABLE:
        logger.info("🚀 GPU Desteği (CUDA) tespit edildi. Yüz tanıma GPU üzerinden çalışacak.")
    else:
        logger.warning("⚠️ CUDA tespit edilemedi. Yüz tanıma CPU (HOG) üzerinden devam edecek.")
except ImportError:
    logger.warning("⚠️ 'face_recognition' veya 'dlib' kütüphanesi bulunamadı. Yüz tanıma devre dışı.")

class SecurityManager:
    """
    LotusAI Güvenlik ve Kimlik Doğrulama Yöneticisi (GPU Optimize Edilmiş).
    
    Yetenekler:
    - Yüz Tanıma: GPU (CNN) veya CPU (HOG) tabanlı anlık doğrulama.
    - Ses İmzası: Konuşmacı teşhisi altyapısı.
    - Ziyaretçi Yönetimi: Biyometrik veri kaydı.
    - Akıllı Durum Takibi: Stabilite kontrolü ve erişim yönetimi.
    """
    
    def __init__(self, camera_manager, memory_manager=None):
        self.camera_manager = camera_manager
        self.memory = memory_manager
        self.user_manager = UserManager()
        self.lock = threading.Lock()
        
        # Bellekte tutulan kimlik verileri
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_voice_profiles = {} # {user_id: path}
        
        # Takip ve Stabilite Ayarları
        self.instability_counter = 0 
        self.MAX_INSTABILITY = 12 
        self.last_known_state = ("BEKLEME", None, "BAŞLATILIYOR")
        self.last_process_time = 0
        
        # Dizinler
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.faces_dir = self.work_dir / "faces"
        self.voices_dir = self.work_dir / "voices"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # GPU ve Model Ayarları
        # Eğer GPU varsa varsayılan 'cnn', yoksa 'hog'
        default_model = 'cnn' if GPU_AVAILABLE else 'hog'
        self.model_type = getattr(Config, 'FACE_REC_MODEL', default_model)
        
        # GPU (CNN) kullanılıyorsa analiz sıklığı artırılabilir, CPU ise tasarruf modunda kalır
        self.process_interval = 0.2 if self.model_type == 'cnn' else 0.4
        self.recognition_tolerance = getattr(Config, 'FACE_TOLERANCE', 0.45) 
        
        if FACE_REC_AVAILABLE:
            logger.info(f"🛡️ Güvenlik Modeli: {self.model_type.upper()} aktif.")
            self.reload_identities()

    def reload_identities(self):
        """Kullanıcı veritabanındaki tüm biyometrik verileri belleğe yükler."""
        with self.lock:
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_voice_profiles = {}
            
            logger.info("👤 Kimlik Veritabanı Belleğe Yükleniyor...")
            
            for user_id, user_data in self.user_manager.users.items():
                face_file = user_data.get("face_file")
                if face_file:
                    img_path = self.work_dir / face_file
                    if img_path.exists():
                        try:
                            # Görüntüyü yükle
                            image = face_recognition.load_image_file(str(img_path))
                            # Encoding işlemi (Burası da GPU varsa hızlanır)
                            # num_jitters=1: Daha hızlı, num_jitters=10: Daha doğru
                            encodings = face_recognition.face_encodings(image, num_jitters=1)
                            if encodings:
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(user_id)
                        except Exception as e:
                            logger.error(f"❌ {user_id} yüz verisi hatası: {e}")
                
                voice_file = user_data.get("voice_file")
                if voice_file:
                    self.known_voice_profiles[user_id] = voice_file

            logger.info(f"✅ Kimlik Yükleme Tamamlandı: {len(self.known_face_names)} yüz profili aktif.")

    def register_new_visitor(self, name: str, audio_data=None) -> Tuple[bool, str]:
        """Anlık kamera görüntüsüyle yeni kullanıcı kaydeder (GPU Destekli TESPİT)."""
        if not FACE_REC_AVAILABLE:
            return False, "Hata: Yüz tanıma modülü sistemde yüklü değil."

        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', name.lower().replace(" ", "_"))
        
        if any(u.get('name').lower() == name.lower() for u in self.user_manager.users.values()):
            return False, f"Hata: '{name}' ismiyle zaten bir kullanıcı kayıtlı."

        frame = self.camera_manager.get_frame()
        if frame is None:
            return False, "Hata: Kamera görüntüsü alınamadı."

        # Yüz Tespiti (GPU modunda cnn kullanılır)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model_type)
        except Exception as e:
            logger.error(f"Yüz tespiti hatası (Model: {self.model_type}): {e}")
            face_locations = []
        
        if not face_locations:
            return False, "Yüz tespit edilemedi. Lütfen kameraya daha net bakın."

        face_filename = f"{clean_name}_{int(time.time())}.jpg"
        face_path = self.faces_dir / face_filename
        cv2.imwrite(str(face_path), frame)
        
        voice_rel_path = None
        if audio_data:
            try:
                voice_filename = f"{clean_name}_{int(time.time())}.wav"
                voice_path = self.voices_dir / voice_filename
                with open(voice_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
                voice_rel_path = f"voices/{voice_filename}"
            except Exception as e:
                logger.error(f"Ses kaydı başarısız: {e}")

        success = self.user_manager.create_new_user(
            name=name,
            level=2,
            face_file=f"faces/{face_filename}",
            voice_file=voice_rel_path
        )

        if success:
            self.reload_identities()
            if self.memory:
                self.memory.save("SECURITY", "system", f"Yeni kullanıcı kaydedildi: {name}")
            return True, f"Seni sisteme kaydettim {name}. Seni artık tanıyorum."
        
        return False, "Kritik Hata: Kullanıcı veritabanına yazılamadı."

    def check_static_frame(self, frame) -> Optional[Dict]:
        """Karede kayıtlı bir yüz olup olmadığını GPU/CPU üzerinden kontrol eder."""
        if not FACE_REC_AVAILABLE or frame is None or not self.known_face_encodings:
            return None

        try:
            # CNN (GPU) kullanılıyorsa görüntüyü çok fazla küçültmeye gerek yok, 
            # çünkü GPU zaten büyük kareleri hızlı işler. HOG için küçültme şart.
            scale = 0.5 if self.model_type == 'hog' else 0.8
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Model tipine göre (hog/cnn) konumları bul
            face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model_type)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.recognition_tolerance
                )
                
                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        user_id = self.known_face_names[best_match_index]
                        return self.user_manager.users.get(user_id)
        except Exception as e:
            logger.error(f"Kimlik doğrulama sırasında hata: {e}")
            
        return None

    def recognize_speaker(self, audio_data) -> Tuple[Optional[str], float]:
        """Ses teşhisi (Placeholder)."""
        if not audio_data or not self.known_voice_profiles:
            return None, 0.0

        if hasattr(audio_data, 'frame_data') and len(audio_data.frame_data) > 4000:
            candidate_id = list(self.known_voice_profiles.keys())[0]
            return candidate_id, 0.70

        return None, 0.0

    def analyze_situation(self, audio_data=None) -> Tuple[str, Any, str]:
        """Anlık güvenlik durum analizi."""
        current_time = time.time()
        
        if (current_time - self.last_process_time) < self.process_interval:
            return self.last_known_state

        self.last_process_time = current_time

        if not FACE_REC_AVAILABLE:
            admin = self.user_manager.get_user_by_level(5) or {"name": "Admin", "level": 5}
            return "ONAYLI", admin, "GÜVENLİ_MOD (Biyometrik Yok)"

        frame = self.camera_manager.get_frame()
        identified_user = self.check_static_frame(frame)

        # 1. Kayıtlı Kullanıcı
        if identified_user:
            self.instability_counter = 0
            self.camera_manager.last_seen_person = identified_user.get('name')
            
            if self.last_known_state[1] != identified_user:
                logger.info(f"🛡️ Erişim: {identified_user.get('name')} ({self.model_type})")
                if self.memory:
                    self.memory.save("SECURITY", "system", f"{identified_user.get('name')} algılandı.")
            
            self.last_known_state = ("ONAYLI", identified_user, f"GÖRSEL_DOGRULAMA_{self.model_type.upper()}")
            return self.last_known_state

        # 2. Yabancı Tespiti
        if frame is not None:
            # Yabancı tespiti için hızlı olması adına HOG kullanılabilir veya 
            # GPU varsa yine CNN ile devam edilir
            small = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if face_recognition.face_locations(rgb, model=self.model_type):
                self.instability_counter = 0
                self.camera_manager.last_seen_person = "Yabancı"
                self.last_known_state = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
                return self.last_known_state

        # 3. Sesli Onay
        if audio_data:
            user_id, confidence = self.recognize_speaker(audio_data)
            if user_id and confidence >= 0.70:
                user = self.user_manager.users.get(user_id)
                if user:
                    self.instability_counter = 0
                    self.last_known_state = ("SES_ONAYLI", user, "SESLI_KIMLIK_DOGRULAMA")
                    return self.last_known_state

        # 4. Takip Toleransı
        if self.instability_counter < self.MAX_INSTABILITY:
            self.instability_counter += 1
            if self.last_known_state[0] in ["ONAYLI", "SES_ONAYLI"]:
                return self.last_known_state 

        # 5. Boş Durum
        self.camera_manager.last_seen_person = None
        self.last_known_state = ("BEKLEME", None, "KAMERA_BOŞ")
        return self.last_known_state


# import cv2
# import logging
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import time

# # Proje içi modüller
# from config import Config
# from core.user_manager import UserManager

# # --- LOGLAMA ---
# logger = logging.getLogger("LotusAI.Security")

# # face_recognition opsiyonel kontrolü
# try:
#     import face_recognition
#     FACE_REC_AVAILABLE = True
# except ImportError:
#     FACE_REC_AVAILABLE = False
#     logger.warning("'face_recognition' kütüphanesi bulunamadı. Yüz tanıma devre dışı.")

# class SecurityManager:
#     """
#     LotusAI Güvenlik Yöneticisi.
#     Kamera görüntülerini analiz eder, yüzleri tanır ve ses imzasını kontrol eder.
#     Tespit edilen olayları MemoryManager'a raporlar.
#     """
#     def __init__(self, camera_manager, memory_manager=None):
#         """
#         SecurityManager'ı başlatır.
#         """
#         self.camera_manager = camera_manager
#         self.memory = memory_manager
#         self.user_manager = UserManager()
        
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_voice_files = {} # {user_id: voice_file_path}
        
#         # Kararsızlık ve Takip Ayarları
#         self.instability_counter = 0 
#         self.MAX_INSTABILITY = 8 # Bir nebze artırıldı (daha stabil takip için)
#         self.last_known_state = ("BEKLEME", None, "KAMERA_YOK")
#         self.last_process_time = 0
#         self.process_interval = 0.5 # Saniyede en fazla 2 kez yüz analizi yap (CPU dostu)
        
#         # Dizin Yapılandırması
#         self.faces_dir = Config.WORK_DIR / "faces"
#         self.voices_dir = Config.WORK_DIR / "voices"

#         self.faces_dir.mkdir(parents=True, exist_ok=True)
#         self.voices_dir.mkdir(parents=True, exist_ok=True)

#         self.model_type = getattr(Config, 'FACE_REC_MODEL', 'hog')
#         self.recognition_tolerance = getattr(Config, 'FACE_TOLERANCE', 0.50) # Düşük değer = daha katı tanıma
        
#         if FACE_REC_AVAILABLE:
#             self._load_identities()

#     def _load_identities(self):
#         """Kayıtlı kullanıcıların yüz ve ses verilerini belleğe yükler."""
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_voice_files = {}
        
#         logger.info("👤 Kimlik Veritabanı Yükleniyor...")
        
#         for user_id, user_data in self.user_manager.users.items():
#             # 1. Yüz Verisi Yükleme
#             raw_face_path = user_data.get("face_file")
#             if raw_face_path:
#                 img_path = Config.WORK_DIR / raw_face_path
                
#                 if img_path.exists():
#                     try:
#                         img = face_recognition.load_image_file(str(img_path))
#                         # Birden fazla yüz varsa ilkini al
#                         encodings = face_recognition.face_encodings(img)
#                         if encodings:
#                             self.known_face_encodings.append(encodings[0])
#                             self.known_face_names.append(user_id)
#                     except Exception as e:
#                         logger.error(f"{user_data.get('name', user_id)} yüz verisi yüklenemedi: {e}")
            
#             # 2. Ses Dosyası Yükleme
#             raw_voice_path = user_data.get("voice_file")
#             if raw_voice_path:
#                 voice_path = Config.WORK_DIR / raw_voice_path
#                 if voice_path.exists():
#                     self.known_voice_files[user_id] = str(voice_path)

#         logger.info(f"✅ Kimlik yükleme tamamlandı: {len(self.known_face_names)} yüz, {len(self.known_voice_files)} ses profili.")

#     def register_new_visitor(self, name, audio_data=None):
#         """Yeni bir ziyaretçiyi kamera ve ses verisiyle sisteme kaydeder."""
#         clean_name = "".join([c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_"])
        
#         # 1. Görsel Kayıt
#         frame = self.camera_manager.get_frame()
#         face_rel_path = None
        
#         if frame is not None and FACE_REC_AVAILABLE:
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             boxes = face_recognition.face_locations(rgb, model=self.model_type)
            
#             if boxes:
#                 face_filename = f"{clean_name}_{int(datetime.now().timestamp())}.jpg"
#                 face_full_path = self.faces_dir / face_filename
#                 cv2.imwrite(str(face_full_path), frame)
                
#                 face_rel_path = f"faces/{face_filename}"
                
#                 # Çalışma anında belleği güncelle
#                 encodings = face_recognition.face_encodings(rgb, boxes)
#                 if encodings:
#                     self.known_face_encodings.append(encodings[0])
#                     self.known_face_names.append(clean_name)
#                     logger.info(f"New face encoding added for {clean_name}")
#             else:
#                 return False, "Yüzünüzü tam göremiyorum, lütfen kameraya biraz daha yaklaşın."
#         else:
#             return False, "Kamera veya yüz tanıma modülü şu an aktif değil."

#         # 2. Ses Kaydı
#         voice_rel_path = None
#         if audio_data:
#             try:
#                 voice_filename = f"{clean_name}.wav"
#                 voice_full_path = self.voices_dir / voice_filename
#                 with open(voice_full_path, "wb") as f:
#                     f.write(audio_data.get_wav_data())
#                 voice_rel_path = f"voices/{voice_filename}"
#                 self.known_voice_files[clean_name] = str(voice_full_path)
#             except Exception as e:
#                 logger.error(f"Ses kayıt hatası: {e}")

#         # 3. Veritabanı ve Hafıza Kaydı
#         self.user_manager.create_new_user(
#             name=name, 
#             level=2, # Standart Misafir Seviyesi
#             face_file=face_rel_path, 
#             voice_file=voice_rel_path
#         )
        
#         if self.memory:
#             self.memory.save("SECURITY", "system", f"Sisteme yeni bir kullanıcı eklendi: {name}")
        
#         return True, f"Memnun oldum {name}, artık seni tanıyorum."

#     def check_static_frame(self, frame):
#         """Verilen kare üzerinden kimlik tespiti yapar."""
#         if not FACE_REC_AVAILABLE or frame is None or not self.known_face_encodings:
#             return None

#         try:
#             # Performans için: Görüntü boyutunu küçült (Hızı 4 kat artırır)
#             small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
#             # Yüz tespiti
#             face_locations = face_recognition.face_locations(rgb, model=self.model_type)
#             face_encodings = face_recognition.face_encodings(rgb, face_locations)
            
#             for face_encoding in face_encodings:
#                 # Bilinen yüzlerle karşılaştır
#                 matches = face_recognition.compare_faces(
#                     self.known_face_encodings, 
#                     face_encoding, 
#                     tolerance=self.recognition_tolerance
#                 )
                
#                 if True in matches:
#                     # En yakın eşleşmeyi bul (Euclidean distance)
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     best_match_index = np.argmin(face_distances)
                    
#                     if matches[best_match_index]:
#                         user_id = self.known_face_names[best_match_index]
#                         return self.user_manager.users.get(user_id)
                        
#         except Exception as e:
#             logger.error(f"Yüz tanıma motoru hatası: {e}")
            
#         return None

#     def recognize_speaker(self, audio_data):
#         """
#         Ses verisi üzerinden konuşanı teşhis eder.
#         Şu anlık temel uzunluk ve yapı kontrolü yapar, 
#         gelecekte tam spektral analiz için geliştirilmeye hazırdır.
#         """
#         if not audio_data or not self.known_voice_files:
#             return None, 0.0

#         # İleride: librosa.feature.mfcc karşılaştırması eklenecek.
#         # Temel mantık: Ses verisi varsa ve sistemde kayıtlı sesler varsa
#         # mevcut akışın bozulmaması için güvenli bir skor döner.
#         if hasattr(audio_data, 'frame_data') and len(audio_data.frame_data) > 3000:
#             # Şimdilik kayıtlı ilk kullanıcıyı simüle ediyor (Geliştirilecek)
#             candidate_id = list(self.known_voice_files.keys())[0]
#             return candidate_id, 0.85 
        
#         return None, 0.0

#     def analyze_situation(self, audio_data=None):
#         """
#         Sistemin o anki güvenlik durumunu analiz eder. 
#         Kamera ve ses verilerini birleştirerek 'ONAYLI', 'SORGULAMA' veya 'BEKLEME' döner.
#         """
#         current_time = time.time()
        
#         # CPU Tasarrufu: Çok sık analiz yapma
#         if (current_time - self.last_process_time) < self.process_interval:
#             return self.last_known_state

#         self.last_process_time = current_time

#         # Yüz tanıma kapalıysa varsayılan admin onayı
#         if not FACE_REC_AVAILABLE:
#             # Config'den veya UserManager'dan admini çek
#             admin = self.user_manager.get_user_by_level(5) # Seviye 5 = Admin
#             if not admin: # Fallback
#                  admin = {"name": "Admin", "level": 5}
#             return "ONAYLI", admin, "GÜVENLİ_MOD (Yüz Tanıma Devre Dışı)"

#         frame = self.camera_manager.get_frame()
        
#         # 1. GÖRSEL ANALİZ
#         identified_user = self.check_static_frame(frame)
        
#         if identified_user:
#             self.instability_counter = 0
#             self.camera_manager.last_seen_person = identified_user.get('name')
            
#             # Durum Değişikliği Logu
#             if self.last_known_state[1] != identified_user:
#                 logger.info(f"🛡️ Erişim Yetkisi Onaylandı: {identified_user.get('name')}")
#                 if self.memory:
#                     self.memory.save("SECURITY", "system", f"{identified_user.get('name')} tespit edildi ve erişim sağlandı.")
            
#             self.last_known_state = ("ONAYLI", identified_user, None)
#             return self.last_known_state

#         # 2. YABANCI TESPİTİ (Yüz var ama tanınmıyor)
#         if frame is not None:
#             rgb = cv2.cvtColor(cv2.resize(frame, (0,0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
#             if face_recognition.face_locations(rgb, model=self.model_type):
#                 self.instability_counter = 0
#                 self.camera_manager.last_seen_person = "Yabancı"
#                 self.last_known_state = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
#                 return self.last_known_state

#         # 3. SES İLE DOĞRULAMA (Görüntü net değilse veya kişi karanlıktaysa)
#         if audio_data:
#             v_user_id, confidence = self.recognize_speaker(audio_data)
#             if v_user_id and confidence > 0.80:
#                 user = self.user_manager.users.get(v_user_id)
#                 if user:
#                     self.instability_counter = 0
#                     logger.info(f"🎤 Ses İmzası ile Giriş: {user.get('name')}")
#                     self.last_known_state = ("SES_ONAYLI", user, "GÖRÜNTÜ_YOK_SES_VAR")
#                     return self.last_known_state

#         # 4. KARARSIZLIK KONTROLÜ
#         # Kişi anlık olarak kafasını çevirmiş veya ışık patlamış olabilir.
#         # Hemen 'BEKLEME'ye düşmemek için birkaç frame daha tolerans tanıyoruz.
#         if self.instability_counter < self.MAX_INSTABILITY:
#             self.instability_counter += 1
#             if self.last_known_state[0] in ["ONAYLI", "SES_ONAYLI"]:
#                 return self.last_known_state 
        
#         # 5. KİMSE YOK
#         self.camera_manager.last_seen_person = None
#         self.last_known_state = ("BEKLEME", None, "KAMERA_YOK")
#         return self.last_known_state