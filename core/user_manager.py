import json
import shutil
import threading
import logging
import re
import torch # GPU kontrolü ve tensör işlemleri için eklendi
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from config import Config

# --- LOGGING ---
logger = logging.getLogger("LotusAI.UserManager")

class UserManager:
    """
    LotusAI User Authority, Profile, and Identity Management System.
    
    Features:
    - RBAC (Role-Based Access Control): 6-level authorization.
    - GPU Awareness: Detects and prepares for GPU-accelerated identity tasks.
    - JSON DB: Persistent storage with automated backup.
    - Thread-Safe: Multi-agent compatible locking mechanism (RLock).
    - Auto-Migration: Automatically updates user data structures.
    """
    
    # --- AUTHORIZATION LEVELS ---
    LVL_PATRON = 5      # Full Authority: Halil & Hatice Sevim (System, Finance, Security)
    LVL_SEF = 4         # Manager: Operational authorities
    LVL_CALISAN = 3      # Personnel: Task tracking and inventory
    LVL_MISAFIR = 2      # Standard: General chat and info
    LVL_BILINMIYOR = 1   # Restricted: Unidentified persons
    LVL_TEHLIKE = 0      # Forbidden: Triggers security alarms

    def __init__(self):
        # Path configuration
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.db_file = self.work_dir / "users_db.json"
        self.backup_file = self.db_file.with_suffix(".json.backup")
        
        # Thread safety lock
        self._lock = threading.RLock() 
        
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU Check
        self.device = self._detect_hardware()
        
        # Load database
        self.users = self._load_db()
        logger.info(f"🚀 UserManager initialized on {self.device.upper()}.")

    def _detect_hardware(self) -> str:
        """Detects if a GPU is available for future AI/Identity tasks."""
        if torch.cuda.is_available():
            try:
                # Get GPU name for logging
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU Detected: {gpu_name}")
                return "cuda"
            except Exception:
                return "cuda"
        return "cpu"

    def _load_db(self) -> Dict[str, Any]:
        """Loads database, restores from backup on failure, or creates default."""
        with self._lock:
            if not self.db_file.exists():
                return self._create_default_db()
            
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Run migrations for data structure updates
                    if self._check_migrations(data):
                        self._save_db_internal(data)
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"⚠️ Database corrupted: {e}. Attempting backup recovery.")
                return self._restore_backup()
            except Exception as e:
                logger.error(f"❌ Critical User DB Error: {e}")
                return {}

    def _generate_user_id(self, name: str) -> str:
        """Generates a secure, slugified User ID from a name."""
        name = name.lower().strip()
        tr_map = str.maketrans("çğıöşü ", "cgiosu_")
        clean_name = name.translate(tr_map)
        return re.sub(r'[^a-z0-9_]', '', clean_name)

    def _create_default_db(self) -> Dict[str, Any]:
        """Creates initial family and personnel list during first setup."""
        now = datetime.now().isoformat()
        default_db = {
            "halil_sevim": {
                "name": "Halil Sevim",
                "level": self.LVL_PATRON,
                "face_file": "faces/halil_sevim.jpg",
                "face_embedding": None, # For GPU-based recognition
                "voice_id": "voice_halil", 
                "bio_file": "halil_bio.txt",
                "created_at": now,
                "last_seen": None,
                "status": "active"
            },
            "hatice_sevim": {
                "name": "Hatice Sevim",
                "level": self.LVL_PATRON,
                "face_file": "faces/hatice_sevim.jpg",
                "face_embedding": None,
                "voice_id": "voice_hatice",
                "bio_file": None,
                "created_at": now,
                "last_seen": None,
                "status": "active"
            },
            "bengi_nisa_sevim": {
                "name": "Bengi Nisa Sevim",
                "level": self.LVL_MISAFIR,
                "face_file": "faces/bengi_nisa.jpg",
                "face_embedding": None,
                "voice_id": None,
                "bio_file": None,
                "created_at": now,
                "last_seen": None,
                "status": "active"
            },
            "eray_sef": {
                "name": "Eray (Şef)",
                "level": self.LVL_SEF,
                "face_file": "faces/eray.jpg",
                "face_embedding": None,
                "voice_id": None,
                "bio_file": None,
                "created_at": now,
                "last_seen": None,
                "status": "active"
            },
            "abdullah_usta": {
                "name": "Abdullah Usta", 
                "level": self.LVL_CALISAN, 
                "face_file": "faces/abdullah.jpg", 
                "created_at": now,
                "status": "active"
            },
            "muzeyyen_abla": {
                "name": "Müzeyyen Abla", 
                "level": self.LVL_CALISAN, 
                "face_file": "faces/muzeyyen.jpg", 
                "created_at": now,
                "status": "active"
            }
        }
        
        self._save_db_internal(default_db)
        logger.info("✅ Default user database created.")
        return default_db

    def _restore_backup(self) -> Dict[str, Any]:
        """Restores backup if main DB fails."""
        if self.backup_file.exists():
            try:
                shutil.copy(self.backup_file, self.db_file)
                logger.info("✅ Database recovered from backup.")
                return self._load_db()
            except Exception as e:
                logger.error(f"❌ Backup recovery failed: {e}")
        return self._create_default_db()

    def _check_migrations(self, data: Dict[str, Any]) -> bool:
        """Automatically completes missing fields in the data structure."""
        changed = False
        default_fields = {
            "bio_file": None,
            "voice_id": None,
            "face_embedding": None, # New field for GPU compatibility
            "created_at": datetime.now().isoformat(),
            "last_seen": None,
            "status": "active"
        }
        
        for user_id, user_data in data.items():
            for field, default_val in default_fields.items():
                if field not in user_data:
                    if field == "bio_file" and user_id == "halil_sevim":
                        user_data[field] = "halil_bio.txt"
                    else:
                        user_data[field] = default_val
                    changed = True
        return changed

    def _save_db_internal(self, data: Dict[str, Any]):
        """Writes data to disk and takes a backup."""
        try:
            if self.db_file.exists():
                shutil.copy(self.db_file, self.backup_file)
            
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ User DB Save Error: {e}")

    def save(self):
        """Manually saves user data to disk."""
        with self._lock:
            self._save_db_internal(self.users)

    # --- QUERY METHODS ---

    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Returns full profile data for a user."""
        return self.users.get(user_id, {})

    def get_user_level(self, user_id: str) -> int:
        """Returns authority level and updates 'last seen'."""
        user = self.users.get(user_id)
        if user:
            user["last_seen"] = datetime.now().isoformat()
            return user.get("level", self.LVL_MISAFIR)
        return self.LVL_BILINMIYOR

    def get_user_by_level(self, level: int) -> Optional[Dict[str, Any]]:
        """Returns the first user found at a specific level (e.g., for finding Admin)."""
        for user in self.users.values():
            if user.get("level") == level:
                return user
        return None

    def get_all_users_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Lists all users at a specific authority level."""
        return [u for u in self.users.values() if u.get("level") == level]

    # --- USER OPERATIONS ---

    def create_new_user(self, name: str, level: int = None, face_file: str = None, voice_file: str = None) -> str:
        """
        Registers a new user or updates an existing one.
        Optimized to handle identity files for AI modules.
        """
        with self._lock:
            user_id = self._generate_user_id(name)
            
            if user_id in self.users:
                u = self.users[user_id]
                if face_file: u["face_file"] = face_file
                if voice_file: u["voice_file"] = voice_file
                if level is not None: u["level"] = level
                u["last_seen"] = datetime.now().isoformat()
                logger.info(f"🔄 User updated: {name} ({user_id})")
            else:
                self.users[user_id] = {
                    "name": name.title(),
                    "level": level if level is not None else self.LVL_MISAFIR,
                    "face_file": face_file, 
                    "voice_file": voice_file,
                    "face_embedding": None, # Placeholder for GPU embeddings
                    "bio_file": None,
                    "created_at": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "status": "active"
                }
                logger.info(f"🆕 New user created: {name}")
            
            self._save_db_internal(self.users)
            return user_id

    def delete_user(self, user_id: str) -> bool:
        """Permanently deletes a user from the database."""
        with self._lock:
            if user_id in self.users:
                del self.users[user_id]
                self._save_db_internal(self.users)
                logger.warning(f"🗑️ User deleted: {user_id}")
                return True
            return False

    def update_user_status(self, user_id: str, status: str):
        """Updates user status (active, inactive, banned)."""
        with self._lock:
            if user_id in self.users:
                self.users[user_id]["status"] = status
                self._save_db_internal(self.users)
                
    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns the current hardware environment of the UserManager."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }



# import json
# import os
# import shutil
# from datetime import datetime
# from config import Config

# class UserManager:
#     """
#     LotusAI Kullanıcı Yetki ve Kimlik Yönetim Sistemi.
#     Kimlik doğrulama, yetkilendirme (RBAC) ve kullanıcı veritabanı yönetimini sağlar.
#     """
    
#     # YETKİ SEVİYELERİ (Role-Based Access Control)
#     LVL_PATRON = 5      # Tam Yetki: Sistem, Finans, Güvenlik
#     LVL_SEF = 4          # Yönetici: Operasyon ve Raporlama
#     LVL_CALISAN = 3      # Personel: Stok ve Rezervasyon
#     LVL_MISAFIR = 2      # Standart: Genel bilgi ve sohbet
#     LVL_BILINMIYOR = 1   # Kısıtlı: Henüz tanımlanmamış kişiler
#     LVL_TEHLIKE = 0      # Yasaklı: Alarm durumunu tetikleyen kişiler

#     def __init__(self):
#         # Dosya yollarını Config üzerinden dinamik alıyoruz
#         self.db_file = os.path.join(Config.WORK_DIR, "users_db.json")
#         self.backup_file = self.db_file + ".backup"
#         self.users = self._load_db()

#     def _load_db(self):
#         """Veritabanını yükler, hata durumunda yedekten geri döner."""
#         if not os.path.exists(self.db_file):
#             return self._create_default_db()
        
#         try:
#             with open(self.db_file, "r", encoding="utf-8") as f:
#                 data = json.load(f)
                
#                 # Migrasyon kontrolü ve otomatik güncelleme
#                 if self._check_migrations(data):
#                     self._save_db(data)
#                 return data
#         except (json.JSONDecodeError, IOError) as e:
#             print(f"⚠️ HATA: Veritabanı okunamadı ({e}). Yedekten kurtarma deneniyor...")
#             return self._restore_backup()
#         except Exception as e:
#             print(f"❌ Beklenmedik User DB Hatası: {e}")
#             return {}

#     def _generate_user_id(self, name):
#         """İsimden güvenli ve standart bir kullanıcı ID oluşturur."""
#         name = name.lower().strip().replace(" ", "_")
#         tr_map = str.maketrans("çğıöşü", "cgiosu")
#         return name.translate(tr_map)

#     def _create_default_db(self):
#         """Proje gereksinimlerine göre varsayılan kullanıcı listesini oluşturur."""
#         default_db = {
#             # --- YÖNETİM ---
#             "halil_sevim": {
#                 "name": "Halil Sevim",
#                 "level": self.LVL_PATRON,
#                 "face_file": "faces/halil_sevim.jpg",     
#                 "voice_id": "voice_halil", 
#                 "bio_file": "halil_bio.txt",
#                 "created_at": datetime.now().isoformat()
#             },
#             "hatice_sevim": {
#                 "name": "Hatice Sevim",
#                 "level": self.LVL_PATRON,
#                 "face_file": "faces/hatice_sevim.jpg",
#                 "voice_id": "voice_hatice",
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },
            
#             # --- AİLE ---
#             "bengi_nisa_sevim": {
#                 "name": "Bengi Nisa Sevim",
#                 "level": self.LVL_MISAFIR,
#                 "face_file": "faces/bengi_nisa.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },
#             "eylul_erva_sevim": {
#                 "name": "Eylül Erva Sevim",
#                 "level": self.LVL_MISAFIR,
#                 "face_file": "faces/eylul_erva.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },

#             # --- PERSONEL & ŞEFLER ---
#             "eray_sef": {
#                 "name": "Eray (Şef)",
#                 "level": self.LVL_SEF,
#                 "face_file": "faces/eray.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },
#             "abdullah_usta": {
#                 "name": "Abdullah Usta",
#                 "level": self.LVL_CALISAN,
#                 "face_file": "faces/abdullah.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },
#             "muzeyyen_abla": {
#                 "name": "Müzeyyen Abla",
#                 "level": self.LVL_CALISAN,
#                 "face_file": "faces/muzeyyen.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             },
#             "orcun_kurye": {
#                 "name": "Orçun (Kurye)",
#                 "level": self.LVL_CALISAN,
#                 "face_file": "faces/orcun.jpg",
#                 "voice_id": None,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat()
#             }
#         }
        
#         self._save_db(default_db)
#         print("✅ Varsayılan kullanıcı veritabanı başarıyla oluşturuldu.")
#         return default_db

#     def _restore_backup(self):
#         """Bozulma durumunda yedek dosyayı devreye alır."""
#         if os.path.exists(self.backup_file):
#             try:
#                 shutil.copy(self.backup_file, self.db_file)
#                 print("✅ Veritabanı yedek dosyadan başarıyla kurtarıldı.")
#                 return self._load_db()
#             except Exception as e:
#                 print(f"❌ Kritik Hata: Yedek dosyası da kurtarılamadı: {e}")
#         return self._create_default_db()

#     def _check_migrations(self, data):
#         """Eski veritabanı şemalarını yeni özelliklerle günceller."""
#         changed = False
#         for user_id, user_data in data.items():
#             # Eksik bio_file ekleme
#             if "bio_file" not in user_data:
#                 user_data["bio_file"] = "halil_bio.txt" if user_id == "halil_sevim" else None
#                 changed = True
#             # Eksik oluşturulma tarihi ekleme
#             if "created_at" not in user_data:
#                 user_data["created_at"] = datetime.now().isoformat()
#                 changed = True
#         return changed

#     def _save_db(self, data):
#         """Verileri güvenli bir şekilde (önce yedek alarak) kaydeder."""
#         try:
#             if os.path.exists(self.db_file):
#                 shutil.copy(self.db_file, self.backup_file)
            
#             with open(self.db_file, "w", encoding="utf-8") as f:
#                 json.dump(data, f, indent=4, ensure_ascii=False)
#         except Exception as e:
#             print(f"❌ User DB Kayıt Hatası: {e}")

#     def get_user_by_face(self, face_encoding_match_id):
#         """Yüz tanıma ID'sine göre kullanıcıyı döndürür."""
#         return self.users.get(face_encoding_match_id)
        
#     def get_user_data(self, user_id):
#         """Kullanıcının tüm profil verilerini döndürür."""
#         return self.users.get(user_id, {})

#     def get_user_level(self, user_id):
#         """Kullanıcının yetki seviyesini kontrol eder."""
#         user = self.users.get(user_id)
#         if user:
#             # Kullanıcı her sorgulandığında 'son görülme' tarihini güncelleyebiliriz
#             user["last_seen"] = datetime.now().isoformat()
#             return user.get("level", self.LVL_MISAFIR)
#         return self.LVL_BILINMIYOR

#     def update_user_last_seen(self, user_id):
#         """Kullanıcının sistemle girdiği son etkileşim zamanını kaydeder."""
#         if user_id in self.users:
#             self.users[user_id]["last_seen"] = datetime.now().isoformat()
#             self._save_db(self.users)

#     def create_new_user(self, name, level=None, face_file=None, voice_id=None):
#         """
#         Sisteme yeni bir kullanıcı kaydeder veya mevcut kullanıcıyı günceller.
#         """
#         if level is None:
#             level = self.LVL_MISAFIR

#         user_id = self._generate_user_id(name)
        
#         if user_id in self.users:
#             # Mevcut kullanıcı güncelleme
#             if face_file: self.users[user_id]["face_file"] = face_file
#             if voice_id: self.users[user_id]["voice_id"] = voice_id
#             if level != self.LVL_MISAFIR: self.users[user_id]["level"] = level
#         else:
#             # Yeni kullanıcı oluşturma
#             self.users[user_id] = {
#                 "name": name.title(),
#                 "level": level,
#                 "face_file": face_file, 
#                 "voice_id": voice_id,
#                 "bio_file": None,
#                 "created_at": datetime.now().isoformat(),
#                 "last_seen": datetime.now().isoformat()
#             }
        
#         self._save_db(self.users)
#         return user_id

#     def delete_user(self, user_id):
#         """Kullanıcıyı veritabanından siler."""
#         if user_id in self.users:
#             del self.users[user_id]
#             self._save_db(self.users)
#             return True
#         return False