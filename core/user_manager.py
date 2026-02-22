"""
LotusAI User Management System
Sürüm: 2.5.4 (Eklendi: Erişim Seviyesi Desteği - uyumluluk için)
Açıklama: Kullanıcı yönetimi, yetkilendirme ve kimlik profilleri

Özellikler:
- RBAC (Role-Based Access Control)
- JSON persistent storage
- Auto-backup & migration
- Thread-safe operations
- GPU-aware identity management
- Audit logging
- Erişim seviyesi bilgisi (sistem modu için)
"""

import json
import shutil
import threading
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from contextlib import contextmanager

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.UserManager")


# ═══════════════════════════════════════════════════════════════
# TORCH (GPU)
# ═══════════════════════════════════════════════════════════════
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# USER LEVELS (RBAC)
# ═══════════════════════════════════════════════════════════════
class UserLevel(IntEnum):
    """
    Kullanıcı yetki seviyeleri (RBAC)
    
    Levels:
    - FORBIDDEN (0): Güvenlik alarmı tetikler
    - UNKNOWN (1): Tanınmamış kişi (kısıtlı erişim)
    - GUEST (2): Misafir (genel sohbet)
    - PERSONNEL (3): Personel (görev takibi, envanter)
    - MANAGER (4): Yönetici (operasyonel yetkiler)
    - OWNER (5): Patron (tam yetki: sistem, finans, güvenlik)
    """
    FORBIDDEN = 0
    UNKNOWN = 1
    GUEST = 2
    PERSONNEL = 3
    MANAGER = 4
    OWNER = 5
    
    @property
    def name_tr(self) -> str:
        """Türkçe seviye adı"""
        names = {
            UserLevel.FORBIDDEN: "Tehlikeli",
            UserLevel.UNKNOWN: "Bilinmiyor",
            UserLevel.GUEST: "Misafir",
            UserLevel.PERSONNEL: "Çalışan",
            UserLevel.MANAGER: "Şef",
            UserLevel.OWNER: "Patron"
        }
        return names.get(self, "Bilinmiyor")


class UserStatus(IntEnum):
    """Kullanıcı durumları"""
    ACTIVE = 1
    INACTIVE = 2
    BANNED = 3
    
    @property
    def name_tr(self) -> str:
        """Türkçe durum adı"""
        names = {
            UserStatus.ACTIVE: "Aktif",
            UserStatus.INACTIVE: "Pasif",
            UserStatus.BANNED: "Yasaklı"
        }
        return names.get(self, "Bilinmiyor")


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class User:
    """Kullanıcı profili"""
    name: str
    level: int
    user_id: Optional[str] = None
    face_file: Optional[str] = None
    voice_file: Optional[str] = None
    face_embedding: Optional[List[float]] = None
    bio_file: Optional[str] = None
    created_at: Optional[str] = None
    last_seen: Optional[str] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        # Validate level
        if not (0 <= self.level <= 5):
            raise ValueError(f"Geçersiz seviye: {self.level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Dictionary'den oluştur"""
        return cls(**data)
    
    def update_last_seen(self) -> None:
        """Son görülme zamanını güncelle"""
        self.last_seen = datetime.now().isoformat()


@dataclass
class AuditLogEntry:
    """Audit log kaydı"""
    timestamp: str
    action: str
    user_id: str
    details: str
    performed_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# USER MANAGER
# ═══════════════════════════════════════════════════════════════
class UserManager:
    """
    LotusAI Kullanıcı Yönetim Sistemi
    
    Sorumluluklar:
    - Kullanıcı profili yönetimi
    - RBAC (Role-Based Access Control)
    - Biyometrik veri referansları
    - JSON persistent storage
    - Auto-backup & migration
    - Audit logging
    
    Thread-safe design ile concurrent access desteklenir.
    """
    
    # Default users (ilk kurulum)
    DEFAULT_USERS = {
        "halil_sevim": {
            "name": "Halil Sevim",
            "level": UserLevel.OWNER,
            "face_file": "faces/halil_sevim.jpg",
            "bio_file": "halil_bio.txt",
            "voice_id": "voice_halil"
        },
        "hatice_sevim": {
            "name": "Hatice Sevim",
            "level": UserLevel.OWNER,
            "face_file": "faces/hatice_sevim.jpg",
            "voice_id": "voice_hatice"
        },
        "bengi_nisa_sevim": {
            "name": "Bengi Nisa Sevim",
            "level": UserLevel.GUEST,
            "face_file": "faces/bengi_nisa.jpg"
        },
        "eray_sef": {
            "name": "Eray (Şef)",
            "level": UserLevel.MANAGER,
            "face_file": "faces/eray.jpg"
        },
        "abdullah_usta": {
            "name": "Abdullah Usta",
            "level": UserLevel.PERSONNEL,
            "face_file": "faces/abdullah.jpg"
        },
        "muzeyyen_abla": {
            "name": "Müzeyyen Abla",
            "level": UserLevel.PERSONNEL,
            "face_file": "faces/muzeyyen.jpg"
        }
    }
    
    def __init__(self, access_level: str = "sandbox"):
        """
        User manager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full) - sadece bilgi amaçlı
        """
        self.access_level = access_level
        
        # Paths
        self.work_dir = Config.WORK_DIR
        self.db_file = self.work_dir / "lotus/users_db.json"
        self.backup_file = self.db_file.with_suffix(".json.backup")
        self.audit_log_file = self.work_dir / "users_audit.log"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Ensure directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU detection
        self.device = self._detect_hardware()
        
        # Load database
        self.users: Dict[str, Dict[str, Any]] = self._load_db()
        
        # Audit log
        self.audit_entries: List[AuditLogEntry] = []
        
        logger.info(f"✅ UserManager başlatıldı (Device: {self.device.upper()}, Erişim: {self.access_level})")
    
    def _detect_hardware(self) -> str:
        """
        GPU tespiti
        
        Returns:
            'cuda' veya 'cpu'
        """
        if Config.USE_GPU and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 GPU tespit edildi: {gpu_name}")
                return "cuda"
            except Exception:
                return "cuda"
        
        return "cpu"
    
    # ───────────────────────────────────────────────────────────
    # DATABASE MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def _load_db(self) -> Dict[str, Dict[str, Any]]:
        """
        Veritabanını yükle
        
        Returns:
            User dictionary
        """
        with self._lock:
            if not self.db_file.exists():
                return self._create_default_db()
            
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Migration check
                if self._check_migrations(data):
                    self._save_db_internal(data)
                
                return data
            
            except json.JSONDecodeError as e:
                logger.error(f"⚠️ Veritabanı bozuk: {e}, backup'tan geri yükleniyor")
                return self._restore_backup()
            
            except Exception as e:
                logger.error(f"❌ Kritik veritabanı hatası: {e}")
                return {}
    
    def _create_default_db(self) -> Dict[str, Dict[str, Any]]:
        """
        Varsayılan veritabanını oluştur
        
        Returns:
            User dictionary
        """
        now = datetime.now().isoformat()
        default_db = {}
        
        for user_id, user_data in self.DEFAULT_USERS.items():
            default_db[user_id] = {
                "name": user_data["name"],
                "level": user_data["level"],
                "face_file": user_data.get("face_file"),
                "voice_file": user_data.get("voice_id"),
                "face_embedding": None,
                "bio_file": user_data.get("bio_file"),
                "created_at": now,
                "last_seen": None,
                "status": "active"
            }
        
        self._save_db_internal(default_db)
        logger.info("✅ Varsayılan kullanıcı veritabanı oluşturuldu")
        
        return default_db
    
    def _restore_backup(self) -> Dict[str, Dict[str, Any]]:
        """
        Backup'tan geri yükle
        
        Returns:
            User dictionary
        """
        if self.backup_file.exists():
            try:
                shutil.copy(self.backup_file, self.db_file)
                logger.info("✅ Veritabanı backup'tan geri yüklendi")
                return self._load_db()
            except Exception as e:
                logger.error(f"❌ Backup geri yükleme hatası: {e}")
        
        return self._create_default_db()
    
    def _check_migrations(self, data: Dict[str, Dict[str, Any]]) -> bool:
        """
        Veri yapısı migrasyonları
        
        Args:
            data: User dictionary
        
        Returns:
            Değişiklik yapıldıysa True
        """
        changed = False
        
        default_fields = {
            "bio_file": None,
            "voice_file": None,
            "face_embedding": None,
            "created_at": datetime.now().isoformat(),
            "last_seen": None,
            "status": "active"
        }
        
        for user_id, user_data in data.items():
            for field, default_val in default_fields.items():
                if field not in user_data:
                    # Special case: halil_sevim bio
                    if field == "bio_file" and user_id == "halil_sevim":
                        user_data[field] = "halil_bio.txt"
                    else:
                        user_data[field] = default_val
                    
                    changed = True
        
        return changed
    
    def _save_db_internal(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Veritabanını kaydet
        
        Args:
            data: User dictionary
        """
        try:
            # Backup oluştur
            if self.db_file.exists():
                shutil.copy(self.db_file, self.backup_file)
            
            # Kaydet
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"❌ Veritabanı kaydetme hatası: {e}")
    
    @contextmanager
    def _db_transaction(self):
        """
        Database transaction context manager
        
        Usage:
            with user_manager._db_transaction():
                # Database operations
                pass
        """
        with self._lock:
            try:
                yield self.users
                self._save_db_internal(self.users)
            except Exception as e:
                logger.error(f"Transaction hatası: {e}")
                raise
    
    def save(self) -> None:
        """Manuel kaydetme"""
        with self._lock:
            self._save_db_internal(self.users)
    
    # ───────────────────────────────────────────────────────────
    # USER ID GENERATION
    # ───────────────────────────────────────────────────────────
    
    @staticmethod
    def _generate_user_id(name: str) -> str:
        """
        İsimden user ID oluştur
        
        Args:
            name: Kullanıcı adı
        
        Returns:
            Slugified user ID
        """
        name = name.lower().strip()
        
        # Türkçe karakter dönüşümü
        tr_map = str.maketrans("çğıöşü ", "cgiosu_")
        clean_name = name.translate(tr_map)
        
        # Geçersiz karakterleri kaldır
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        
        return clean_name
    
    # ───────────────────────────────────────────────────────────
    # QUERY METHODS
    # ───────────────────────────────────────────────────────────
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Kullanıcı profilini getir
        
        Args:
            user_id: Kullanıcı ID
        
        Returns:
            Kullanıcı verisi
        """
        with self._lock:
            return self.users.get(user_id, {}).copy()
    
    def get_user_level(self, user_id: str) -> int:
        """
        Kullanıcı seviyesini getir ve son görülme güncelle
        
        Args:
            user_id: Kullanıcı ID
        
        Returns:
            Yetki seviyesi
        """
        with self._lock:
            user = self.users.get(user_id)
            
            if user:
                user["last_seen"] = datetime.now().isoformat()
                return user.get("level", UserLevel.GUEST)
            
            return UserLevel.UNKNOWN
    
    def get_user_by_level(self, level: int) -> Optional[Dict[str, Any]]:
        """
        Seviyeye göre ilk kullanıcıyı bul
        
        Args:
            level: Yetki seviyesi
        
        Returns:
            Kullanıcı verisi veya None
        """
        with self._lock:
            for user in self.users.values():
                if user.get("level") == level:
                    return user.copy()
            
            return None
    
    def get_all_users_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        Seviyeye göre tüm kullanıcıları listele
        
        Args:
            level: Yetki seviyesi
        
        Returns:
            Kullanıcı listesi
        """
        with self._lock:
            return [
                user.copy()
                for user in self.users.values()
                if user.get("level") == level
            ]
    
    def search_users(
        self,
        query: str,
        field: str = "name"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Kullanıcı ara
        
        Args:
            query: Arama terimi
            field: Aranacak alan
        
        Returns:
            Eşleşen kullanıcılar [(user_id, user_data), ...]
        """
        with self._lock:
            query_lower = query.lower()
            results = []
            
            for user_id, user_data in self.users.items():
                value = str(user_data.get(field, "")).lower()
                
                if query_lower in value:
                    results.append((user_id, user_data.copy()))
            
            return results
    
    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Tüm kullanıcıları getir
        
        Returns:
            User dictionary (copy)
        """
        with self._lock:
            return self.users.copy()
    
    # ───────────────────────────────────────────────────────────
    # USER OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def create_new_user(
        self,
        name: str,
        level: Optional[int] = None,
        face_file: Optional[str] = None,
        voice_file: Optional[str] = None,
        bio_file: Optional[str] = None
    ) -> bool:
        """
        Yeni kullanıcı oluştur veya güncelle
        
        Args:
            name: Kullanıcı adı
            level: Yetki seviyesi
            face_file: Yüz dosya yolu
            voice_file: Ses dosya yolu
            bio_file: Bio dosya yolu
        
        Returns:
            Başarılı ise True
        """
        with self._db_transaction():
            user_id = self._generate_user_id(name)
            
            # Update existing user
            if user_id in self.users:
                user = self.users[user_id]
                
                if face_file:
                    user["face_file"] = face_file
                
                if voice_file:
                    user["voice_file"] = voice_file
                
                if bio_file:
                    user["bio_file"] = bio_file
                
                if level is not None:
                    user["level"] = level
                
                user["last_seen"] = datetime.now().isoformat()
                
                logger.info(f"🔄 Kullanıcı güncellendi: {name} ({user_id})")
                
                # Audit log
                self._add_audit_log(
                    action="UPDATE",
                    user_id=user_id,
                    details=f"User {name} updated"
                )
            
            # Create new user
            else:
                self.users[user_id] = {
                    "name": name.title(),
                    "level": level if level is not None else UserLevel.GUEST,
                    "face_file": face_file,
                    "voice_file": voice_file,
                    "bio_file": bio_file,
                    "face_embedding": None,
                    "created_at": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "status": "active"
                }
                
                logger.info(f"🆕 Yeni kullanıcı: {name} ({user_id})")
                
                # Audit log
                self._add_audit_log(
                    action="CREATE",
                    user_id=user_id,
                    details=f"User {name} created"
                )
            
            return True
    
    def delete_user(self, user_id: str) -> bool:
        """
        Kullanıcıyı sil
        
        Args:
            user_id: Kullanıcı ID
        
        Returns:
            Başarılı ise True
        """
        with self._db_transaction():
            if user_id in self.users:
                user_name = self.users[user_id].get("name", user_id)
                del self.users[user_id]
                
                logger.warning(f"🗑️ Kullanıcı silindi: {user_name} ({user_id})")
                
                # Audit log
                self._add_audit_log(
                    action="DELETE",
                    user_id=user_id,
                    details=f"User {user_name} deleted"
                )
                
                return True
            
            return False
    
    def update_user_status(
        self,
        user_id: str,
        status: str
    ) -> bool:
        """
        Kullanıcı durumunu güncelle
        
        Args:
            user_id: Kullanıcı ID
            status: Yeni durum (active/inactive/banned)
        
        Returns:
            Başarılı ise True
        """
        with self._db_transaction():
            if user_id in self.users:
                old_status = self.users[user_id].get("status", "active")
                self.users[user_id]["status"] = status
                
                logger.info(
                    f"📝 Durum değişti: {user_id} "
                    f"({old_status} → {status})"
                )
                
                # Audit log
                self._add_audit_log(
                    action="STATUS_UPDATE",
                    user_id=user_id,
                    details=f"Status changed: {old_status} → {status}"
                )
                
                return True
            
            return False
    
    # ───────────────────────────────────────────────────────────
    # AUDIT LOGGING
    # ───────────────────────────────────────────────────────────
    
    def _add_audit_log(
        self,
        action: str,
        user_id: str,
        details: str,
        performed_by: Optional[str] = None
    ) -> None:
        """
        Audit log ekle
        
        Args:
            action: Eylem tipi
            user_id: İlgili kullanıcı ID
            details: Detaylar
            performed_by: Eylemi yapan (opsiyonel)
        """
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            user_id=user_id,
            details=details,
            performed_by=performed_by
        )
        
        self.audit_entries.append(entry)
        
        # Log dosyasına yaz
        try:
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Audit log yazma hatası: {e}")
    
    def get_audit_logs(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> List[AuditLogEntry]:
        """
        Audit logları getir
        
        Args:
            limit: Maksimum kayıt sayısı
            user_id: Belirli kullanıcının logları
        
        Returns:
            Audit log listesi
        """
        logs = self.audit_entries
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    # ───────────────────────────────────────────────────────────
    # STATISTICS & METRICS
    # ───────────────────────────────────────────────────────────
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Kullanıcı istatistiklerini getir
        
        Returns:
            İstatistik dictionary'si
        """
        with self._lock:
            total_users = len(self.users)
            
            # Seviye dağılımı
            level_distribution = {}
            for level in UserLevel:
                count = sum(
                    1 for u in self.users.values()
                    if u.get("level") == level
                )
                level_distribution[level.name_tr] = count
            
            # Durum dağılımı
            status_distribution = {}
            for status_val in ["active", "inactive", "banned"]:
                count = sum(
                    1 for u in self.users.values()
                    if u.get("status") == status_val
                )
                status_distribution[status_val] = count
            
            # Biyometrik veriler
            faces_count = sum(
                1 for u in self.users.values()
                if u.get("face_file")
            )
            voices_count = sum(
                1 for u in self.users.values()
                if u.get("voice_file")
            )
            
            return {
                "total_users": total_users,
                "level_distribution": level_distribution,
                "status_distribution": status_distribution,
                "faces_registered": faces_count,
                "voices_registered": voices_count,
                "device": self.device,
                "access_level": self.access_level
            }
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """
        Donanım durumunu getir
        
        Returns:
            Hardware bilgileri
        """
        gpu_count = 0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        
        return {
            "device": self.device,
            "cuda_available": (self.device == "cuda"),
            "gpu_count": gpu_count
        }
    
    # ───────────────────────────────────────────────────────────
    # CLEANUP
    # ───────────────────────────────────────────────────────────
    
    def shutdown(self) -> None:
        """User manager'ı kapat"""
        logger.info("UserManager kapatılıyor...")
        
        with self._lock:
            # Son kaydet
            self._save_db_internal(self.users)
            
            # Audit logları temizle
            self.audit_entries.clear()
        
        logger.info("✅ UserManager kapatıldı")