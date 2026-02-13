"""
LotusAI System State Manager
Sürüm: 2.5.3
Açıklama: Merkezi durum yönetimi, kaynak koordinasyonu ve FSM

Özellikler:
- Finite State Machine (FSM)
- Thread-safe operations
- Observer pattern
- Resource locking
- State history
- Timeout protection
- Hardware monitoring
"""

import threading
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import IntEnum, auto
from contextlib import contextmanager

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config

logger = logging.getLogger("LotusAI.SystemState")


# ═══════════════════════════════════════════════════════════════
# STATE ENUMERATION
# ═══════════════════════════════════════════════════════════════
class SystemState(IntEnum):
    """
    Sistem durumları
    
    State Transition Flow:
    INITIALIZING → IDLE ⟷ LISTENING → THINKING → SPEAKING → IDLE
                     ↓
                  PROCESSING → IDLE
                     ↓
                   ERROR → IDLE
                     ↓
                SHUTTING_DOWN
    """
    INITIALIZING = -1  # Sistem başlatılıyor
    IDLE = 0           # Boşta
    LISTENING = 1      # Mikrofon aktif
    THINKING = 2       # LLM processing
    SPEAKING = 3       # TTS çalıyor
    PROCESSING = 4     # Arka plan işlemi
    ERROR = 5          # Hata durumu
    SHUTTING_DOWN = 6  # Kapanıyor
    
    @property
    def name_tr(self) -> str:
        """Türkçe durum adı"""
        names = {
            SystemState.INITIALIZING: "BAŞLATILIYOR",
            SystemState.IDLE: "BOŞTA",
            SystemState.LISTENING: "DİNLİYOR",
            SystemState.THINKING: "DÜŞÜNÜYOR",
            SystemState.SPEAKING: "KONUŞUYOR",
            SystemState.PROCESSING: "İŞLEM YAPIYOR",
            SystemState.ERROR: "HATA",
            SystemState.SHUTTING_DOWN: "KAPANIYOR"
        }
        return names.get(self, "BİLİNMİYOR")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
class StateConfig:
    """State manager konfigürasyonu"""
    # Timeout'lar (saniye)
    TIMEOUT_THINKING = 45.0
    TIMEOUT_SPEAKING = 30.0
    TIMEOUT_LISTENING = 20.0
    TIMEOUT_PROCESSING = 60.0
    
    # History
    MAX_HISTORY_SIZE = 50
    
    # Resource locks
    RESOURCE_GPU_COMPUTE = "gpu_compute"
    RESOURCE_GPU_VRAM = "gpu_vram"
    RESOURCE_CAMERA = "camera"
    RESOURCE_MICROPHONE = "microphone"
    RESOURCE_SPEAKER = "speaker"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class StateTransition:
    """Durum geçişi kaydı"""
    from_state: SystemState
    to_state: SystemState
    reason: str
    timestamp: datetime
    duration_seconds: float
    
    def __str__(self) -> str:
        return (
            f"{self.from_state.name_tr} → {self.to_state.name_tr} "
            f"({self.duration_seconds:.2f}s): {self.reason}"
        )


@dataclass
class UserInfo:
    """Kullanıcı bilgileri"""
    name: str
    level: int = 0
    authenticated: bool = False
    last_seen: Optional[datetime] = None


@dataclass
class HardwareStatus:
    """Donanım durumu"""
    gpu_available: bool
    gpu_name: Optional[str] = None
    vram_allocated_mb: float = 0.0
    vram_total_mb: float = 0.0
    active_resources: List[str] = field(default_factory=list)
    is_heavy_load: bool = False


# ═══════════════════════════════════════════════════════════════
# STATE MANAGER
# ═══════════════════════════════════════════════════════════════
class SystemStateManager:
    """
    LotusAI Merkezi Durum Yöneticisi
    
    Sorumluluklar:
    - Sistem durumu yönetimi (FSM)
    - Kaynak koordinasyonu (GPU, Camera, etc.)
    - Durum geçiş validasyonu
    - Observer pattern implementasyonu
    - Timeout koruması
    - Hardware monitoring
    
    Thread-safe design ile concurrent access desteklenir.
    """
    
    # Geçerli durum geçişleri (FSM kuralları)
    VALID_TRANSITIONS = {
        SystemState.INITIALIZING: {SystemState.IDLE, SystemState.ERROR},
        SystemState.IDLE: {
            SystemState.LISTENING,
            SystemState.PROCESSING,
            SystemState.ERROR,
            SystemState.SHUTTING_DOWN
        },
        SystemState.LISTENING: {
            SystemState.IDLE,
            SystemState.THINKING,
            SystemState.ERROR
        },
        SystemState.THINKING: {
            SystemState.SPEAKING,
            SystemState.IDLE,
            SystemState.ERROR
        },
        SystemState.SPEAKING: {
            SystemState.IDLE,
            SystemState.ERROR
        },
        SystemState.PROCESSING: {
            SystemState.IDLE,
            SystemState.ERROR
        },
        SystemState.ERROR: {
            SystemState.IDLE,
            SystemState.SHUTTING_DOWN
        },
        SystemState.SHUTTING_DOWN: set()  # Terminal state
    }
    
    # State timeout'ları
    STATE_TIMEOUTS = {
        SystemState.THINKING: StateConfig.TIMEOUT_THINKING,
        SystemState.SPEAKING: StateConfig.TIMEOUT_SPEAKING,
        SystemState.LISTENING: StateConfig.TIMEOUT_LISTENING,
        SystemState.PROCESSING: StateConfig.TIMEOUT_PROCESSING
    }
    
    def __init__(self):
        """State manager başlatıcı"""
        # Thread safety
        self.lock = threading.RLock()
        
        # Current state
        self._current_state = SystemState.INITIALIZING
        self._last_state_change = datetime.now()
        self._is_running = True
        
        # Agents & Users
        self._active_agent = "LOTUS"
        self._active_user = UserInfo(name="Bilinmiyor", level=0)
        
        # Hardware
        self._gpu_available = Config.USE_GPU
        
        # Resources
        self._resource_locks: Dict[str, str] = {}  # {resource: agent}
        
        # Error tracking
        self._last_error: Optional[str] = None
        self._error_time: Optional[datetime] = None
        
        # History
        self._state_history: deque = deque(maxlen=StateConfig.MAX_HISTORY_SIZE)
        
        # Observers
        self._observers: List[Callable[[SystemState], None]] = []
        
        # Metrics
        self._state_counts: Dict[SystemState, int] = {}
        self._total_transitions = 0
        
        logger.info(
            f"✅ SystemStateManager başlatıldı "
            f"(GPU: {'Aktif' if self._gpu_available else 'Pasif'})"
        )
    
    # ───────────────────────────────────────────────────────────
    # STATE MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def set_state(
        self,
        new_state: SystemState,
        reason: str = "Genel İşlem",
        force: bool = False
    ) -> bool:
        """
        Sistem durumunu değiştir
        
        Args:
            new_state: Yeni durum
            reason: Değişim sebebi
            force: FSM kurallarını atla
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            current = self._current_state
            
            # Same state check
            if current == new_state:
                logger.debug(f"Durum zaten {new_state.name_tr}")
                return True
            
            # Validate transition
            if not force and not self._is_valid_transition(current, new_state):
                logger.warning(
                    f"⚠️ Geçersiz durum geçişi: {current.name_tr} → {new_state.name_tr}"
                )
                return False
            
            # Calculate duration
            now = datetime.now()
            duration = (now - self._last_state_change).total_seconds()
            
            # Create transition record
            transition = StateTransition(
                from_state=current,
                to_state=new_state,
                reason=reason,
                timestamp=now,
                duration_seconds=duration
            )
            
            # Update state
            self._current_state = new_state
            self._last_state_change = now
            
            # Update metrics
            self._state_counts[new_state] = self._state_counts.get(new_state, 0) + 1
            self._total_transitions += 1
            
            # Add to history
            self._state_history.append(transition)
            
            # Log
            logger.info(f"🔄 {transition}")
            
            # Notify observers
            self._notify_observers(new_state)
            
            return True
    
    def _is_valid_transition(
        self,
        from_state: SystemState,
        to_state: SystemState
    ) -> bool:
        """
        Durum geçişinin geçerli olup olmadığını kontrol et
        
        Args:
            from_state: Mevcut durum
            to_state: Hedef durum
        
        Returns:
            Geçerli ise True
        """
        valid_next_states = self.VALID_TRANSITIONS.get(from_state, set())
        return to_state in valid_next_states
    
    def get_state(self) -> SystemState:
        """
        Mevcut durumu döndür (timeout kontrolü ile)
        
        Returns:
            Mevcut durum
        """
        with self.lock:
            current = self._current_state
            
            # Timeout check
            if current in self.STATE_TIMEOUTS:
                duration = (datetime.now() - self._last_state_change).total_seconds()
                timeout = self.STATE_TIMEOUTS[current]
                
                if duration > timeout:
                    logger.warning(
                        f"⚠️ {current.name_tr} modunda zaman aşımı "
                        f"({duration:.1f}s > {timeout}s)"
                    )
                    self.set_state(
                        SystemState.IDLE,
                        reason="Zaman Aşımı Koruması",
                        force=True
                    )
                    return SystemState.IDLE
            
            return current
    
    def get_state_name(self) -> str:
        """Mevcut durum adını döndür"""
        return self.get_state().name_tr
    
    def get_state_duration(self) -> float:
        """Mevcut durumda kalınan süre (saniye)"""
        with self.lock:
            return (datetime.now() - self._last_state_change).total_seconds()
    
    # ───────────────────────────────────────────────────────────
    # RESOURCE MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def lock_resource(self, resource_name: str, agent_name: str) -> bool:
        """
        Bir kaynağı kilitle
        
        Args:
            resource_name: Kaynak adı
            agent_name: Kilit isteyen agent
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            if resource_name in self._resource_locks:
                current_owner = self._resource_locks[resource_name]
                logger.warning(
                    f"🚫 Kaynak çakışması: {resource_name} "
                    f"zaten {current_owner} kullanımında"
                )
                return False
            
            self._resource_locks[resource_name] = agent_name
            logger.debug(f"🔒 Kaynak kilitlendi: {resource_name} → {agent_name}")
            return True
    
    def unlock_resource(self, resource_name: str) -> bool:
        """
        Kaynağı serbest bırak
        
        Args:
            resource_name: Kaynak adı
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            if resource_name in self._resource_locks:
                agent = self._resource_locks.pop(resource_name)
                logger.debug(f"🔓 Kaynak serbest: {resource_name} (Eski: {agent})")
                return True
            
            logger.warning(f"⚠️ Kilit yok: {resource_name}")
            return False
    
    def is_resource_locked(self, resource_name: str) -> bool:
        """Kaynak kilitli mi kontrol et"""
        with self.lock:
            return resource_name in self._resource_locks
    
    def get_resource_owner(self, resource_name: str) -> Optional[str]:
        """Kaynak sahibini döndür"""
        with self.lock:
            return self._resource_locks.get(resource_name)
    
    @contextmanager
    def acquire_resource(self, resource_name: str, agent_name: str):
        """
        Context manager ile kaynak kilitleme
        
        Usage:
            with state_manager.acquire_resource("gpu_compute", "ATLAS"):
                # GPU işlemleri
                pass
        """
        try:
            if not self.lock_resource(resource_name, agent_name):
                raise RuntimeError(f"Kaynak kilitlenemedi: {resource_name}")
            yield
        finally:
            self.unlock_resource(resource_name)
    
    def unlock_all_resources(self, agent_name: Optional[str] = None) -> int:
        """
        Tüm kaynakları veya belirli bir agent'ın kaynaklarını serbest bırak
        
        Args:
            agent_name: Sadece bu agent'ın kaynakları (None ise hepsi)
        
        Returns:
            Serbest bırakılan kaynak sayısı
        """
        with self.lock:
            if agent_name is None:
                count = len(self._resource_locks)
                self._resource_locks.clear()
                logger.info(f"🔓 Tüm kaynaklar serbest bırakıldı ({count} adet)")
                return count
            
            to_remove = [
                res for res, owner in self._resource_locks.items()
                if owner == agent_name
            ]
            
            for res in to_remove:
                self._resource_locks.pop(res)
            
            if to_remove:
                logger.info(
                    f"🔓 {agent_name}'ın {len(to_remove)} kaynağı serbest bırakıldı"
                )
            
            return len(to_remove)
    
    # ───────────────────────────────────────────────────────────
    # HARDWARE MONITORING
    # ───────────────────────────────────────────────────────────
    
    def get_hardware_status(self) -> HardwareStatus:
        """
        Donanım durumunu döndür
        
        Returns:
            HardwareStatus objesi
        """
        with self.lock:
            status = HardwareStatus(
                gpu_available=self._gpu_available,
                active_resources=list(self._resource_locks.keys()),
                is_heavy_load=self._current_state in {
                    SystemState.THINKING,
                    SystemState.PROCESSING
                }
            )
            
            # GPU detayları
            if self._gpu_available:
                try:
                    import torch
                    if torch.cuda.is_available():
                        status.gpu_name = torch.cuda.get_device_name(0)
                        status.vram_allocated_mb = (
                            torch.cuda.memory_allocated(0) / (1024 ** 2)
                        )
                        status.vram_total_mb = (
                            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                        )
                except Exception as e:
                    logger.debug(f"GPU bilgi alınamadı: {e}")
            
            return status
    
    # ───────────────────────────────────────────────────────────
    # OBSERVER PATTERN
    # ───────────────────────────────────────────────────────────
    
    def register_observer(self, callback: Callable[[SystemState], None]) -> None:
        """
        Observer kaydet
        
        Args:
            callback: State değiştiğinde çağrılacak fonksiyon
        """
        with self.lock:
            if callback not in self._observers:
                self._observers.append(callback)
                logger.debug(f"Observer kaydedildi: {callback.__name__}")
    
    def unregister_observer(self, callback: Callable[[SystemState], None]) -> bool:
        """
        Observer'ı kaldır
        
        Args:
            callback: Kaldırılacak fonksiyon
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            if callback in self._observers:
                self._observers.remove(callback)
                logger.debug(f"Observer kaldırıldı: {callback.__name__}")
                return True
            return False
    
    def _notify_observers(self, new_state: SystemState) -> None:
        """Observer'ları bilgilendir"""
        for callback in self._observers:
            try:
                # Thread'de çalıştır (async notification)
                thread = threading.Thread(
                    target=self._safe_observer_call,
                    args=(callback, new_state),
                    daemon=True
                )
                thread.start()
            except Exception as e:
                logger.error(f"Observer thread hatası: {e}")
    
    def _safe_observer_call(
        self,
        callback: Callable,
        state: SystemState
    ) -> None:
        """Observer'ı güvenli şekilde çağır"""
        try:
            callback(state)
        except Exception as e:
            logger.error(f"Observer callback hatası ({callback.__name__}): {e}")
    
    # ───────────────────────────────────────────────────────────
    # AGENT & USER MANAGEMENT
    # ───────────────────────────────────────────────────────────
    
    def set_active_agent(self, agent_name: str) -> None:
        """Aktif agent'ı ayarla"""
        with self.lock:
            self._active_agent = agent_name
            logger.info(f"🤖 Aktif agent: {agent_name}")
    
    def get_active_agent(self) -> str:
        """Aktif agent'ı döndür"""
        with self.lock:
            return self._active_agent
    
    def set_active_user(self, user_info: Dict[str, Any]) -> None:
        """Aktif kullanıcıyı ayarla"""
        with self.lock:
            self._active_user = UserInfo(
                name=user_info.get("name", "Bilinmiyor"),
                level=user_info.get("level", 0),
                authenticated=user_info.get("authenticated", False),
                last_seen=datetime.now()
            )
            logger.info(f"👤 Kullanıcı: {self._active_user.name}")
    
    def get_active_user(self) -> UserInfo:
        """Aktif kullanıcıyı döndür"""
        with self.lock:
            return self._active_user
    
    # ───────────────────────────────────────────────────────────
    # ERROR HANDLING
    # ───────────────────────────────────────────────────────────
    
    def set_error(self, error_msg: str) -> None:
        """Hata durumu ayarla"""
        with self.lock:
            self._last_error = error_msg
            self._error_time = datetime.now()
            self.set_state(
                SystemState.ERROR,
                reason=f"Hata: {error_msg[:50]}",
                force=True
            )
            logger.error(f"❌ Sistem hatası: {error_msg}")
    
    def clear_error(self) -> None:
        """Hata durumunu temizle"""
        with self.lock:
            if self._last_error:
                logger.info("✅ Hata durumu temizlendi")
                self._last_error = None
                self._error_time = None
                self.set_state(SystemState.IDLE, reason="Hata giderildi")
    
    def get_last_error(self) -> Optional[Tuple[str, datetime]]:
        """Son hatayı döndür"""
        with self.lock:
            if self._last_error and self._error_time:
                return (self._last_error, self._error_time)
            return None
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM CONTROL
    # ───────────────────────────────────────────────────────────
    
    def stop_system(self) -> None:
        """Sistemi durdur"""
        with self.lock:
            self.set_state(
                SystemState.SHUTTING_DOWN,
                reason="Sistem kapatılıyor",
                force=True
            )
            self._is_running = False
            logger.info("🛑 Sistem durdurma sinyali verildi")
    
    def is_running(self) -> bool:
        """Sistem çalışıyor mu"""
        with self.lock:
            return self._is_running
    
    def should_listen(self) -> bool:
        """Ses girişi için uygun mu"""
        with self.lock:
            return self._current_state in {SystemState.IDLE, SystemState.LISTENING}
    
    # ───────────────────────────────────────────────────────────
    # HISTORY & METRICS
    # ───────────────────────────────────────────────────────────
    
    def get_history(self, limit: Optional[int] = None) -> List[StateTransition]:
        """
        Durum geçmişini döndür
        
        Args:
            limit: Maksimum kayıt sayısı
        
        Returns:
            StateTransition listesi
        """
        with self.lock:
            history = list(self._state_history)
            if limit:
                history = history[-limit:]
            return history
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Sistem metriklerini döndür
        
        Returns:
            Metrik dictionary'si
        """
        with self.lock:
            return {
                "current_state": self._current_state.name_tr,
                "state_duration": self.get_state_duration(),
                "total_transitions": self._total_transitions,
                "state_counts": {
                    state.name_tr: count
                    for state, count in self._state_counts.items()
                },
                "active_agent": self._active_agent,
                "active_user": self._active_user.name,
                "locked_resources": len(self._resource_locks),
                "gpu_available": self._gpu_available,
                "is_running": self._is_running
            }
    
    # ───────────────────────────────────────────────────────────
    # CLEANUP
    # ───────────────────────────────────────────────────────────
    
    def shutdown(self) -> None:
        """State manager'ı kapat"""
        logger.info("SystemStateManager kapatılıyor...")
        
        with self.lock:
            # Kaynakları serbest bırak
            count = self.unlock_all_resources()
            if count > 0:
                logger.info(f"✓ {count} kaynak serbest bırakıldı")
            
            # Observer'ları temizle
            self._observers.clear()
            
            # State'i kapat
            if self._current_state != SystemState.SHUTTING_DOWN:
                self.set_state(
                    SystemState.SHUTTING_DOWN,
                    reason="Shutdown",
                    force=True
                )
        
        logger.info("✅ SystemStateManager kapatıldı")
    
    # ───────────────────────────────────────────────────────────
    # STRING REPRESENTATION
    # ───────────────────────────────────────────────────────────
    
    def __str__(self) -> str:
        """String representation"""
        with self.lock:
            gpu_str = "GPU" if self._gpu_available else "CPU"
            duration = int(self.get_state_duration())
            
            return (
                f"LotusState[{self._current_state.name_tr} ({duration}s) | "
                f"{gpu_str} | Agent: {self._active_agent} | "
                f"User: {self._active_user.name}]"
            )
    
    def __repr__(self) -> str:
        return self.__str__()