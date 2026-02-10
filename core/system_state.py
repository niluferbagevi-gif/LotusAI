import threading
import logging
import collections
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    # Bağımsız çalışma durumu için Fallback
    class Config:
        WORK_DIR = os.getcwd()
        USE_GPU = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.SystemState")

class SystemState:
    """
    LotusAI Merkezi Durum, Bağlam ve Donanım Kaynak Yöneticisi.
    
    Bu sınıf şunları yönetir:
    1. Sistemin Anlık Modu ve Mod Geçişleri.
    2. Donanım (GPU/CPU) Durum Takibi ve Kaynak Paylaşımı (Config tabanlı).
    3. Global Sistem Değişkenleri (Aktif Ajan, Aktif Kullanıcı).
    4. Kaynak Kullanım Takibi (Resource Locking) - GPU Çakışma Önleyici.
    5. Thread-Safe veri erişimi ve Durum Değişikliği Bildirimleri.
    """
    
    # --- Durum Sabitleri (State Constants) ---
    INITIALIZING = -1 # Sistem Başlatılıyor
    IDLE = 0          # Boşta / Beklemede
    LISTENING = 1     # Mikrofon açık, kullanıcıyı dinliyor
    THINKING = 2      # Yapay zeka cevap üretiyor (GPU/LLM isteği aktif)
    SPEAKING = 3      # Sistem konuşuyor (TTS Aktif)
    PROCESSING = 4    # Arka plan görevi (Görüntü işleme, dosya analizi - GPU yoğun olabilir)
    ERROR = 5         # Sistem hata durumunda
    SHUTTING_DOWN = 6 # Sistem kapanıyor

    # Durum isimleri
    STATE_NAMES = {
        -1: "BAŞLATILIYOR",
        0: "BOŞTA",
        1: "DİNLİYOR",
        2: "DÜŞÜNÜYOR",
        3: "KONUŞUYOR",
        4: "İŞLEM YAPIYOR",
        5: "HATA",
        6: "KAPANIYOR"
    }

    # Maksimum bekleme süreleri (Saniye)
    STATE_TIMEOUTS = {
        2: 45.0, # THINKING max 45 sn
        3: 30.0, # SPEAKING max 30 sn
        1: 20.0  # LISTENING max 20 sn
    }

    def __init__(self):
        # Çoklu thread ve async erişimi için Reentrant Lock
        self.lock = threading.RLock()
        
        # Temel durum değişkenleri
        self._current_state = self.INITIALIZING
        self._last_state_change = datetime.now()
        self._is_running = True
        self._active_agent = "Lotus"
        self._active_user = {"name": "Bilinmiyor", "level": 0}
        
        # --- GPU ve Donanım Bilgileri ---
        # Doğrudan Config'den gelen kararı kullanıyoruz (Tek Gerçek Kaynak Prensibi)
        self._gpu_available = getattr(Config, "USE_GPU", False)
        self._gpu_load = 0.0  # % cinsinden tahmini yük
        self._vram_usage = 0.0 # MB cinsinden tahmini kullanım
        
        # Kaynak Takibi (Hangi ajan GPU'yu veya Kamerayı kullanıyor?)
        self._resource_locks = {} # {resource_name: agent_name}
        
        # Hata takibi
        self._last_error = None
        self._error_time = None
        
        # Durum Geçmişi ve Gözlemciler
        self._state_history = collections.deque(maxlen=20) 
        self._observers: List[Callable] = [] 
        
        logger.info(f"✅ Sistem Durum Yöneticisi Başlatıldı. GPU Desteği: {'AKTİF' if self._gpu_available else 'PASİF'}")

    def _check_gpu_support(self) -> bool:
        """
        Sistemde CUDA destekli bir GPU olup olmadığını kontrol eder.
        Artık doğrudan Config'e bakıyor, tekrar torch import edip sorgulama yapmıyor.
        """
        return self._gpu_available

    # --- DURUM YÖNETİMİ ---

    def set_state(self, state: int, reason: str = "Genel İşlem"):
        """Sistemin durumunu güvenli bir şekilde değiştirir ve bildirim yapar."""
        with self.lock:
            if state not in self.STATE_NAMES:
                logger.error(f"❌ Geçersiz Durum Tanımı: {state}")
                return

            if self._current_state != state:
                old_state = self._current_state
                self._current_state = state
                
                now = datetime.now()
                duration = (now - self._last_state_change).total_seconds()
                self._last_state_change = now
                
                old_name = self.STATE_NAMES.get(old_state, "Bilinmiyor")
                new_name = self.STATE_NAMES.get(state, "Bilinmiyor")
                
                # Geçmişe kayıt ekle
                self._state_history.append({
                    "from": old_name,
                    "to": new_name,
                    "reason": reason,
                    "time": now.strftime("%H:%M:%S"),
                    "stayed_duration": round(duration, 2)
                })
                
                logger.debug(f"🔄 Durum Değişti: {old_name} -> {new_name} (Neden: {reason})")
                self._notify_observers(state)

    def get_state(self) -> int:
        """Anlık durumu döndürür ve zaman aşımı kontrolü yapar."""
        with self.lock:
            if self._current_state in self.STATE_TIMEOUTS:
                duration = (datetime.now() - self._last_state_change).total_seconds()
                if duration > self.STATE_TIMEOUTS[self._current_state]:
                    logger.warning(f"⚠️ {self.STATE_NAMES[self._current_state]} modunda zaman aşımı! Reset atılıyor.")
                    self.set_state(self.IDLE, reason="Zaman Aşımı Koruması")
            
            return self._current_state

    def get_state_name(self) -> str:
        return self.STATE_NAMES.get(self.get_state(), "BİLİNMİYOR")

    # --- DONANIM VE KAYNAK YÖNETİMİ ---

    def get_hardware_status(self) -> Dict[str, Any]:
        """GPU ve sistem kaynaklarının anlık özetini döndürür."""
        with self.lock:
            status = {
                "gpu_active": self._gpu_available,
                "active_resources": list(self._resource_locks.keys()),
                "is_heavy_load": self._current_state in [self.THINKING, self.PROCESSING]
            }
            # Eğer GPU aktifse (Config onaylı), anlık VRAM bilgisini çekmeyi dene
            if self._gpu_available:
                try:
                    import torch
                    if torch.cuda.is_available():
                        status["gpu_name"] = torch.cuda.get_device_name(0)
                        status["vram_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                except Exception:
                    # Torch hatası olursa sessizce geç, sistemi bozma
                    status["gpu_error"] = "Veri alınamadı"
            return status

    def lock_resource(self, resource_name: str, agent_name: str) -> bool:
        """
        Bir kaynağı (GPU, kamera, mic vb.) bir ajana tahsis eder.
        Özellikle 'gpu_vram' veya 'gpu_compute' gibi kaynakları yönetmek için kullanılır.
        """
        with self.lock:
            if resource_name in self._resource_locks:
                logger.warning(f"🚫 Kaynak Çakışması: {resource_name} zaten {self._resource_locks[resource_name]} kullanımında.")
                return False
            
            self._resource_locks[resource_name] = agent_name
            logger.debug(f"🔒 Kaynak Kilitlendi: {resource_name} -> {agent_name}")
            return True

    def unlock_resource(self, resource_name: str):
        with self.lock:
            if resource_name in self._resource_locks:
                agent = self._resource_locks.pop(resource_name)
                logger.debug(f"🔓 Kaynak Serbest: {resource_name} (Eski sahibi: {agent})")

    # --- GÖZLEMCİ (OBSERVER) SİSTEMİ ---

    def register_observer(self, callback_func: Callable):
        with self.lock:
            if callback_func not in self._observers:
                self._observers.append(callback_func)

    def _notify_observers(self, new_state: int):
        for callback in self._observers:
            try:
                t = threading.Thread(target=callback, args=(new_state,), daemon=True)
                t.start()
            except Exception as e:
                logger.error(f"❌ State Observer Hatası: {e}")

    # --- GLOBAL VERİ YÖNETİMİ ---

    def set_active_agent(self, agent_name: str):
        with self.lock:
            self._active_agent = agent_name
            logger.info(f"🤖 Aktif Ajan: {agent_name}")

    def get_active_agent(self) -> str:
        with self.lock:
            return self._active_agent

    def set_active_user(self, user_obj: Dict):
        with self.lock:
            if user_obj:
                self._active_user = user_obj
                logger.info(f"👤 Kullanıcı Bağlandı: {user_obj.get('name', 'Misafir')}")

    def get_active_user(self) -> Dict:
        with self.lock:
            return self._active_user

    def set_error(self, error_msg: str):
        with self.lock:
            self._last_error = error_msg
            self._error_time = datetime.now()
            self.set_state(self.ERROR, reason=f"Kritik Hata: {error_msg}")

    def clear_error(self):
        with self.lock:
            self._last_error = None
            self.set_state(self.IDLE, reason="Hata Durumu Giderildi")

    # --- SİSTEM KONTROLLERİ ---

    def stop_system(self):
        with self.lock:
            self.set_state(self.SHUTTING_DOWN, reason="Sistem Kapatılıyor")
            self._is_running = False

    def is_running(self) -> bool:
        with self.lock:
            return self._is_running

    def should_listen(self) -> bool:
        """Sistemin ses girişine uygun olup olmadığını kontrol eder."""
        with self.lock:
            # Eğer GPU üzerinde ağır bir LLM işlemi (THINKING) varsa dinleme gecikebilir
            return self._current_state in [self.IDLE, self.LISTENING]

    def get_history(self) -> List[Dict]:
        with self.lock:
            return list(self._state_history)

    def __str__(self) -> str:
        with self.lock:
            status = self.get_state_name()
            gpu = "GPU OK" if self._gpu_available else "CPU ONLY"
            user = self._active_user.get("name", "Bilinmiyor")
            duration = int((datetime.now() - self._last_state_change).total_seconds())
            return f"LotusState[{status} ({duration}s) | {gpu} | Ajan: {self._active_agent} | Kullanıcı: {user}]"