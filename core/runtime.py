"""
LotusAI Runtime Context
Sürüm: 2.5.4 (Fix: Metaclass Property Support & Silent Init)
Açıklama: Global runtime state yönetimi (Thread-safe singleton pattern)
"""

import queue
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional, Any
from contextlib import suppress

logger = logging.getLogger("LotusAI.Runtime")

# ═══════════════════════════════════════════════════════════════
# METACLASS (Sınıf Seviyesi Özellik Yönetimi)
# ═══════════════════════════════════════════════════════════════
class RuntimeMeta(type):
    """
    RuntimeContext için sınıf seviyesinde property (özellik) erişimi sağlayan yönetici sınıf.
    Bu yapı sayesinde 'RuntimeContext.engine' gibi çağrılar hata vermeden çalışır.
    """
    
    @property
    def msg_queue(cls):
        return cls.get_msg_queue()

    @property
    def messaging_manager(cls):
        return cls.get_messaging_manager()
    
    @messaging_manager.setter
    def messaging_manager(cls, value):
        cls.set_messaging_manager(value)

    @property
    def engine(cls):
        return cls.get_engine()
    
    @engine.setter
    def engine(cls, value):
        cls.set_engine(value)

    @property
    def loop(cls):
        return cls.get_loop()
    
    @loop.setter
    def loop(cls, value):
        cls.set_loop(value)

    @property
    def security_instance(cls):
        return cls.get_security_instance()
    
    @security_instance.setter
    def security_instance(cls, value):
        cls.set_security_instance(value)

    @property
    def state_manager(cls):
        return cls.get_state_manager()
    
    @state_manager.setter
    def state_manager(cls, value):
        cls.set_state_manager(value)

    @property
    def active_web_agent(cls):
        return cls.get_active_web_agent()
    
    @active_web_agent.setter
    def active_web_agent(cls, value):
        cls.set_active_web_agent(value)

    @property
    def voice_mode_active(cls):
        return cls.is_voice_mode_active()
    
    @voice_mode_active.setter
    def voice_mode_active(cls, value):
        cls.set_voice_mode(value)

    @property
    def executor(cls):
        return cls.get_executor()


class RuntimeContext(metaclass=RuntimeMeta):
    """
    LotusAI Global Runtime Context Manager
    
    Singleton pattern ile tüm sistem bileşenlerine erişim sağlar.
    Thread-safe yapıda tasarlanmıştır.
    
    Özellikler:
    - Mesaj kuyruğu (msg_queue)
    - Manager referansları
    - Agent engine
    - Asyncio loop
    - Web durumu
    - Thread pool yönetimi
    """
    
    # ───────────────────────────────────────────────────────────
    # SINGLETON PATTERN
    # ───────────────────────────────────────────────────────────
    _instance: Optional['RuntimeContext'] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    # ───────────────────────────────────────────────────────────
    # PRIVATE DEĞIŞKENLER (Thread-safe)
    # ───────────────────────────────────────────────────────────
    _msg_queue: queue.Queue = queue.Queue(maxsize=100)
    _messaging_manager: Optional[Any] = None
    _engine: Optional[Any] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _security_instance: Optional[Any] = None
    _state_manager: Optional[Any] = None
    
    # Web/Voice durumları
    _active_web_agent: str = "ATLAS"
    _voice_mode_active: bool = False
    
    # Thread Pool
    _executor: Optional[ThreadPoolExecutor] = None
    _executor_max_workers: int = 5
    
    def __new__(cls):
        """Singleton instance oluştur"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, max_workers: int = 5) -> None:
        """
        Runtime context'i başlat
        
        Args:
            max_workers: Thread pool boyutu
        """
        with cls._lock:
            if cls._initialized:
                # GÜNCELLEME: Warning yerine Debug kullanıldı (Log kirliliğini önlemek için)
                logger.debug("RuntimeContext zaten başlatılmış")
                return
            
            cls._executor_max_workers = max_workers
            cls._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="LotusWorker"
            )
            cls._initialized = True
            logger.info(f"✅ RuntimeContext başlatıldı (workers: {max_workers})")
    
    # ───────────────────────────────────────────────────────────
    # MESAJ KUYRUĞU
    # ───────────────────────────────────────────────────────────
    @classmethod
    def get_msg_queue(cls) -> queue.Queue:
        """Mesaj kuyruğunu döndür"""
        return cls._msg_queue
    
    @classmethod
    def put_message(cls, message: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Mesaj kuyruğuna ekle
        
        Args:
            message: Eklenecek mesaj
            block: Kuyruk doluysa bekle
            timeout: Maksimum bekleme süresi
        
        Returns:
            Başarılı ise True
        """
        try:
            cls._msg_queue.put(message, block=block, timeout=timeout)
            return True
        except queue.Full:
            logger.warning("Mesaj kuyruğu dolu, mesaj atıldı")
            return False
    
    @classmethod
    def get_message(cls, block: bool = False, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Mesaj kuyruğundan al
        
        Args:
            block: Kuyruk boşsa bekle
            timeout: Maksimum bekleme süresi
        
        Returns:
            Mesaj veya None
        """
        try:
            return cls._msg_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    # ───────────────────────────────────────────────────────────
    # MESSAGING MANAGER
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_messaging_manager(cls, manager: Any) -> None:
        """Messaging manager'ı ayarla"""
        with cls._lock:
            cls._messaging_manager = manager
            logger.debug("Messaging manager ayarlandı")
    
    @classmethod
    def get_messaging_manager(cls) -> Optional[Any]:
        """Messaging manager'ı döndür"""
        return cls._messaging_manager
    
    # ───────────────────────────────────────────────────────────
    # AGENT ENGINE
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_engine(cls, engine: Any) -> None:
        """Agent engine'i ayarla"""
        with cls._lock:
            cls._engine = engine
            logger.debug("Agent engine ayarlandı")
    
    @classmethod
    def get_engine(cls) -> Optional[Any]:
        """Agent engine'i döndür"""
        if cls._engine is None:
            logger.warning("Agent engine henüz ayarlanmadı")
        return cls._engine
    
    # ───────────────────────────────────────────────────────────
    # ASYNCIO LOOP
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_loop(cls, loop: asyncio.AbstractEventLoop) -> None:
        """Event loop'u ayarla"""
        with cls._lock:
            cls._loop = loop
            logger.debug("Event loop ayarlandı")
    
    @classmethod
    def get_loop(cls) -> Optional[asyncio.AbstractEventLoop]:
        """Event loop'u döndür"""
        if cls._loop is None:
            logger.warning("Event loop henüz ayarlanmadı")
        return cls._loop
    
    # ───────────────────────────────────────────────────────────
    # SECURITY INSTANCE
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_security_instance(cls, security: Any) -> None:
        """Security manager'ı ayarla"""
        with cls._lock:
            cls._security_instance = security
            logger.debug("Security manager ayarlandı")
    
    @classmethod
    def get_security_instance(cls) -> Optional[Any]:
        """Security manager'ı döndür"""
        return cls._security_instance
    
    # ───────────────────────────────────────────────────────────
    # STATE MANAGER
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_state_manager(cls, state_manager: Any) -> None:
        """State manager'ı ayarla"""
        with cls._lock:
            cls._state_manager = state_manager
            logger.debug("State manager ayarlandı")
    
    @classmethod
    def get_state_manager(cls) -> Optional[Any]:
        """State manager'ı döndür"""
        return cls._state_manager
    
    # ───────────────────────────────────────────────────────────
    # WEB AGENT
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_active_web_agent(cls, agent_name: str) -> None:
        """Aktif web agent'ı ayarla"""
        with cls._lock:
            cls._active_web_agent = agent_name.upper()
            logger.debug(f"Aktif web agent: {cls._active_web_agent}")
    
    @classmethod
    def get_active_web_agent(cls) -> str:
        """Aktif web agent'ı döndür"""
        return cls._active_web_agent
    
    # ───────────────────────────────────────────────────────────
    # VOICE MODE
    # ───────────────────────────────────────────────────────────
    @classmethod
    def set_voice_mode(cls, active: bool) -> None:
        """Ses modunu ayarla"""
        with cls._lock:
            cls._voice_mode_active = active
            logger.info(f"Ses modu: {'Aktif' if active else 'Pasif'}")
    
    @classmethod
    def get_voice_mode(cls) -> bool:
        """Ses modu durumunu döndür"""
        return cls._voice_mode_active
    
    @classmethod
    def is_voice_mode_active(cls) -> bool:
        """Ses modu aktif mi kontrol et"""
        return cls._voice_mode_active
    
    # ───────────────────────────────────────────────────────────
    # THREAD POOL EXECUTOR
    # ───────────────────────────────────────────────────────────
    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """
        Thread pool executor'ı döndür
        
        Returns:
            ThreadPoolExecutor instance
        """
        if cls._executor is None:
            cls.initialize()
        return cls._executor
    
    @classmethod
    def submit_task(cls, func: callable, *args, **kwargs) -> Any:
        """
        Thread pool'a görev ekle
        
        Args:
            func: Çalıştırılacak fonksiyon
            *args: Pozisyonel argümanlar
            **kwargs: İsimli argümanlar
        
        Returns:
            Future nesnesi
        """
        if cls._executor is None:
            cls.initialize()
        
        try:
            return cls._executor.submit(func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Görev eklenemedi: {e}")
            raise
    
    # ───────────────────────────────────────────────────────────
    # CLEANUP / SHUTDOWN
    # ───────────────────────────────────────────────────────────
    @classmethod
    def shutdown(cls, wait: bool = True, timeout: Optional[float] = 5.0) -> None:
        """
        RuntimeContext'i temizle ve kapat
        
        Args:
            wait: Thread'lerin bitmesini bekle
            timeout: Maksimum bekleme süresi
        """
        logger.info("RuntimeContext kapatılıyor...")
        
        with cls._lock:
            # Thread pool'u kapat
            if cls._executor is not None:
                with suppress(Exception):
                    cls._executor.shutdown(wait=wait)
                    logger.info("✓ Thread pool kapatıldı")
                cls._executor = None
            
            # Değişkenleri sıfırla
            cls._messaging_manager = None
            cls._engine = None
            cls._loop = None
            cls._security_instance = None
            cls._state_manager = None
            
            # Kuyruğu temizle
            with suppress(Exception):
                while not cls._msg_queue.empty():
                    cls._msg_queue.get_nowait()
            
            cls._initialized = False
            logger.info("✅ RuntimeContext temizlendi")
    
    # ───────────────────────────────────────────────────────────
    # DEBUG / BİLGİ METOTLARI
    # ───────────────────────────────────────────────────────────
    @classmethod
    def get_status(cls) -> dict:
        """
        RuntimeContext durumunu döndür
        
        Returns:
            Durum bilgileri
        """
        return {
            "initialized": cls._initialized,
            "messaging_manager": cls._messaging_manager is not None,
            "engine": cls._engine is not None,
            "loop": cls._loop is not None,
            "security": cls._security_instance is not None,
            "state_manager": cls._state_manager is not None,
            "active_web_agent": cls._active_web_agent,
            "voice_mode": cls._voice_mode_active,
            "executor_workers": cls._executor_max_workers if cls._executor else 0,
            "queue_size": cls._msg_queue.qsize()
        }
    
    @classmethod
    def print_status(cls) -> None:
        """Durum bilgilerini terminale yazdır"""
        status = cls.get_status()
        
        print("\n" + "═" * 50)
        print("  RuntimeContext Durumu")
        print("═" * 50)
        for key, value in status.items():
            print(f"  {key:20s}: {value}")
        print("═" * 50 + "\n")

# ═══════════════════════════════════════════════════════════════
# OTOMATİK BAŞLATMA
# ═══════════════════════════════════════════════════════════════
RuntimeContext.initialize()