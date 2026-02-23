"""
LotusAI Ses Sistemi (Audio System)
Sürüm: 2.6.0 (Merkezi Config Senkronizasyonu)
Açıklama: Text-to-Speech (TTS) ve ses çalma yönetimi
"""

import os
import io
import time
import re
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import suppress
from threading import Lock

# ═══════════════════════════════════════════════════════════════
# CORE IMPORTS
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel
from core.utils import ignore_stderr
from core.system_state import SystemState

# ═══════════════════════════════════════════════════════════════
# EXTERNAL LIBRARIES
# ═══════════════════════════════════════════════════════════════
try:
    from pygame import mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.error("⚠️ pygame yüklü değil, ses sistemi çalışmayacak")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.warning("⚠️ edge-tts yüklü değil, Edge TTS devre dışı")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logging.warning("⚠️ keyboard yüklü değil, kesme tuşları çalışmayacak")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════
logger = logging.getLogger("LotusAI.Audio")


# ═══════════════════════════════════════════════════════════════
# SABITLER
# ═══════════════════════════════════════════════════════════════
class AudioConfig:
    """Ses sistemi konfigürasyonu"""
    # Dosya isimleri
    TEMP_OUTPUT_FILE = "temp_tts_output.wav"
    
    # Timing
    PLAYBACK_CHECK_INTERVAL = 0.05  # saniye
    MIXER_INIT_RETRY_COUNT = 3
    MIXER_INIT_RETRY_DELAY = 0.5
    
    # Cleanup
    AUTO_CLEANUP_TEMP_FILES = True
    
    # Interruptible keys
    INTERRUPT_KEYS = ['space', 'esc']


# ═══════════════════════════════════════════════════════════════
# TTS MANAGER SINIFI
# ═══════════════════════════════════════════════════════════════
@dataclass
class TTSEngineInfo:
    """TTS engine bilgileri"""
    name: str
    available: bool
    device: str = "cpu"
    model: Optional[any] = None


class AudioSystem:
    """
    LotusAI Ses Sistemi Yöneticisi
    
    Singleton pattern ile TTS engine'leri ve ses çalma işlemlerini yönetir.
    Thread-safe yapıdadır.
    
    Desteklenen TTS Motorları:
    - XTTS (GPU/CPU desteği ile)
    - Edge TTS (Microsoft Azure)
    
    Özellikler:
    - Otomatik fallback (XTTS → Edge TTS)
    - GPU desteği
    - Kesme tuşları (Space/Esc)
    - State management entegrasyonu
    - Erişim seviyesi kontrolü (kısıtlı modda ses çalma devre dışı)
    """
    
    _instance: Optional['AudioSystem'] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Audio system başlatıcı"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # TTS Engines
            self.xtts_engine: Optional[TTSEngineInfo] = None
            self.edge_engine: Optional[TTSEngineInfo] = None
            
            # Device
            self.device = "cuda" if (Config.USE_GPU and TORCH_AVAILABLE) else "cpu"
            
            # Pygame mixer durumu
            self.mixer_initialized = False
            
            # Temp files
            self.temp_files: set = set()
            
            self._initialized = True
            logger.info("AudioSystem instance oluşturuldu")
    
    def initialize(self) -> bool:
        """
        Ses sistemini başlat
        
        Returns:
            Başarılı ise True
        """
        if not PYGAME_AVAILABLE:
            logger.error("Pygame yüklü değil, ses sistemi başlatılamadı")
            return False
        
        # Pygame mixer'ı başlat
        if not self._init_mixer():
            logger.error("Pygame mixer başlatılamadı")
            return False
        
        # XTTS'yi yükle (opsiyonel)
        if Config.USE_XTTS:
            self._load_xtts()
        
        # Edge TTS kontrolü
        if EDGE_TTS_AVAILABLE:
            self.edge_engine = TTSEngineInfo(
                name="Edge TTS",
                available=True
            )
            logger.info("✅ Edge TTS hazır")
        
        logger.info(
            f"✅ Ses sistemi hazır (Device: {self.device.upper()}, "
            f"XTTS: {self.xtts_engine is not None}, "
            f"Edge: {self.edge_engine is not None})"
        )
        
        return True
    
    def _init_mixer(self) -> bool:
        """
        Pygame mixer'ı başlat
        
        Returns:
            Başarılı ise True
        """
        for attempt in range(AudioConfig.MIXER_INIT_RETRY_COUNT):
            try:
                with ignore_stderr():
                    if mixer.get_init() is None:
                        mixer.init()
                
                self.mixer_initialized = True
                logger.debug("Pygame mixer başlatıldı")
                return True
            
            except Exception as e:
                logger.warning(f"Mixer başlatma denemesi {attempt + 1} başarısız: {e}")
                time.sleep(AudioConfig.MIXER_INIT_RETRY_DELAY)
        
        return False
    
    def _load_xtts(self) -> None:
        """XTTS modelini yükle (GPU desteği ile)"""
        try:
            from TTS.api import TTS
            
            logger.info("🔊 XTTS modeli yükleniyor...")
            
            with ignore_stderr():
                model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                
                if self.device == "cuda":
                    model = model.to("cuda")
            
            self.xtts_engine = TTSEngineInfo(
                name="XTTS",
                available=True,
                device=self.device,
                model=model
            )
            
            logger.info(f"✅ XTTS hazır (Device: {self.device.upper()})")
        
        except ImportError:
            logger.warning("⚠️ TTS kütüphanesi yüklü değil, XTTS devre dışı")
        
        except Exception as e:
            logger.error(f"❌ XTTS yükleme hatası: {e}")
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        TTS için metni temizle
        
        Args:
            text: Ham metin
        
        Returns:
            Temizlenmiş metin
        """
        # Markdown ve özel karakterleri kaldır
        clean = re.sub(r'#.*', '', text)  # Hashtag'leri kaldır
        clean = clean.replace('*', '')    # Yıldızları kaldır
        clean = clean.replace('_', '')    # Alt çizgileri kaldır
        clean = clean.strip()
        
        return clean
    
    async def _generate_edge_tts(
        self,
        text: str,
        voice: str
    ) -> Optional[bytes]:
        """
        Edge TTS ile ses üret
        
        Args:
            text: Konuşulacak metin
            voice: Ses modeli
        
        Returns:
            Audio bytes veya None
        """
        if not EDGE_TTS_AVAILABLE:
            return None
        
        try:
            comm = edge_tts.Communicate(text, voice)
            data = b""
            
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            
            return data if data else None
        
        except Exception as e:
            logger.error(f"Edge TTS hatası: {e}")
            return None
    
    def _generate_xtts(
        self,
        text: str,
        speaker_wav: str,
        output_path: str
    ) -> bool:
        """
        XTTS ile ses üret
        
        Args:
            text: Konuşulacak metin
            speaker_wav: Referans ses dosyası
            output_path: Çıktı dosya yolu
        
        Returns:
            Başarılı ise True
        """
        if not self.xtts_engine or not self.xtts_engine.model:
            return False
        
        try:
            self.xtts_engine.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="tr",
                file_path=output_path
            )
            
            # Temp file listesine ekle
            self.temp_files.add(output_path)
            
            return True
        
        except Exception as e:
            logger.error(f"XTTS üretim hatası: {e}")
            return False
    
    def _get_agent_voice_config(
        self,
        agent_name: str
    ) -> tuple[str, str]:
        """
        Agent için ses konfigürasyonu al
        
        Args:
            agent_name: Agent adı
        
        Returns:
            Tuple[speaker_wav_path, edge_voice_id]
        """
        # GÜNCELLEME: Ajan ayarları artık doğrudan Config'den alınıyor
        agent_data = Config.AGENT_CONFIGS.get(
            agent_name.upper(),
            Config.AGENT_CONFIGS.get("ATLAS", {})
        )
        
        # Speaker wav dosyası
        wav_filename = f"{agent_name.lower()}.wav"
        wav_path = Config.VOICES_DIR / wav_filename
        
        if not wav_path.exists():
            # Fallback to agent definition
            wav_ref = agent_data.get("voice_ref", "voices/atlas.wav")
            wav_path = Path(wav_ref)
        
        # Edge voice
        edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")
        
        return str(wav_path), edge_voice
    
    def _check_interrupt_keys(self) -> bool:
        """
        Kesme tuşlarını kontrol et
        
        Returns:
            Basıldıysa True
        """
        if not KEYBOARD_AVAILABLE:
            return False
        
        try:
            for key in AudioConfig.INTERRUPT_KEYS:
                if keyboard.is_pressed(key):
                    return True
        except Exception:
            pass
        
        return False
    
    def play_audio(
        self,
        text: str,
        agent_name: str,
        state_manager: Optional[SystemState] = None
    ) -> None:
        """
        Metni seslendir
        
        Args:
            text: Konuşulacak metin
            agent_name: Konuşan agent
            state_manager: State manager (opsiyonel)
        """
        # Erişim seviyesi kontrolü: Kısıtlı modda ses çalma devre dışı
        if Config.ACCESS_LEVEL == AccessLevel.RESTRICTED:
            logger.debug("Kısıtlı modda ses çalma atlandı")
            return
        
        if not text:
            return
        
        if not self.mixer_initialized:
            logger.warning("Mixer başlatılmamış, ses çalınamıyor")
            return
        
        # Metni temizle
        clean_text = self._clean_text(text)
        if not clean_text:
            return
        
        # State'i güncelle
        if state_manager:
            state_manager.set_state(SystemState.SPEAKING)
        
        try:
            # Agent voice config
            speaker_wav, edge_voice = self._get_agent_voice_config(agent_name)
            
            # TTS seçimi ve üretim
            audio_loaded = False
            
            # 1. XTTS dene
            if (self.xtts_engine and 
                self.xtts_engine.available and 
                Path(speaker_wav).exists()):
                
                output_path = Config.WORK_DIR / AudioConfig.TEMP_OUTPUT_FILE
                
                if self._generate_xtts(clean_text, speaker_wav, str(output_path)):
                    try:
                        with ignore_stderr():
                            mixer.music.unload()
                            mixer.music.load(str(output_path))
                        audio_loaded = True
                        logger.debug(f"XTTS kullanıldı: {agent_name}")
                    except Exception as e:
                        logger.warning(f"XTTS ses yükleme hatası: {e}")
            
            # 2. Edge TTS fallback
            if not audio_loaded and self.edge_engine and self.edge_engine.available:
                try:
                    # Asyncio event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    audio_bytes = loop.run_until_complete(
                        self._generate_edge_tts(clean_text, edge_voice)
                    )
                    
                    loop.close()
                    
                    if audio_bytes:
                        with ignore_stderr():
                            mixer.music.unload()
                            mixer.music.load(io.BytesIO(audio_bytes))
                        audio_loaded = True
                        logger.debug(f"Edge TTS kullanıldı: {agent_name}")
                
                except Exception as e:
                    logger.error(f"Edge TTS hatası: {e}")
            
            if not audio_loaded:
                logger.warning("Hiçbir TTS engine çalışmadı")
                return
            
            # Ses çalma
            mixer.music.play()
            
            # Çalma döngüsü (interrupt kontrolü ile)
            while mixer.music.get_busy():
                if self._check_interrupt_keys():
                    mixer.music.stop()
                    logger.info("Ses çalma kullanıcı tarafından durduruldu")
                    break
                
                time.sleep(AudioConfig.PLAYBACK_CHECK_INTERVAL)
        
        except Exception as e:
            logger.error(f"Ses çalma hatası: {e}", exc_info=True)
        
        finally:
            # State'i geri al
            if state_manager:
                state_manager.set_state(SystemState.IDLE)
            
            # GPU temizliği
            if self.device == "cuda" and TORCH_AVAILABLE:
                with suppress(Exception):
                    torch.cuda.empty_cache()
    
    def cleanup(self) -> None:
        """Geçici dosyaları temizle"""
        if not AudioConfig.AUTO_CLEANUP_TEMP_FILES:
            return
        
        for temp_file in self.temp_files:
            try:
                path = Path(temp_file)
                if path.exists():
                    path.unlink()
                    logger.debug(f"Temp file silindi: {temp_file}")
            except Exception as e:
                logger.warning(f"Temp file silinemedi ({temp_file}): {e}")
        
        self.temp_files.clear()
    
    def shutdown(self) -> None:
        """Audio system'i kapat"""
        logger.info("Audio system kapatılıyor...")
        
        # Mixer'ı durdur
        if self.mixer_initialized:
            with suppress(Exception):
                mixer.music.stop()
                mixer.quit()
        
        # Temp dosyaları temizle
        self.cleanup()
        
        # XTTS modelini temizle
        if self.xtts_engine and self.xtts_engine.model:
            with suppress(Exception):
                del self.xtts_engine.model
                self.xtts_engine.model = None
        
        logger.info("✅ Audio system kapatıldı")


# ═══════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════
_audio_system: Optional[AudioSystem] = None


def init_audio_system() -> bool:
    """
    Ses sistemini başlat (Global fonksiyon)
    
    Returns:
        Başarılı ise True
    """
    global _audio_system
    
    if _audio_system is None:
        _audio_system = AudioSystem()
    
    return _audio_system.initialize()


def play_voice(
    text: str,
    agent_name: str,
    state_manager: Optional[SystemState] = None
) -> None:
    """
    Metni seslendir (Global fonksiyon - geriye uyumluluk)
    
    Args:
        text: Konuşulacak metin
        agent_name: Konuşan agent
        state_manager: State manager (opsiyonel)
    """
    global _audio_system
    
    if _audio_system is None:
        logger.warning("Audio system başlatılmamış, başlatılıyor...")
        init_audio_system()
    
    if _audio_system:
        _audio_system.play_audio(text, agent_name, state_manager)


def get_audio_system() -> Optional[AudioSystem]:
    """
    Audio system instance'ını döndür
    
    Returns:
        AudioSystem veya None
    """
    return _audio_system


def cleanup_audio_system() -> None:
    """Audio system'i temizle"""
    global _audio_system
    
    if _audio_system:
        _audio_system.shutdown()
        _audio_system = None