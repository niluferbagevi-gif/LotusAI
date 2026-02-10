import asyncio
import contextlib
import io
import logging
import os
import re
import sys
import time

import keyboard
import torch
from pygame import mixer

from agents.definitions import AGENTS_CONFIG
from config import Config
from core.system_state import SystemState

logger = logging.getLogger("LotusSystem")

device = "cuda" if Config.USE_GPU else "cpu"


@contextlib.contextmanager
def ignore_stderr():
    """ALSA/JACK/PortAudio/OpenCV stderr gürültüsünü geçici olarak bastırır."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)
    except Exception:
        yield
    finally:
        os.close(devnull)


try:
    import edge_tts
except ImportError:
    edge_tts = None
    logger.warning("edge_tts modülü bulunamadı. Bulut tabanlı ses pasif.")


tts_model = None
if Config.USE_XTTS:
    try:
        from TTS.api import TTS

        with ignore_stderr():
            logger.info("🔊 XTTS (GPU) Modeli Yükleniyor...")
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            logger.info("🔊 XTTS Kullanıma Hazır.")
    except Exception as e:
        logger.error(f"XTTS Başlatılamadı: {e}")


async def edge_stream(text, voice):
    if edge_tts is None:
        return None

    try:
        comm = edge_tts.Communicate(text, voice)
        data = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                data += chunk["data"]
        return data
    except Exception as e:
        logger.error(f"EdgeTTS Stream Hatası: {e}")
        return None


def play_voice(text, agent_name, state_mgr):
    if not text or not state_mgr:
        return

    clean = re.sub(r"#.*", "", text)
    clean = clean.replace("*", "").replace("_", "").strip()
    if not clean:
        return

    state_mgr.set_state(SystemState.SPEAKING)

    try:
        with ignore_stderr():
            if mixer.get_init() is None:
                mixer.init()

            mixer.music.unload()
            agent_data = AGENTS_CONFIG.get(agent_name, AGENTS_CONFIG.get("ATLAS", {}))

            wav_path = str(Config.VOICES_DIR / f"{agent_name.lower()}.wav")
            if not os.path.exists(wav_path):
                wav_path = agent_data.get("voice_ref", "voices/atlas.wav")

            edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")

            use_xtts_now = Config.USE_XTTS and tts_model and os.path.exists(wav_path)

            if use_xtts_now:
                try:
                    output_path = "out.wav"
                    tts_model.tts_to_file(text=clean, speaker_wav=wav_path, language="tr", file_path=output_path)
                    mixer.music.load(output_path)
                except Exception as e:
                    logger.error(f"XTTS Hatası (EdgeTTS'e geçiliyor): {e}")
                    use_xtts_now = False

            if not use_xtts_now:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    audio = loop.run_until_complete(edge_stream(clean, edge_voice))
                    loop.close()

                    if audio:
                        mixer.music.load(io.BytesIO(audio))
                    else:
                        return
                except Exception as e:
                    logger.error(f"EdgeTTS Fallback Hatası: {e}")
                    return

            mixer.music.play()

            while mixer.music.get_busy():
                if keyboard.is_pressed("space") or keyboard.is_pressed("esc"):
                    mixer.music.stop()
                    logger.info("🔇 Konuşma kullanıcı tarafından kesildi.")
                    break
                time.sleep(0.05)

    except Exception as e:
        logger.error(f"Ses Çalma İşlemi Başarısız: {e}")
    finally:
        state_mgr.set_state(SystemState.IDLE)
        if Config.USE_GPU:
            torch.cuda.empty_cache()
