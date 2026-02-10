import asyncio
import logging
import os
import sys
import threading
import time
import webbrowser

import speech_recognition as sr
import torch
from flask import Flask

from config import Config
from core.memory import MemoryManager
from core.security import SecurityManager
from core.system_state import SystemState
from managers.accounting import AccountingManager
from managers.camera import CameraManager
from managers.code_manager import CodeManager
from managers.delivery import DeliveryManager
from managers.finance import FinanceManager
from managers.messaging import MessagingManager
from managers.nlp import NLPManager
from managers.operations import OperationsManager
from managers.system_health import SystemHealthManager
from agents.engine import AgentEngine
from agents.poyraz import PoyrazAgent
from agents.sidar import SidarAgent
from runtime.runtime_context import RuntimeContext
from runtime.web_routes import create_web_blueprint, run_flask
from services.voice_service import ignore_stderr, play_voice

# --- TERMİNAL TEMİZLİĞİ VE LOG FİLTRELEME ---
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- KRİTİK HATA DÜZELTMESİ (Monkey Patch) ---
try:
    import transformers.pytorch_utils

    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):

        def _isin_mps_friendly():
            return False

        transformers.pytorch_utils.isin_mps_friendly = _isin_mps_friendly
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LotusSystem")

app = Flask(__name__, template_folder=str(Config.TEMPLATE_DIR), static_folder=str(Config.STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(Config.UPLOAD_DIR)
app.register_blueprint(create_web_blueprint(app, play_voice))

# GPU Durumunu Config'den alıyoruz (TEK KAYNAK)
device = "cuda" if Config.USE_GPU else "cpu"
if Config.USE_GPU:
    try:
        torch.cuda.empty_cache()
        logger.debug(f"System Device: {device.upper()} - VRAM Optimized.")
    except Exception as e:
        logger.warning(f"GPU Bellek temizleme hatası: {e}")

# Media Manager Opsiyonel
try:
    from managers.media import MediaManager

    MEDIA_AVAILABLE = True
except ImportError:
    MEDIA_AVAILABLE = False
    logger.info("MediaManager bulunamadı, medya özellikleri devre dışı.")

# RuntimeContext üzerindeki messaging manager'ı hazır tut
RuntimeContext.messaging_manager = MessagingManager()


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


async def main_loop(mode):
    RuntimeContext.loop = asyncio.get_running_loop()

    Config.set_provider_mode(mode)
    print(f"{Colors.HEADER}--- {Config.PROJECT_NAME.upper()} SİSTEMİ v{Config.VERSION} ---{Colors.ENDC}")
    print(f"{Colors.CYAN}⚙️ Donanım: {Config.GPU_INFO} | Cihaz: {device.upper()} | Sağlayıcı: {Config.AI_PROVIDER.upper()}{Colors.ENDC}")

    state_manager = SystemState()
    RuntimeContext.state_manager = state_manager
    memory_manager = MemoryManager()

    camera_manager = CameraManager()
    with ignore_stderr():
        camera_manager.start()

    security_manager = SecurityManager(camera_manager)
    RuntimeContext.security_instance = security_manager

    code_manager = CodeManager(Config.WORK_DIR)
    sys_health_manager = SystemHealthManager()
    finance_manager = FinanceManager()
    accounting_manager = AccountingManager()
    ops_manager = OperationsManager()

    delivery_manager = DeliveryManager()
    if Config.FINANCE_MODE:
        logger.info("🛵 Paket Servis Modülü Aktif Edildi.")
        delivery_manager.start_service()

    nlp_manager = NLPManager()

    poyraz_agent = PoyrazAgent(nlp_manager, {})

    sidar_tools = {
        "code": code_manager,
        "system": sys_health_manager,
        "security": security_manager,
        "memory": memory_manager,
    }
    sidar_agent = SidarAgent(sidar_tools)

    tools = {
        "camera": camera_manager,
        "code": code_manager,
        "system": sys_health_manager,
        "finance": finance_manager,
        "operations": ops_manager,
        "accounting": accounting_manager,
        "messaging": RuntimeContext.messaging_manager,
        "delivery": delivery_manager,
        "nlp": nlp_manager,
        "poyraz_special": poyraz_agent,
        "sidar_special": sidar_agent,
        "state": state_manager,
    }

    if MEDIA_AVAILABLE:
        tools["media"] = MediaManager()

    poyraz_agent.update_tools(tools)
    RuntimeContext.engine = AgentEngine(memory_manager, tools)

    if (Config.TEMPLATE_DIR / "index.html").exists():
        flask_thread = threading.Thread(target=lambda: run_flask(app), daemon=True)
        flask_thread.start()
        logger.info("🌐 Web Dashboard Hazır: http://localhost:5000")
        threading.Timer(3.0, lambda: webbrowser.open("http://localhost:5000")).start()
    else:
        logger.error("❌ HATA: Dashboard dosyaları bulunamadı!")

    r = sr.Recognizer()
    mic = None

    try:
        with ignore_stderr():
            mic = sr.Microphone()
            with mic as source:
                print("🎤 Ortam sesi kalibre ediliyor...")
                r.adjust_for_ambient_noise(source, duration=1.0)
    except OSError as e:
        logger.warning(f"⚠️ Mikrofon bulunamadı veya erişilemiyor: {e}")
        logger.warning("🎤 Sesli komut sistemi devre dışı bırakıldı.")
        RuntimeContext.voice_mode_active = False
    except Exception as e:
        logger.error(f"Mikrofon başlatma hatası: {e}")
        RuntimeContext.voice_mode_active = False

    print(f"{Colors.GREEN}✅ {Config.PROJECT_NAME.upper()} TÜM SİSTEMLER AKTİF (Cihaz: {device.upper()}).{Colors.ENDC}")

    while state_manager.is_running():
        try:
            current_time = time.time()

            if delivery_manager.is_selenium_active and int(current_time) % 30 == 0:
                order_alerts = delivery_manager.check_new_orders()
                if order_alerts:
                    for alert in order_alerts:
                        RuntimeContext.executor.submit(play_voice, f"Yeni bildirim: {alert}", "GAYA", state_manager)

            if RuntimeContext.voice_mode_active and mic and state_manager.should_listen():
                state_manager.set_state(SystemState.LISTENING)
                user_input = ""
                audio_data = None

                try:
                    with mic as source:
                        audio_data = await asyncio.to_thread(r.listen, source, timeout=3, phrase_time_limit=10)
                        user_input = await asyncio.to_thread(r.recognize_google, audio_data, language="tr-TR")
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass

                if user_input:
                    print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")
                    sec_result = security_manager.analyze_situation(audio_data=audio_data)

                    if "hafızayı sil" in user_input.lower() or "hafızayı temizle" in user_input.lower():
                        memory_manager.clear_history()
                        RuntimeContext.executor.submit(play_voice, "Hafıza temizlendi.", "ATLAS", state_manager)
                        continue

                    detected_agent = RuntimeContext.engine.determine_agent(user_input)
                    current_agent = detected_agent if detected_agent else "ATLAS"

                    state_manager.set_state(SystemState.THINKING)
                    resp_data = await RuntimeContext.engine.get_response(current_agent, user_input, sec_result)

                    print(f"🤖 {resp_data['agent']}: {resp_data['content']}")
                    RuntimeContext.executor.submit(play_voice, resp_data["content"], resp_data["agent"], state_manager)

            else:
                if state_manager.get_state() not in [SystemState.THINKING, SystemState.SPEAKING]:
                    state_manager.set_state(SystemState.IDLE)
                await asyncio.sleep(0.5)

            await asyncio.sleep(0.05)

        except Exception as main_err:
            logger.error(f"ANA DÖNGÜ HATASI: {main_err}")
            await asyncio.sleep(2)

    camera_manager.stop()
    delivery_manager.stop_service()
    RuntimeContext.executor.shutdown(wait=False)
    logger.info("LotusAI güvenli bir şekilde kapatıldı.")


def start_lotus_system(mode="online"):
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(main_loop(mode))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!] LotusAI kullanıcı tarafından kapatıldı.{Colors.ENDC}")
    except Exception as e:
        logger.critical(f"BAŞLATMA SIRASINDA KRİTİK HATA: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    mode = os.getenv("AI_PROVIDER", "online")
    start_lotus_system(mode)
