"""
LotusAI Ana Sistem Motoru
Sürüm: 2.5.3
Açıklama: Multi-agent AI sistemi, ses tanıma, güvenlik ve otomasyon
"""

import asyncio
import time
import speech_recognition as sr
import threading
import os
import sys
import logging
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import suppress

# ═══════════════════════════════════════════════════════════════
# TORCH IMPORT (GPU İÇİN)
# ═══════════════════════════════════════════════════════════════
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("⚠️ PyTorch yüklü değil, GPU desteği devre dışı")

# ═══════════════════════════════════════════════════════════════
# CORE MODÜLLER
# ═══════════════════════════════════════════════════════════════
from config import Config
from core.utils import setup_logging, patch_transformers, ignore_stderr
from core.runtime import RuntimeContext
from core.audio import init_audio_system, play_voice
from core.system_state import SystemState
from core.memory import MemoryManager
from core.security import SecurityManager

# ═══════════════════════════════════════════════════════════════
# MANAGER MODÜLLER
# ═══════════════════════════════════════════════════════════════
from managers.camera import CameraManager
from managers.code_manager import CodeManager
from managers.system_health import SystemHealthManager
from managers.finance import FinanceManager
from managers.operations import OperationsManager
from managers.accounting import AccountingManager
from managers.messaging import MessagingManager
from managers.delivery import DeliveryManager
from managers.nlp import NLPManager

# ═══════════════════════════════════════════════════════════════
# AGENT MODÜLLER
# ═══════════════════════════════════════════════════════════════
from agents.engine import AgentEngine
from agents.poyraz import PoyrazAgent
from agents.sidar import SidarAgent

# ═══════════════════════════════════════════════════════════════
# WEB SERVER
# ═══════════════════════════════════════════════════════════════
from server import run_flask

# ═══════════════════════════════════════════════════════════════
# LOGLAMA SİSTEMİ
# ═══════════════════════════════════════════════════════════════
setup_logging()
patch_transformers()
logger = logging.getLogger("LotusSystem")

# ═══════════════════════════════════════════════════════════════
# TERMINAL RENK KODLARI
# ═══════════════════════════════════════════════════════════════
class Colors:
    """Terminal renklendirme için ANSI kodları"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'


# ═══════════════════════════════════════════════════════════════
# SİSTEM SABITLERI
# ═══════════════════════════════════════════════════════════════
class SystemConfig:
    """Sistem çalışma parametreleri"""
    # Ses tanıma
    AMBIENT_NOISE_DURATION: float = 1.0
    LISTEN_TIMEOUT: float = 3.0
    PHRASE_TIME_LIMIT: float = 10.0
    SPEECH_LANGUAGE: str = "tr-TR"
    
    # Sipariş kontrolü
    ORDER_CHECK_INTERVAL: int = 30  # saniye
    
    # Döngü bekleme süreleri
    IDLE_SLEEP: float = 0.5
    LOOP_SLEEP: float = 0.05
    ERROR_SLEEP: float = 2.0
    
    # Web dashboard
    DASHBOARD_PORT: int = 5000
    DASHBOARD_OPEN_DELAY: float = 3.0
    
    # Thread pool
    MAX_WORKERS: int = 5


# ═══════════════════════════════════════════════════════════════
# SİSTEM YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════════
class LotusSystem:
    """
    LotusAI Ana Sistem Yöneticisi
    
    Sorumluluklar:
    - Tüm managerleri başlatma
    - Agent'ları koordine etme
    - Ses tanıma döngüsü
    - Web dashboard
    - Sistem kapatma
    """
    
    def __init__(self, mode: str = "online"):
        """
        Sistem başlatıcı
        
        Args:
            mode: 'online' (gemini) veya 'local' (ollama)
        """
        self.mode = mode
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Managerlar
        self.state_manager: Optional[SystemState] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.delivery_manager: Optional[DeliveryManager] = None
        
        # Agents
        self.engine: Optional[AgentEngine] = None
        
        # Ses tanıma
        self.recognizer: Optional[sr.Recognizer] = None
        self.microphone: Optional[sr.Microphone] = None
        
        logger.info(f"LotusSystem başlatılıyor - Mod: {mode}")
    
    def _setup_gpu(self) -> str:
        """
        GPU'yu yapılandır ve device döndür
        
        Returns:
            'cuda' veya 'cpu'
        """
        if Config.USE_GPU and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                device = "cuda"
                logger.info(f"🚀 GPU aktif: {Config.GPU_INFO}")
            except Exception as e:
                logger.warning(f"⚠️ GPU temizleme hatası: {e}")
                device = "cpu"
        else:
            device = "cpu"
        
        return device
    
    def _print_startup_banner(self, device: str) -> None:
        """Başlangıç banner'ını yazdır"""
        print(f"\n{Colors.HEADER}{'═' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}  {Config.PROJECT_NAME.upper()} SİSTEMİ v{Config.VERSION}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'═' * 70}{Colors.ENDC}")
        print(f"{Colors.CYAN}  ⚙️  Donanım    : {Config.GPU_INFO}{Colors.ENDC}")
        print(f"{Colors.CYAN}  🖥️  Cihaz      : {device.upper()}{Colors.ENDC}")
        print(f"{Colors.CYAN}  🧠 Sağlayıcı  : {Config.AI_PROVIDER.upper()}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'═' * 70}{Colors.ENDC}\n")
    
    def _initialize_managers(self) -> Tuple[Dict[str, Any], Any]:
        """
        Tüm managerleri başlat
        
        Returns:
            Tuple[Manager dictionary, NLP Manager]
        """
        logger.info("📦 Managerlar başlatılıyor...")
        
        # Core managerlar
        self.state_manager = SystemState()
        RuntimeContext.set_state_manager(self.state_manager)  # ✅ YENİ
        
        self.memory_manager = MemoryManager()
        
        # Kamera
        self.camera_manager = CameraManager()
        with ignore_stderr():
            self.camera_manager.start()
        logger.info("📷 Kamera sistemi aktif")
        
        # Güvenlik
        self.security_manager = SecurityManager(self.camera_manager)
        RuntimeContext.set_security_instance(self.security_manager)  # ✅ YENİ
        logger.info("🔒 Güvenlik sistemi aktif")
        
        # Diğer managerlar
        code_manager = CodeManager(Config.WORK_DIR)
        sys_health_manager = SystemHealthManager()
        finance_manager = FinanceManager()
        accounting_manager = AccountingManager()
        ops_manager = OperationsManager()
        messaging_manager = MessagingManager()
        RuntimeContext.set_messaging_manager(messaging_manager)  # ✅ YENİ
        
        # Delivery manager
        self.delivery_manager = DeliveryManager()
        if Config.FINANCE_MODE:
            logger.info("🛵 Paket Servis Modülü aktif")
            self.delivery_manager.start_service()
        
        nlp_manager = NLPManager()
        
        # Ses sistemi
        init_audio_system()
        logger.info("🔊 Ses sistemi aktif")
        
        # Tools dictionary
        tools = {
            "camera": self.camera_manager,
            "code": code_manager,
            "system": sys_health_manager,
            "finance": finance_manager,
            "operations": ops_manager,
            "accounting": accounting_manager,
            "messaging": messaging_manager,
            "delivery": self.delivery_manager,
            "nlp": nlp_manager,
            "state": self.state_manager
        }
        
        # Media manager (opsiyonel)
        try:
            from managers.media import MediaManager
            tools['media'] = MediaManager()
            logger.info("🎬 Media manager yüklendi")
        except ImportError:
            logger.debug("Media manager bulunamadı, atlanıyor")
        
        return tools, nlp_manager
    
    def _initialize_agents(self, tools: Dict[str, Any], nlp_manager: Any) -> None:
        """
        Agent sistemini başlat
        
        Args:
            tools: Manager dictionary
            nlp_manager: NLP Manager instance
        """
        logger.info("🤖 Agentlar başlatılıyor...")
        
        # Poyraz agent
        poyraz_agent = PoyrazAgent(nlp_manager, {})
        
        # Sidar agent
        sidar_tools = {
            'code': tools['code'],
            'system': tools['system'],
            'security': self.security_manager,
            'memory': self.memory_manager
        }
        sidar_agent = SidarAgent(sidar_tools)
        
        # Agent'ları tools'a ekle
        tools['poyraz_special'] = poyraz_agent
        tools['sidar_special'] = sidar_agent
        
        # Poyraz'a tüm toolları ver
        poyraz_agent.update_tools(tools)
        
        # Engine'i oluştur
        self.engine = AgentEngine(self.memory_manager, tools)
        RuntimeContext.set_engine(self.engine)  # ✅ YENİ
        
        logger.info("✅ Agent engine hazır")
    
    def _start_web_dashboard(self) -> None:
        """Web dashboard'u başlat"""
        dashboard_path = Config.TEMPLATE_DIR / "index.html"
        
        if not dashboard_path.exists():
            logger.error(f"❌ Dashboard bulunamadı: {dashboard_path}")
            return
        
        try:
            # Flask thread'i başlat
            flask_thread = threading.Thread(
                target=run_flask,
                daemon=True,
                name="FlaskServer"
            )
            flask_thread.start()
            
            dashboard_url = f"http://localhost:{SystemConfig.DASHBOARD_PORT}"
            logger.info(f"🌐 Web Dashboard: {dashboard_url}")
            
            # Tarayıcıyı gecikmeyle aç
            threading.Timer(
                SystemConfig.DASHBOARD_OPEN_DELAY,
                lambda: webbrowser.open(dashboard_url)
            ).start()
            
        except Exception as e:
            logger.error(f"❌ Dashboard başlatma hatası: {e}")
    
    def _setup_microphone(self) -> bool:
        """
        Mikrofonu yapılandır
        
        Returns:
            Başarılı ise True
        """
        self.recognizer = sr.Recognizer()
        
        try:
            with ignore_stderr():
                self.microphone = sr.Microphone()
                
                with self.microphone as source:
                    print(f"{Colors.YELLOW}🎤 Ortam sesi kalibre ediliyor...{Colors.ENDC}")
                    self.recognizer.adjust_for_ambient_noise(
                        source,
                        duration=SystemConfig.AMBIENT_NOISE_DURATION
                    )
                
                logger.info("✅ Mikrofon kalibre edildi")
                RuntimeContext.set_voice_mode(True)  # ✅ YENİ
                return True
        
        except Exception as e:
            logger.warning(f"⚠️ Mikrofon hatası: {e}")
            RuntimeContext.set_voice_mode(False)  # ✅ YENİ
            return False
    
    async def _listen_for_speech(self) -> Optional[str]:
        """
        Kullanıcı konuşmasını dinle
        
        Returns:
            Tanınan metin veya None
        """
        if not self.microphone or not self.recognizer:
            return None
        
        try:
            with self.microphone as source:
                audio_data = await asyncio.to_thread(
                    self.recognizer.listen,
                    source,
                    timeout=SystemConfig.LISTEN_TIMEOUT,
                    phrase_time_limit=SystemConfig.PHRASE_TIME_LIMIT
                )
                
                # Google Speech Recognition
                text = await asyncio.to_thread(
                    self.recognizer.recognize_google,
                    audio_data,
                    language=SystemConfig.SPEECH_LANGUAGE
                )
                
                return text
        
        except sr.WaitTimeoutError:
            # Normal - zaman aşımı
            return None
        
        except sr.UnknownValueError:
            # Normal - anlaşılamayan ses
            return None
        
        except Exception as e:
            logger.error(f"Ses tanıma hatası: {e}")
            return None
    
    async def _process_user_input(self, user_input: str, audio_data: Optional[Any] = None) -> None:
        """
        Kullanıcı girdisini işle
        
        Args:
            user_input: Tanınan metin
            audio_data: Ham ses verisi (opsiyonel)
        """
        print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")
        
        # Güvenlik analizi
        sec_result = self.security_manager.analyze_situation(audio_data=audio_data)
        
        # Uygun agent'ı belirle
        detected_agent = self.engine.determine_agent(user_input)
        current_agent = detected_agent if detected_agent else "ATLAS"
        
        # Düşünme modu
        self.state_manager.set_state(SystemState.THINKING)
        
        # Yanıt al
        resp_data = await self.engine.get_response(
            current_agent,
            user_input,
            sec_result
        )
        
        # Yanıtı göster
        print(f"{Colors.GREEN}🤖 {resp_data['agent']}: {resp_data['content']}{Colors.ENDC}")
        
        # Seslendirme (RuntimeContext'in yeni metodunu kullan)
        RuntimeContext.submit_task(  # ✅ YENİ
            play_voice,
            resp_data['content'],
            resp_data['agent'],
            self.state_manager
        )
    
    async def _check_delivery_orders(self) -> None:
        """Yeni siparişleri kontrol et"""
        if not self.delivery_manager.is_selenium_active:
            return
        
        try:
            order_alerts = self.delivery_manager.check_new_orders()
            
            if order_alerts:
                for alert in order_alerts:
                    logger.info(f"📦 Yeni sipariş: {alert}")
                    RuntimeContext.submit_task(  # ✅ YENİ
                        play_voice,
                        f"Yeni bildirim: {alert}",
                        "GAYA",
                        self.state_manager
                    )
        
        except Exception as e:
            logger.error(f"Sipariş kontrol hatası: {e}")
    
    async def _main_loop(self) -> None:
        """Ana sistem döngüsü"""
        last_order_check = 0
        
        while self.state_manager.is_running():
            try:
                current_time = time.time()
                
                # Sipariş kontrolü (periyodik)
                if (Config.FINANCE_MODE and 
                    current_time - last_order_check >= SystemConfig.ORDER_CHECK_INTERVAL):
                    await self._check_delivery_orders()
                    last_order_check = current_time
                
                # Ses dinleme modu
                if RuntimeContext.is_voice_mode_active() and self.state_manager.should_listen():  # ✅ YENİ
                    self.state_manager.set_state(SystemState.LISTENING)
                    
                    user_input = await self._listen_for_speech()
                    
                    if user_input:
                        await self._process_user_input(user_input)
                
                # Idle modu
                else:
                    current_state = self.state_manager.get_state()
                    if current_state not in [SystemState.THINKING, SystemState.SPEAKING]:
                        self.state_manager.set_state(SystemState.IDLE)
                    
                    await asyncio.sleep(SystemConfig.IDLE_SLEEP)
                
                # Kısa bekleme (CPU kullanımını azalt)
                await asyncio.sleep(SystemConfig.LOOP_SLEEP)
            
            except KeyboardInterrupt:
                logger.info("Kullanıcı tarafından durduruldu")
                break
            
            except Exception as e:
                logger.error(f"Ana döngü hatası: {e}", exc_info=True)
                await asyncio.sleep(SystemConfig.ERROR_SLEEP)
    
    def _cleanup(self) -> None:
        """Sistem kapatma işlemleri"""
        logger.info("🛑 Sistem kapatılıyor...")
        
        # Kamerayı durdur
        if self.camera_manager:
            with suppress(Exception):
                self.camera_manager.stop()
                logger.info("✓ Kamera durduruldu")
        
        # Delivery'yi durdur
        if self.delivery_manager:
            with suppress(Exception):
                self.delivery_manager.stop_service()
                logger.info("✓ Delivery servisi durduruldu")
        
        # RuntimeContext'i temizle (YENİ!)
        with suppress(Exception):
            RuntimeContext.shutdown(wait=True, timeout=5.0)  # ✅ YENİ
            logger.info("✓ RuntimeContext kapatıldı")
        
        logger.info("✅ LotusAI temiz bir şekilde kapatıldı")
    
    async def run(self) -> None:
        """Sistemi başlat ve çalıştır"""
        try:
            # RuntimeContext'i başlat (YENİ!)
            RuntimeContext.initialize(max_workers=SystemConfig.MAX_WORKERS)  # ✅ YENİ
            
            # Event loop'u kaydet
            self.loop = asyncio.get_running_loop()
            RuntimeContext.set_loop(self.loop)  # ✅ YENİ
            
            # Provider modunu ayarla
            Config.set_provider_mode(self.mode)
            
            # GPU'yu hazırla
            device = self._setup_gpu()
            
            # Banner yazdır
            self._print_startup_banner(device)
            
            # Managerleri başlat
            tools, nlp_manager = self._initialize_managers()
            
            # Agent'ları başlat
            self._initialize_agents(tools, nlp_manager)
            
            # Web dashboard'u başlat
            self._start_web_dashboard()
            
            # Mikrofonu ayarla
            mic_ready = self._setup_microphone()
            
            # Sistem hazır
            print(f"\n{Colors.GREEN}{'═' * 70}{Colors.ENDC}")
            print(f"{Colors.GREEN}{Colors.BOLD}  ✅ {Config.PROJECT_NAME.upper()} TÜM SİSTEMLER AKTİF{Colors.ENDC}")
            print(f"{Colors.GREEN}{'═' * 70}{Colors.ENDC}\n")
            
            if not mic_ready:
                print(f"{Colors.YELLOW}  ⚠️  Mikrofon devre dışı - Sadece dashboard aktif{Colors.ENDC}\n")
            
            # Debug: RuntimeContext durumu
            if Config.DEBUG_MODE:
                RuntimeContext.print_status()
            
            # Ana döngüyü başlat
            await self._main_loop()
        
        finally:
            # Temizlik
            self._cleanup()


# ═══════════════════════════════════════════════════════════════
# BAŞLATMA FONKSİYONU (Launcher için)
# ═══════════════════════════════════════════════════════════════
def start_lotus_system(mode: str = "online") -> None:
    """
    LotusAI sistemini başlatır (Launcher tarafından çağrılır)
    
    Args:
        mode: 'online' (Gemini) veya 'local' (Ollama)
    """
    try:
        # Windows için event loop policy
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Sistem instance'ı oluştur
        system = LotusSystem(mode=mode)
        
        # Asenkron olarak çalıştır
        asyncio.run(system.run())
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!] LotusAI kullanıcı tarafından kapatıldı{Colors.ENDC}")
    
    except Exception as e:
        logger.critical(f"Kritik sistem hatası: {e}", exc_info=True)
        print(f"\n{Colors.FAIL}❌ Sistem başlatılamadı: {e}{Colors.ENDC}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# DİREKT ÇALIŞTIRMA (Test için)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = os.getenv("AI_PROVIDER", "online")
    start_lotus_system(mode)

# import asyncio
# import time
# import speech_recognition as sr
# from pygame import mixer
# import keyboard
# import threading
# import os
# import sys
# import re
# import io
# import queue
# import webbrowser
# import cv2
# import numpy as np
# import logging
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# from concurrent.futures import ThreadPoolExecutor

# # --- YAPILANDIRMA VE MODÜLLER ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("LotusSystem")

# try:
#     from config import Config
#     from core.system_state import SystemState
#     from core.memory import MemoryManager
#     from core.security import SecurityManager

#     from managers.camera import CameraManager
#     from managers.code_manager import CodeManager
#     from managers.system_health import SystemHealthManager 
#     from managers.finance import FinanceManager 
#     from managers.operations import OperationsManager
#     from managers.accounting import AccountingManager
#     from managers.messaging import MessagingManager
#     from managers.delivery import DeliveryManager
    
#     from managers.nlp import NLPManager
#     from agents.definitions import AGENTS_CONFIG
#     from agents.engine import AgentEngine
#     from agents.poyraz import PoyrazAgent
#     from agents.sidar import SidarAgent

# except ImportError as e:
#     logger.critical(f"KRİTİK HATA: Modüller yüklenirken sorun oluştu. Eksik dosya olabilir.\nHata: {e}")
#     # Kritik modüller yoksa devam edemeyiz
#     if "config" in str(e) or "core" in str(e):
#         sys.exit(1)

# # Media Manager Opsiyonel
# try:
#     from managers.media import MediaManager
#     MEDIA_AVAILABLE = True
# except ImportError:
#     MEDIA_AVAILABLE = False
#     logger.info("MediaManager bulunamadı, medya özellikleri devre dışı.")

# # --- RENKLER ---
# class Colors:
#     HEADER = '\033[95m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     YELLOW = '\033[93m'
#     CYAN = '\033[96m'

# # --- GLOBAL CONTEXT (RUNTIME) ---
# class RuntimeContext:
#     """Tüm global değişkenlerin merkezi yönetimi."""
#     msg_queue = queue.Queue()
#     messaging_manager = MessagingManager()
#     engine = None 
#     loop = None
#     security_instance = None 
#     state_manager = None
    
#     # Web Durumları
#     active_web_agent = "ATLAS"
#     voice_mode_active = False
#     executor = ThreadPoolExecutor(max_workers=5)

# # --- FLASK VE WEB SİSTEMİ YAPILANDIRMASI ---
# app = Flask(__name__, 
#             template_folder=str(Config.TEMPLATE_DIR), 
#             static_folder=str(Config.STATIC_DIR))
# app.config['UPLOAD_FOLDER'] = str(Config.UPLOAD_DIR)

# # --- FLASK ROUTE TANIMLARI ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/toggle_voice', methods=['POST'])
# def toggle_voice_api():
#     """Web arayüzünden sesli dinlemeyi açıp kapatır."""
#     data = request.json
#     if data and 'active' in data:
#         RuntimeContext.voice_mode_active = data['active']
#     else:
#         RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active
        
#     status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
#     logger.info(f"🎙️ Mikrofon Modu Değiştirildi: {status_msg}")
#     return jsonify({"status": "success", "active": RuntimeContext.voice_mode_active})

# @app.route('/api/chat_history', methods=['GET'])
# def get_chat_history():
#     agent_name = request.args.get('agent', 'ATLAS')
#     if RuntimeContext.engine and RuntimeContext.engine.memory:
#         try:
#             history = RuntimeContext.engine.memory.get_agent_history_for_web(agent_name, limit=20)
#             return jsonify({"status": "success", "history": history})
#         except Exception as e:
#             return jsonify({"status": "error", "message": str(e)})
#     return jsonify({"status": "error", "message": "Hafıza modülü hazır değil."})

# @app.route('/api/chat', methods=['POST'])
# def web_chat():
#     user_msg = request.form.get('message', '')
#     target_agent_req = request.form.get('target_agent', 'GENEL') 
    
#     uploaded_file = request.files.get('file')
#     auth_file = request.files.get('auth_frame') 
#     file_path = None

#     if uploaded_file and uploaded_file.filename != '':
#         filename = secure_filename(uploaded_file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         uploaded_file.save(file_path)

#     if not user_msg and not file_path:
#         return jsonify({"status": "error", "reply": "Mesaj içeriği boş."})

#     # Hafıza Temizleme Komutu
#     if user_msg.lower().strip() in ["hafızayı sil", "hafızayı temizle"]:
#         if RuntimeContext.engine and RuntimeContext.engine.memory:
#             RuntimeContext.engine.memory.clear_history()
#             return jsonify({"status": "success", "agent": "SİSTEM", "reply": "Hafıza başarıyla temizlendi."})

#     # --- KİMLİK DOĞRULAMA (Web Kamera Üzerinden) ---
#     identified_user = None
#     frame_present = False
    
#     if auth_file and RuntimeContext.security_instance:
#         try:
#             file_bytes = np.frombuffer(auth_file.read(), np.uint8)
#             frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             if frame is not None:
#                 frame_present = True
#                 identified_user = RuntimeContext.security_instance.check_static_frame(frame)
#         except Exception as e:
#             logger.error(f"Web Auth İşleme Hatası: {e}")

#     # Güvenlik Kararı (Security Decision)
#     if identified_user:
#         sec_result = ("ONAYLI", identified_user, None)
#     elif frame_present:
#         sec_result = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
#     else:
#         # Web üzerinden fiziksel kamera erişimi yoksa kısıtlı yetki ver
#         sec_result = ("SORGULAMA", {"name": "Web Kullanıcısı", "level": 1}, "KAMERA_YOK")

#     try:
#         # Grup Sohbeti Kontrolü
#         group_triggers = ["millet", "ekip", "herkes", "gençler", "arkadaşlar", "team", "tüm ekip", "hepiniz"]
#         is_group_call = target_agent_req == "GENEL" and any(t in user_msg.lower() for t in group_triggers)

#         if is_group_call and RuntimeContext.engine and RuntimeContext.loop:
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_team_response(user_msg, sec_result),
#                 RuntimeContext.loop
#             )
#             replies_list = future.result(timeout=120)
#             return jsonify({"status": "success", "replies": replies_list})
            
#         # Tekil Ajan Belirleme
#         final_agent = RuntimeContext.active_web_agent 
#         if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
#             final_agent = target_agent_req
#         else:
#             if RuntimeContext.engine:
#                 detected_agent = RuntimeContext.engine.determine_agent(user_msg)
#                 if detected_agent: final_agent = detected_agent
#                 else: final_agent = "ATLAS"

#         RuntimeContext.active_web_agent = final_agent 

#         if RuntimeContext.loop and RuntimeContext.loop.is_running():
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_response(final_agent, user_msg, sec_result, file_path=file_path), 
#                 RuntimeContext.loop
#             )
#             response_data = future.result(timeout=90)
            
#             # Seslendirme gerekiyorsa thread üzerinden çalıştır
#             if RuntimeContext.voice_mode_active:
#                  RuntimeContext.executor.submit(play_voice, response_data['content'], response_data['agent'], RuntimeContext.state_manager)

#             return jsonify({
#                 "status": "success", 
#                 "agent": response_data['agent'],
#                 "reply": response_data['content']
#             })
#         else:
#             return jsonify({"status": "error", "reply": "Lotus motoru şu an hazır değil."})
            
#     except Exception as e:
#         logger.error(f"Web Chat İşlem Hatası: {e}")
#         return jsonify({"status": "error", "reply": f"Sistem hatası oluştu: {str(e)}"})

# @app.route('/webhook', methods=['GET', 'POST'])
# def webhook_handler():
#     if request.method == 'GET':
#         verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "lotus_ai_guvenlik_tokeni")
#         mode = request.args.get("hub.mode")
#         token = request.args.get("hub.verify_token")
#         challenge = request.args.get("hub.challenge")
#         if mode == "subscribe" and token == verify_token:
#             return challenge, 200
#         return "Verification failed", 403
    
#     elif request.method == 'POST':
#         try:
#             data = request.json
#             parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
#             if parsed:
#                 RuntimeContext.msg_queue.put(parsed)
#                 return jsonify({"status": "ok"}), 200
#             return jsonify({"status": "ignored"}), 200
#         except Exception as e:
#             logger.error(f"Webhook Mesaj Hatası: {e}")
#             return jsonify({"status": "error"}), 500

# def run_flask():
#     """Flask sunucusunu güvenli şekilde başlatır."""
#     try:
#         import logging
#         log = logging.getLogger('werkzeug')
#         log.setLevel(logging.ERROR)
#         # Port çakışmalarını önlemek için kontroller eklenebilir
#         app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
#     except Exception as e:
#         logger.error(f"Flask Sunucu Hatası: {e}")

# # --- SES İŞLEMLERİ (TTS) ---
# try:
#     import edge_tts
# except ImportError:
#     logger.warning("edge_tts modülü bulunamadı. Bulut tabanlı ses pasif.")

# tts_model = None
# if Config.USE_XTTS:
#     try:
#         from TTS.api import TTS
#         import torch
#         if torch.cuda.is_available():
#             logger.info("🔊 XTTS (GPU) Modeli Yükleniyor...")
#             tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
#             logger.info("🔊 XTTS Kullanıma Hazır.")
#         else:
#             logger.warning("⚠️ CUDA bulunamadı, XTTS otomatik kapatıldı.")
#     except Exception as e:
#         logger.error(f"XTTS Başlatılamadı: {e}")

# async def edge_stream(text, voice):
#     """EdgeTTS ile bulut tabanlı asenkron ses sentezi."""
#     try:
#         comm = edge_tts.Communicate(text, voice)
#         data = b""
#         async for chunk in comm.stream():
#             if chunk["type"] == "audio":
#                 data += chunk["data"]
#         return data
#     except Exception as e:
#         logger.error(f"EdgeTTS Stream Hatası: {e}")
#         return None

# def play_voice(text, agent_name, state_mgr):
#     """Sesi çalan ana fonksiyon. Sistem durumunu SPEAKING yapar."""
#     if not text or not state_mgr: return
    
#     # Markdown ve özel karakter temizliği
#     clean = re.sub(r'#.*', '', text)
#     clean = clean.replace('*', '').replace('_', '').strip()
#     if not clean: return
    
#     state_mgr.set_state(SystemState.SPEAKING)
    
#     try:
#         if mixer.get_init() is None:
#             mixer.init()

#         mixer.music.unload()
#         # Config 2.4 üzerinden ajan ayarlarını al
#         agent_settings = Config.get_agent_settings(agent_name)
#         agent_data = AGENTS_CONFIG.get(agent_name, AGENTS_CONFIG.get("ATLAS", {}))
        
#         wav_path = str(Config.VOICES_DIR / f"{agent_name.lower()}.wav")
#         if not os.path.exists(wav_path):
#              wav_path = agent_data.get("voice_ref", "voices/atlas.wav")
             
#         edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")
        
#         use_xtts_now = Config.USE_XTTS and tts_model and os.path.exists(wav_path)
        
#         # 1. Öncelik: XTTS (Yerel/Gerçekçi)
#         if use_xtts_now:
#             try:
#                 output_path = "out.wav"
#                 tts_model.tts_to_file(text=clean, speaker_wav=wav_path, language="tr", file_path=output_path)
#                 mixer.music.load(output_path)
#             except Exception as e:
#                 logger.error(f"XTTS Hatası (EdgeTTS'e geçiliyor): {e}")
#                 use_xtts_now = False 
        
#         # 2. Öncelik: EdgeTTS (Hızlı/Bulut)
#         if not use_xtts_now:
#             try:
#                 # Asenkron fonksiyonu senkron içinde çalıştırma
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
#                 audio = loop.run_until_complete(edge_stream(clean, edge_voice))
#                 loop.close()
                
#                 if audio:
#                     mixer.music.load(io.BytesIO(audio))
#                 else:
#                     return
#             except Exception as e:
#                 logger.error(f"EdgeTTS Fallback Hatası: {e}")
#                 return
            
#         mixer.music.play()
        
#         while mixer.music.get_busy():
#             # Konuşmayı kesme kontrolü (Space veya Esc)
#             if keyboard.is_pressed('space') or keyboard.is_pressed('esc'): 
#                 mixer.music.stop()
#                 logger.info("🔇 Konuşma kullanıcı tarafından kesildi.")
#                 break
#             time.sleep(0.05)
            
#     except Exception as e:
#         logger.error(f"Ses Çalma İşlemi Başarısız: {e}")
#     finally:
#         state_mgr.set_state(SystemState.IDLE)

# # --- ANA ASYNC MOTOR DÖNGÜSÜ ---
# async def main_loop(mode):
#     """Sistemin ana kalbi: Tüm servisleri koordine eder."""
#     RuntimeContext.loop = asyncio.get_running_loop()

#     # 1. BAŞLANGIÇ BİLGİSİ
#     Config.set_provider_mode(mode)
#     print(f"{Colors.HEADER}--- {Config.PROJECT_NAME.upper()} SİSTEMİ v{Config.VERSION} ---{Colors.ENDC}")
#     print(f"{Colors.CYAN}⚙️ Donanım: {Config.GPU_INFO} | Sağlayıcı: {Config.AI_PROVIDER.upper()}{Colors.ENDC}")

#     # 2. SERVİSLERİ VE YÖNETİCİLERİ BAŞLAT
#     state_manager = SystemState()
#     RuntimeContext.state_manager = state_manager
#     memory_manager = MemoryManager()
    
#     camera_manager = CameraManager()
#     camera_manager.start()
    
#     security_manager = SecurityManager(camera_manager)
#     RuntimeContext.security_instance = security_manager
    
#     code_manager = CodeManager(Config.WORK_DIR)
#     sys_health_manager = SystemHealthManager()
#     finance_manager = FinanceManager()
#     accounting_manager = AccountingManager()
#     ops_manager = OperationsManager()
    
#     delivery_manager = DeliveryManager()
#     if Config.FINANCE_MODE:
#          logger.info("🛵 Paket Servis Modülü Aktif Edildi.")
#          delivery_manager.start_service()
    
#     # NLP ve Ajan Yapılandırması
#     nlp_manager = NLPManager()
#     poyraz_agent = PoyrazAgent(nlp_manager)
    
#     sidar_tools = {
#         'code': code_manager, 
#         'system': sys_health_manager, 
#         'security': security_manager,
#         'memory': memory_manager
#     }
#     sidar_agent = SidarAgent(sidar_tools)
    
#     # Tüm yönetici araçlarını birleştir
#     tools = {
#         "camera": camera_manager, 
#         "code": code_manager, 
#         "system": sys_health_manager,
#         "finance": finance_manager, 
#         "operations": ops_manager, 
#         "accounting": accounting_manager, 
#         "messaging": RuntimeContext.messaging_manager, 
#         "delivery": delivery_manager,
#         "nlp": nlp_manager, 
#         "poyraz_special": poyraz_agent, 
#         "sidar_special": sidar_agent,
#         "state": state_manager
#     }
    
#     if MEDIA_AVAILABLE:
#         tools['media'] = MediaManager()

#     # Ajan Motorunu (Engine) Başlat
#     RuntimeContext.engine = AgentEngine(memory_manager, tools)

#     # 3. WEB SUNUCUSU BAŞLATMA
#     if (Config.TEMPLATE_DIR / "index.html").exists():
#         flask_thread = threading.Thread(target=run_flask, daemon=True)
#         flask_thread.start()
#         logger.info(f"🌐 Web Dashboard Hazır: http://localhost:5000")
#         # Otomatik tarayıcı açma
#         threading.Timer(3.0, lambda: webbrowser.open("http://localhost:5000")).start()
#     else:
#         logger.error("❌ HATA: Dashboard dosyaları bulunamadı!")

#     # Mikrofon Hazırlığı
#     try: mixer.init()
#     except Exception as e: logger.warning(f"Ses kartı uyarısı: {e}")

#     r = sr.Recognizer()
#     mic = sr.Microphone()
    
#     with mic as source:
#          print("🎤 Ortam sesi kalibre ediliyor...")
#          r.adjust_for_ambient_noise(source, duration=1.0)

#     print(f"{Colors.GREEN}✅ {Config.PROJECT_NAME.upper()} TÜM SİSTEMLER AKTİF.{Colors.ENDC}")

#     # --- ANA DÖNGÜ (INFINITE LOOP) ---
#     while state_manager.is_running():
#         try:
#             current_time = time.time()

#             # 1. Otomatik Görev Kontrolü (Paket Servis - 30 sn'de bir)
#             if delivery_manager.is_selenium_active and int(current_time) % 30 == 0:
#                 order_alerts = delivery_manager.check_new_orders()
#                 if order_alerts:
#                     for alert in order_alerts:
#                         RuntimeContext.executor.submit(play_voice, f"Yeni bildirim: {alert}", "GAYA", state_manager)

#             # 2. Sesli Komut Dinleme
#             if RuntimeContext.voice_mode_active and state_manager.should_listen():
#                 state_manager.set_state(SystemState.LISTENING)
#                 user_input = ""
#                 audio_data = None 
                
#                 try:
#                     with mic as source:
#                         audio_data = await asyncio.to_thread(r.listen, source, timeout=3, phrase_time_limit=10)
#                         user_input = await asyncio.to_thread(r.recognize_google, audio_data, language="tr-TR")
#                 except (sr.WaitTimeoutError, sr.UnknownValueError): 
#                     pass 
                
#                 if user_input:
#                     print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")
                    
#                     # Güvenlik Analizi (Kamera + Ses)
#                     sec_result = security_manager.analyze_situation(audio_data=audio_data)

#                     # Hafıza Komutu
#                     if "hafızayı sil" in user_input.lower():
#                         memory_manager.clear_history()
#                         RuntimeContext.executor.submit(play_voice, "Hafıza temizlendi.", "ATLAS", state_manager)
#                         continue

#                     # Ajan Tespiti ve Yanıt
#                     detected_agent = RuntimeContext.engine.determine_agent(user_input)
#                     current_agent = detected_agent if detected_agent else "ATLAS"
                    
#                     state_manager.set_state(SystemState.THINKING)
#                     resp_data = await RuntimeContext.engine.get_response(current_agent, user_input, sec_result)

#                     print(f"🤖 {resp_data['agent']}: {resp_data['content']}")
#                     # Yanıtı Seslendir
#                     RuntimeContext.executor.submit(play_voice, resp_data['content'], resp_data['agent'], state_manager)
            
#             else:
#                 # Dinleme kapalıysa veya sistem meşgulse bekle
#                 if state_manager.get_state() not in [SystemState.THINKING, SystemState.SPEAKING]:
#                      state_manager.set_state(SystemState.IDLE)
#                 await asyncio.sleep(0.5)

#             await asyncio.sleep(0.05)

#         except Exception as main_err:
#             logger.error(f"ANA DÖNGÜ HATASI: {main_err}")
#             await asyncio.sleep(2)

#     # Sistem Kapanış
#     camera_manager.stop()
#     delivery_manager.stop_service()
#     RuntimeContext.executor.shutdown(wait=False)
#     logger.info("LotusAI güvenli bir şekilde kapatıldı.")

# def start_lotus_system(mode="online"):
#     """Sistemi başlatan ana giriş noktası."""
#     try:
#         # Windows uyumluluğu
#         if sys.platform == 'win32':
#              asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
#         asyncio.run(main_loop(mode))
#     except KeyboardInterrupt:
#         print(f"\n{Colors.YELLOW}[!] LotusAI kullanıcı tarafından kapatıldı.{Colors.ENDC}")
#     except Exception as e:
#         logger.critical(f"BAŞLATMA SIRASINDA KRİTİK HATA: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     # .env'den mod çekilebilir veya varsayılan online başlar
#     mode = os.getenv("AI_PROVIDER", "online")
#     start_lotus_system(mode)




# import asyncio
# import time
# import speech_recognition as sr
# from pygame import mixer
# import keyboard
# import threading
# import os
# import sys
# import re
# import io
# import queue
# import webbrowser
# import cv2
# import numpy as np
# import logging
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename

# # --- YAPILANDIRMA VE MODÜLLER ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("LotusSystem")

# try:
#     from config import Config
#     from core.system_state import SystemState
#     from core.memory import MemoryManager
#     from core.security import SecurityManager

#     from managers.camera import CameraManager
#     from managers.code_manager import CodeManager
#     from managers.system_health import SystemHealthManager 
#     from managers.finance import FinanceManager 
#     from managers.operations import OperationsManager
#     from managers.accounting import AccountingManager
#     from managers.messaging import MessagingManager
#     from managers.delivery import DeliveryManager
    
#     from managers.nlp import NLPManager
#     from agents.definitions import AGENTS_CONFIG
#     from agents.engine import AgentEngine
#     from agents.poyraz import PoyrazAgent
#     from agents.sidar import SidarAgent

# except ImportError as e:
#     logger.critical(f"KRİTİK HATA: Modüller yüklenirken sorun oluştu. Eksik dosya olabilir.\nHata: {e}")
#     sys.exit(1)

# # Media Manager Opsiyonel
# try:
#     from managers.media import MediaManager
#     MEDIA_AVAILABLE = True
# except ImportError:
#     MEDIA_AVAILABLE = False
#     logger.info("MediaManager bulunamadı, medya özellikleri devre dışı.")

# # --- RENKLER ---
# class Colors:
#     HEADER = '\033[95m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     YELLOW = '\033[93m'
#     CYAN = '\033[96m'

# # --- GLOBAL CONTEXT (RUNTIME) ---
# class RuntimeContext:
#     """Tüm global değişkenlerin merkezi yönetimi."""
#     msg_queue = queue.Queue()
#     messaging_manager = MessagingManager()
#     engine = None 
#     loop = None
#     security_instance = None 
#     state_manager = None
    
#     # Web Durumları
#     active_web_agent = "ATLAS"
#     voice_mode_active = False

# # --- FLASK VE WEB SİSTEMİ YAPILANDIRMASI ---
# # Dosya yollarının güvenli şekilde ayarlanması
# template_dir = str(Config.TEMPLATE_DIR) if hasattr(Config, 'TEMPLATE_DIR') else os.path.join(os.getcwd(), 'templates')
# static_dir = str(Config.STATIC_DIR) if hasattr(Config, 'STATIC_DIR') else os.path.join(os.getcwd(), 'static')
# upload_folder = str(Config.UPLOAD_DIR) if hasattr(Config, 'UPLOAD_DIR') else os.path.join(os.getcwd(), 'uploads')

# if not os.path.exists(upload_folder):
#     os.makedirs(upload_folder)

# app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
# app.config['UPLOAD_FOLDER'] = upload_folder

# # --- FLASK ROUTE TANIMLARI ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/toggle_voice', methods=['POST'])
# def toggle_voice_api():
#     """Web arayüzünden sesli dinlemeyi açıp kapatır."""
#     data = request.json
#     if data and 'active' in data:
#         RuntimeContext.voice_mode_active = data['active']
#     else:
#         RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active
        
#     status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
#     print(f"{Colors.YELLOW}🎙️ Mikrofon Modu: {status_msg}{Colors.ENDC}")
#     return jsonify({"status": "success", "active": RuntimeContext.voice_mode_active})

# @app.route('/api/chat_history', methods=['GET'])
# def get_chat_history():
#     agent_name = request.args.get('agent', 'ATLAS')
#     if RuntimeContext.engine and RuntimeContext.engine.memory:
#         try:
#             history = RuntimeContext.engine.memory.get_agent_history_for_web(agent_name, limit=20)
#             return jsonify({"status": "success", "history": history})
#         except Exception as e:
#             return jsonify({"status": "error", "message": str(e)})
#     return jsonify({"status": "error", "message": "Hafıza modülü hazır değil."})

# @app.route('/api/chat', methods=['POST'])
# def web_chat():
#     user_msg = request.form.get('message', '')
#     target_agent_req = request.form.get('target_agent', 'GENEL') 
    
#     uploaded_file = request.files.get('file')
#     auth_file = request.files.get('auth_frame') 
#     file_path = None

#     if uploaded_file and uploaded_file.filename != '':
#         filename = secure_filename(uploaded_file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         uploaded_file.save(file_path)

#     if not user_msg and not file_path:
#         return jsonify({"status": "error", "reply": "Mesaj içeriği boş."})

#     # Hafıza Temizleme Komutu
#     if "hafızayı sil" in user_msg.lower():
#         if RuntimeContext.engine and RuntimeContext.engine.memory:
#             RuntimeContext.engine.memory.clear_history()
#             return jsonify({"status": "success", "agent": "SİSTEM", "reply": "Hafıza başarıyla temizlendi."})

#     # --- KİMLİK DOĞRULAMA ---
#     identified_user = None
#     frame_present = False
    
#     if auth_file and RuntimeContext.security_instance:
#         try:
#             file_bytes = np.frombuffer(auth_file.read(), np.uint8)
#             frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             if frame is not None:
#                 frame_present = True
#                 identified_user = RuntimeContext.security_instance.check_static_frame(frame)
#         except Exception as e:
#             logger.error(f"Web Auth İşleme Hatası: {e}")

#     # Güvenlik Kararı (Security Decision)
#     if identified_user:
#         sec_result = ("ONAYLI", identified_user, None)
#     elif frame_present:
#         sec_result = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
#     else:
#         sec_result = ("SORGULAMA", {"name": "Bilinmiyor", "level": 0}, "KAMERA_YOK")

#     try:
#         # Grup Sohbeti Kontrolü
#         group_triggers = ["millet", "ekip", "herkes", "gençler", "arkadaşlar", "team", "tüm ekip", "hepiniz"]
#         is_group_call = target_agent_req == "GENEL" and any(t in user_msg.lower() for t in group_triggers)

#         if is_group_call and RuntimeContext.engine and RuntimeContext.loop:
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_team_response(user_msg, sec_result),
#                 RuntimeContext.loop
#             )
#             replies_list = future.result(timeout=120)
#             return jsonify({"status": "success", "replies": replies_list})
            
#         # Tekil Ajan Belirleme
#         final_agent = RuntimeContext.active_web_agent 
#         if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
#             final_agent = target_agent_req
#         else:
#             if RuntimeContext.engine:
#                 detected_agent = RuntimeContext.engine.determine_agent(user_msg)
#                 if detected_agent: final_agent = detected_agent
#                 else: final_agent = "ATLAS"

#         RuntimeContext.active_web_agent = final_agent 

#         if RuntimeContext.loop and RuntimeContext.loop.is_running():
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_response(final_agent, user_msg, sec_result, file_path=file_path), 
#                 RuntimeContext.loop
#             )
#             response_data = future.result(timeout=90)
#             return jsonify({
#                 "status": "success", 
#                 "agent": response_data['agent'],
#                 "reply": response_data['content']
#             })
#         else:
#             return jsonify({"status": "error", "reply": "Lotus motoru şu an asenkron döngüde değil."})
            
#     except Exception as e:
#         logger.error(f"Web Chat İşlem Hatası: {e}")
#         return jsonify({"status": "error", "reply": f"Sistem hatası oluştu: {str(e)}"})

# @app.route('/webhook', methods=['GET', 'POST'])
# def webhook_handler():
#     if request.method == 'GET':
#         verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "lotus_ai_guvenlik_tokeni")
#         mode = request.args.get("hub.mode")
#         token = request.args.get("hub.verify_token")
#         challenge = request.args.get("hub.challenge")
#         if mode == "subscribe" and token == verify_token:
#             return challenge, 200
#         return "Verification failed", 403
    
#     elif request.method == 'POST':
#         try:
#             data = request.json
#             parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
#             if parsed:
#                 RuntimeContext.msg_queue.put(parsed)
#                 return jsonify({"status": "ok"}), 200
#             return jsonify({"status": "ignored"}), 200
#         except Exception as e:
#             logger.error(f"Webhook Mesaj Hatası: {e}")
#             return jsonify({"status": "error"}), 500

# def run_flask():
#     try:
#         import logging
#         log = logging.getLogger('werkzeug')
#         log.setLevel(logging.ERROR)
#         app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
#     except Exception as e:
#         logger.error(f"Flask Sunucu Hatası: {e}")

# # --- SES İŞLEMLERİ (TTS) ---
# try:
#     import edge_tts
# except ImportError:
#     logger.warning("edge_tts modülü bulunamadı. Bulut tabanlı ses pasif.")

# tts_model = None
# if Config.USE_XTTS:
#     try:
#         from TTS.api import TTS
#         import torch
#         if torch.cuda.is_available():
#             logger.info("🔊 XTTS (GPU) Modeli Yükleniyor...")
#             tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
#             logger.info("🔊 XTTS Kullanıma Hazır.")
#         else:
#             logger.warning("⚠️ CUDA Desteklenmiyor, XTTS Devre Dışı. EdgeTTS kullanılacak.")
#     except Exception as e:
#         logger.error(f"XTTS Başlatılamadı: {e}")

# async def edge_stream(text, voice):
#     """EdgeTTS ile bulut tabanlı asenkron ses sentezi."""
#     try:
#         comm = edge_tts.Communicate(text, voice)
#         data = b""
#         async for chunk in comm.stream():
#             if chunk["type"] == "audio":
#                 data += chunk["data"]
#         return data
#     except Exception as e:
#         logger.error(f"EdgeTTS Stream Hatası: {e}")
#         return None

# def play_voice(text, agent_name, state_mgr):
#     """Sesi çalan ana fonksiyon. Sistem durumunu SPEAKING yapar."""
#     if not text or not state_mgr: return
#     # Gereksiz karakterleri temizle
#     clean = re.sub(r'#.*', '', text).replace('*', '').strip()
#     if not clean: return
    
#     state_mgr.set_state(SystemState.SPEAKING)
    
#     try:
#         if mixer.get_init() is None:
#             mixer.init()

#         mixer.music.unload()
#         agent_data = AGENTS_CONFIG.get(agent_name, AGENTS_CONFIG.get("ATLAS", {}))
#         wav_path = agent_data.get("voice_ref", "voices/atlas.wav")
#         edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")
        
#         use_xtts_now = Config.USE_XTTS and tts_model and os.path.exists(wav_path)
        
#         # 1. Öncelik: XTTS (Yerel/Kaliteli)
#         if use_xtts_now:
#             try:
#                 output_path = "out.wav"
#                 tts_model.tts_to_file(text=clean, speaker_wav=wav_path, language="tr", file_path=output_path)
#                 mixer.music.load(output_path)
#             except Exception as e:
#                 logger.error(f"XTTS Hatası (EdgeTTS'e geçiliyor): {e}")
#                 use_xtts_now = False 
        
#         # 2. Öncelik: EdgeTTS (Hızlı/Bulut)
#         if not use_xtts_now:
#             try:
#                 audio = asyncio.run(edge_stream(clean, edge_voice))
#                 if audio:
#                     mixer.music.load(io.BytesIO(audio))
#                 else:
#                     return
#             except Exception as e:
#                 logger.error(f"EdgeTTS Fallback Hatası: {e}")
#                 return
            
#         mixer.music.play()
        
#         while mixer.music.get_busy():
#             # Konuşmayı kesme kontrolü
#             if keyboard.is_pressed('space') or keyboard.is_pressed('esc'): 
#                 mixer.music.stop()
#                 break
#             time.sleep(0.05)
            
#     except Exception as e:
#         logger.error(f"Ses Çalma İşlemi Başarısız: {e}")
#     finally:
#         state_mgr.set_state(SystemState.IDLE)

# # --- ANA ASYNC MOTOR DÖNGÜSÜ ---
# async def main_loop(mode):
#     RuntimeContext.loop = asyncio.get_running_loop()

#     # 1. BAŞLANGIÇ BİLGİSİ
#     Config.set_provider_mode(mode)
#     print(f"{Colors.HEADER}--- {Config.PROJECT_NAME.upper()} SİSTEMİ: {mode.upper()} MOD AKTİF ---{Colors.ENDC}")
#     hw_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
#     print(f"{Colors.CYAN}⚙️ Donanım: {hw_info} | Yüz Tanıma: {Config.FACE_REC_MODEL.upper()}{Colors.ENDC}")

#     # 2. SERVİSLERİ VE YÖNETİCİLERİ BAŞLAT
#     logger.info("🛠️ Servis Yöneticileri Başlatılıyor...")
    
#     state_manager = SystemState()
#     RuntimeContext.state_manager = state_manager
#     memory_manager = MemoryManager()
    
#     camera_manager = CameraManager()
#     camera_manager.start()
    
#     security_manager = SecurityManager(camera_manager)
#     RuntimeContext.security_instance = security_manager
    
#     code_manager = CodeManager(Config.WORK_DIR)
#     sys_health_manager = SystemHealthManager()
#     finance_manager = FinanceManager()
#     accounting_manager = AccountingManager()
#     ops_manager = OperationsManager()
    
#     delivery_manager = DeliveryManager()
#     if Config.FINANCE_MODE:
#          logger.info("🛵 Paket Servis Modülü Aktif Edildi.")
#          delivery_manager.start_service()
    
#     # NLP ve Ajan Yapılandırması
#     nlp_manager = NLPManager()
#     poyraz_agent = PoyrazAgent(nlp_manager)
    
#     sidar_tools = {'code': code_manager, 'system': sys_health_manager, 'security': security_manager}
#     sidar_agent = SidarAgent(sidar_tools)
    
#     tools = {
#         "camera": camera_manager, "code": code_manager, "system": sys_health_manager,
#         "finance": finance_manager, "operations": ops_manager, "accounting": accounting_manager, 
#         "messaging": RuntimeContext.messaging_manager, "delivery": delivery_manager,
#         "nlp": nlp_manager, "poyraz_special": poyraz_agent, "sidar_special": sidar_agent
#     }
    
#     if MEDIA_AVAILABLE:
#         tools['media'] = MediaManager()

#     # Ajan Motorunu (Engine) Başlat
#     RuntimeContext.engine = AgentEngine(memory_manager, tools)

#     # 3. WEB SUNUCUSU VE ARAYÜZ
#     if os.path.exists(os.path.join(template_dir, "index.html")):
#         flask_thread = threading.Thread(target=run_flask)
#         flask_thread.daemon = True
#         flask_thread.start()
#         logger.info(f"🌐 Web Dashboard Hazır: http://localhost:5000")
#         threading.Timer(2.0, lambda: webbrowser.open("http://localhost:5000")).start()
#     else:
#         logger.error(f"HATA: {template_dir}/index.html dosyası bulunamadı. Web arayüzü başlatılamadı.")

#     # Mikrofon Hazırlığı ve Kalibrasyon
#     try: mixer.init()
#     except Exception as e: logger.warning(f"Ses kartı uyarısı: {e}")

#     r = sr.Recognizer()
#     mic = sr.Microphone()
    
#     with mic as source:
#          print("🎤 Ortam sesi kalibre ediliyor, lütfen sessiz olun...")
#          r.adjust_for_ambient_noise(source, duration=1.5)

#     active_agent = None
#     print(f"{Colors.GREEN}✅ {Config.PROJECT_NAME.upper()} KULLANIMA HAZIR.{Colors.ENDC}")

#     # --- ANA DÖNGÜ (INFINITE LOOP) ---
#     while state_manager.is_running():
#         try:
#             current_time = time.time()

#             # 1. Otomatik Görev Kontrolü (Paket Servis vb.)
#             if delivery_manager.is_selenium_active and int(current_time) % 30 == 0:
#                 order_alerts = delivery_manager.check_new_orders()
#                 if order_alerts:
#                     for alert in order_alerts:
#                         threading.Thread(target=play_voice, args=(f"Yeni bildirim var: {alert}", "GAYA", state_manager)).start()

#             # 2. Sesli Komut Dinleme
#             if RuntimeContext.voice_mode_active and state_manager.should_listen():
#                 state_manager.set_state(SystemState.LISTENING)
#                 user_input = ""
#                 audio_data = None 
                
#                 try:
#                     with mic as source:
#                         audio_data = await asyncio.to_thread(r.listen, source, timeout=3, phrase_time_limit=8)
#                         user_input = await asyncio.to_thread(r.recognize_google, audio_data, language="tr-TR")
#                 except (sr.WaitTimeoutError, sr.UnknownValueError): pass 
                
#                 if user_input:
#                     print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")
                    
#                     # Güvenlik ve Durum Analizi
#                     sec_result = security_manager.analyze_situation(audio_data=audio_data)

#                     # Hızlı Sistem Komutları
#                     if "hafızayı sil" in user_input.lower():
#                         memory_manager.clear_history()
#                         threading.Thread(target=play_voice, args=("Hafızayı temizledim.", "ATLAS", state_manager)).start()
#                         continue

#                     # Ajan Tespiti ve Yanıt Üretimi
#                     new_agent = RuntimeContext.engine.determine_agent(user_input)
#                     active_agent = new_agent if new_agent else (active_agent if active_agent else "ATLAS")
                    
#                     state_manager.set_state(SystemState.THINKING)
#                     resp_data = await RuntimeContext.engine.get_response(active_agent, user_input, sec_result)

#                     print(f"🤖 {resp_data['agent']}: {resp_data['content']}")
#                     # Yanıtı Seslendir (Thread içinde, asenkronu bloklamadan)
#                     threading.Thread(target=play_voice, args=(resp_data['content'], resp_data['agent'], state_manager)).start()
            
#             else:
#                 # Sistem Boşta (Idle) Durumu
#                 if state_manager.get_state() not in [SystemState.THINKING, SystemState.SPEAKING]:
#                      state_manager.set_state(SystemState.IDLE)
#                 await asyncio.sleep(0.3)

#             await asyncio.sleep(0.05)

#         except Exception as main_err:
#             logger.error(f"ANA DÖNGÜ HATASI: {main_err}")
#             await asyncio.sleep(1)

#     # Sistem Kapanış İşlemleri
#     camera_manager.stop()
#     delivery_manager.stop_service()
#     logger.info("Sistem güvenli bir şekilde kapatıldı.")

# def start_lotus_system(mode="online"):
#     """Sistemi başlatan ana giriş noktası (Main Entry Point)."""
#     try:
#         # Windows için asenkron döngü uyumluluğu
#         if sys.platform == 'win32':
#              asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
#         asyncio.run(main_loop(mode))
#     except KeyboardInterrupt:
#         print("\n[!] LotusAI kullanıcı tarafından kapatılıyor.")
#     except Exception as e:
#         logger.critical(f"BAŞLATMA SIRASINDA KRİTİK HATA: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     # Varsayılan mod 'online' (Config üzerinden Gemini kullanır)
#     start_lotus_system("online")



# import asyncio
# import time
# import speech_recognition as sr
# from pygame import mixer
# import keyboard
# import threading
# import os
# import sys
# import re
# import io
# import queue
# import webbrowser
# import cv2
# import numpy as np
# import logging
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename

# # --- YAPILANDIRMA VE MODÜLLER ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("LotusSystem")

# try:
#     from config import Config
#     from core.system_state import SystemState
#     from core.memory import MemoryManager
#     from core.security import SecurityManager

#     from managers.camera import CameraManager
#     from managers.code_manager import CodeManager
#     from managers.system_health import SystemHealthManager 
#     from managers.finance import FinanceManager 
#     from managers.operations import OperationsManager
#     from managers.accounting import AccountingManager
#     from managers.messaging import MessagingManager
#     from managers.delivery import DeliveryManager
    
#     from managers.nlp import NLPManager
#     from agents.definitions import AGENTS_CONFIG
#     from agents.engine import AgentEngine
#     from agents.poyraz import PoyrazAgent
#     from agents.sidar import SidarAgent

# except ImportError as e:
#     logger.critical(f"KRİTİK HATA: Modüller yüklenirken sorun oluştu. Eksik dosya olabilir.\nHata: {e}")
#     sys.exit(1)

# # Media Manager Opsiyonel
# try:
#     from managers.media import MediaManager
#     MEDIA_AVAILABLE = True
# except ImportError:
#     MEDIA_AVAILABLE = False
#     logger.info("MediaManager bulunamadı, medya özellikleri devre dışı.")

# # --- RENKLER ---
# class Colors:
#     HEADER = '\033[95m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     YELLOW = '\033[93m'
#     CYAN = '\033[96m'

# # --- GLOBAL CONTEXT (RUNTIME) ---
# class RuntimeContext:
#     """Tüm global değişkenlerin merkezi yönetimi."""
#     msg_queue = queue.Queue()
#     messaging_manager = MessagingManager()
#     engine = None 
#     loop = None
#     security_instance = None 
#     state_manager = None
    
#     # Web Durumları
#     active_web_agent = "ATLAS"
#     voice_mode_active = False

# # --- FLASK VE WEBHOOK AYARLARI ---
# app = Flask(__name__, 
#             template_folder=str(Config.TEMPLATE_DIR), 
#             static_folder=str(Config.STATIC_DIR))
# app.config['UPLOAD_FOLDER'] = str(Config.UPLOAD_DIR)

# # --- FLASK ROUTE TANIMLARI ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/toggle_voice', methods=['POST'])
# def toggle_voice_api():
#     """Web arayüzünden sesli dinlemeyi açıp kapatır."""
#     data = request.json
#     if data and 'active' in data:
#         RuntimeContext.voice_mode_active = data['active']
#     else:
#         RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active
        
#     status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
#     print(f"{Colors.YELLOW}🎙️ Mikrofon Modu: {status_msg}{Colors.ENDC}")
#     return jsonify({"status": "success", "active": RuntimeContext.voice_mode_active})

# @app.route('/api/chat_history', methods=['GET'])
# def get_chat_history():
#     agent_name = request.args.get('agent', 'ATLAS')
#     if RuntimeContext.engine and RuntimeContext.engine.memory:
#         try:
#             history = RuntimeContext.engine.memory.get_agent_history_for_web(agent_name, limit=20)
#             return jsonify({"status": "success", "history": history})
#         except Exception as e:
#             return jsonify({"status": "error", "message": str(e)})
#     return jsonify({"status": "error", "message": "Hafıza modülü hazır değil."})

# @app.route('/api/chat', methods=['POST'])
# def web_chat():
#     user_msg = request.form.get('message', '')
#     target_agent_req = request.form.get('target_agent', 'GENEL') 
    
#     uploaded_file = request.files.get('file')
#     auth_file = request.files.get('auth_frame') 
#     file_path = None

#     if uploaded_file and uploaded_file.filename != '':
#         filename = secure_filename(uploaded_file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         uploaded_file.save(file_path)

#     if not user_msg and not file_path:
#         return jsonify({"status": "error", "reply": "Mesaj içeriği boş."})

#     # Hafıza Temizleme Komutu
#     if "hafızayı sil" in user_msg.lower():
#         if RuntimeContext.engine and RuntimeContext.engine.memory:
#             RuntimeContext.engine.memory.clear_history()
#             return jsonify({"status": "success", "agent": "SİSTEM", "reply": "Hafıza başarıyla temizlendi."})

#     # --- KİMLİK DOĞRULAMA ---
#     identified_user = None
#     frame_present = False
    
#     if auth_file and RuntimeContext.security_instance:
#         try:
#             file_bytes = np.frombuffer(auth_file.read(), np.uint8)
#             frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             if frame is not None:
#                 frame_present = True
#                 identified_user = RuntimeContext.security_instance.check_static_frame(frame)
#         except Exception as e:
#             logger.error(f"Web Auth İşleme Hatası: {e}")

#     # Güvenlik Kararı
#     if identified_user:
#         sec_result = ("ONAYLI", identified_user, None)
#     elif frame_present:
#         sec_result = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
#     else:
#         sec_result = ("SORGULAMA", {"name": "Bilinmiyor", "level": 0}, "KAMERA_YOK")

#     try:
#         # Grup Sohbeti Kontrolü
#         group_triggers = ["millet", "ekip", "herkes", "gençler", "arkadaşlar", "team", "tüm ekip", "hepiniz"]
#         is_group_call = target_agent_req == "GENEL" and any(t in user_msg.lower() for t in group_triggers)

#         if is_group_call and RuntimeContext.engine and RuntimeContext.loop:
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_team_response(user_msg, sec_result),
#                 RuntimeContext.loop
#             )
#             replies_list = future.result(timeout=120)
#             return jsonify({"status": "success", "replies": replies_list})
            
#         # Tekil Ajan Belirleme
#         final_agent = RuntimeContext.active_web_agent 
#         if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
#             final_agent = target_agent_req
#         else:
#             if RuntimeContext.engine:
#                 detected_agent = RuntimeContext.engine.determine_agent(user_msg)
#                 if detected_agent: final_agent = detected_agent
#                 else: final_agent = "ATLAS"

#         RuntimeContext.active_web_agent = final_agent 

#         if RuntimeContext.loop and RuntimeContext.loop.is_running():
#             future = asyncio.run_coroutine_threadsafe(
#                 RuntimeContext.engine.get_response(final_agent, user_msg, sec_result, file_path=file_path), 
#                 RuntimeContext.loop
#             )
#             response_data = future.result(timeout=90)
#             return jsonify({
#                 "status": "success", 
#                 "agent": response_data['agent'],
#                 "reply": response_data['content']
#             })
#         else:
#             return jsonify({"status": "error", "reply": "Lotus motoru şu an asenkron döngüde değil."})
            
#     except Exception as e:
#         logger.error(f"Web Chat İşlem Hatası: {e}")
#         return jsonify({"status": "error", "reply": f"Sistem hatası oluştu: {str(e)}"})

# @app.route('/webhook', methods=['GET', 'POST'])
# def webhook_handler():
#     if request.method == 'GET':
#         verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "lotus_ai_guvenlik_tokeni")
#         mode = request.args.get("hub.mode")
#         token = request.args.get("hub.verify_token")
#         challenge = request.args.get("hub.challenge")
#         if mode == "subscribe" and token == verify_token:
#             return challenge, 200
#         return "Verification failed", 403
    
#     elif request.method == 'POST':
#         try:
#             data = request.json
#             parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
#             if parsed:
#                 RuntimeContext.msg_queue.put(parsed)
#                 return jsonify({"status": "ok"}), 200
#             return jsonify({"status": "ignored"}), 200
#         except Exception as e:
#             logger.error(f"Webhook Mesaj Hatası: {e}")
#             return jsonify({"status": "error"}), 500

# def run_flask():
#     try:
#         import logging
#         log = logging.getLogger('werkzeug')
#         log.setLevel(logging.ERROR)
#         app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
#     except Exception as e:
#         logger.error(f"Flask Sunucu Hatası: {e}")

# # --- SES İŞLEMLERİ (TTS) ---
# try:
#     import edge_tts
# except ImportError:
#     logger.warning("edge_tts modülü bulunamadı.")

# tts_model = None
# if Config.USE_XTTS:
#     try:
#         from TTS.api import TTS
#         import torch
#         if torch.cuda.is_available():
#             logger.info("🔊 XTTS (GPU) Modeli Yükleniyor...")
#             tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
#             logger.info("🔊 XTTS Kullanıma Hazır.")
#         else:
#             logger.warning("⚠️ CUDA Desteklenmiyor, XTTS Devre Dışı. EdgeTTS kullanılacak.")
#     except Exception as e:
#         logger.error(f"XTTS Başlatılamadı: {e}")

# async def edge_stream(text, voice):
#     """EdgeTTS ile bulut tabanlı asenkron ses sentezi."""
#     try:
#         comm = edge_tts.Communicate(text, voice)
#         data = b""
#         async for chunk in comm.stream():
#             if chunk["type"] == "audio":
#                 data += chunk["data"]
#         return data
#     except Exception as e:
#         logger.error(f"EdgeTTS Stream Hatası: {e}")
#         return None

# def play_voice(text, agent_name, state_mgr):
#     """Sesi çalan fonksiyon. Sistem durumunu SPEAKING yapar."""
#     if not text or not state_mgr: return
#     clean = re.sub(r'#.*', '', text).replace('*', '').strip()
#     if not clean: return
    
#     state_mgr.set_state(SystemState.SPEAKING)
    
#     try:
#         if mixer.get_init() is None:
#              mixer.init()

#         mixer.music.unload()
#         agent_data = AGENTS_CONFIG.get(agent_name, AGENTS_CONFIG.get("ATLAS", {}))
#         wav_path = agent_data.get("voice_ref", "voices/atlas.wav")
#         edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")
        
#         use_xtts_now = Config.USE_XTTS and tts_model and os.path.exists(wav_path)
        
#         # 1. XTTS (Yerel)
#         if use_xtts_now:
#             try:
#                 output_path = "out.wav"
#                 tts_model.tts_to_file(text=clean, speaker_wav=wav_path, language="tr", file_path=output_path)
#                 mixer.music.load(output_path)
#             except Exception as e:
#                 logger.error(f"XTTS Hatası (EdgeTTS'e geçiliyor): {e}")
#                 use_xtts_now = False 
        
#         # 2. EdgeTTS (Bulut Fallback)
#         if not use_xtts_now:
#             try:
#                 audio = asyncio.run(edge_stream(clean, edge_voice))
#                 if audio:
#                     mixer.music.load(io.BytesIO(audio))
#                 else:
#                     return
#             except Exception as e:
#                 logger.error(f"EdgeTTS Fallback Hatası: {e}")
#                 return
            
#         mixer.music.play()
        
#         while mixer.music.get_busy():
#             if keyboard.is_pressed('space') or keyboard.is_pressed('esc'): 
#                 mixer.music.stop()
#                 break
#             time.sleep(0.05)
            
#     except Exception as e:
#         logger.error(f"Ses Çalma İşlemi Başarısız: {e}")
#     finally:
#         state_mgr.set_state(SystemState.IDLE)

# # --- ANA ASYNC DÖNGÜSÜ ---
# async def main_loop(mode):
#     RuntimeContext.loop = asyncio.get_running_loop()

#     # 1. MOD VE DONANIM BİLGİSİ
#     Config.set_provider_mode(mode)
#     print(f"{Colors.HEADER}--- {Config.PROJECT_NAME.upper()}: {mode.upper()} MOD AKTİF ---{Colors.ENDC}")
#     print(f"{Colors.CYAN}⚙️ Donanım: {Config.GPU_INFO} | Yüz Tanıma: {Config.FACE_REC_MODEL.upper()}{Colors.ENDC}")

#     # 2. SERVİSLERİ BAŞLAT
#     logger.info("🛠️ Yöneticiler Başlatılıyor...")
    
#     state_manager = SystemState()
#     RuntimeContext.state_manager = state_manager
#     memory_manager = MemoryManager()
    
#     camera_manager = CameraManager()
#     camera_manager.start()
    
#     security_manager = SecurityManager(camera_manager)
#     RuntimeContext.security_instance = security_manager
    
#     code_manager = CodeManager(Config.WORK_DIR)
#     sys_health_manager = SystemHealthManager()
#     finance_manager = FinanceManager()
#     accounting_manager = AccountingManager()
#     ops_manager = OperationsManager()
    
#     delivery_manager = DeliveryManager()
#     if Config.FINANCE_MODE:
#          logger.info("🛵 Paket Servis Modülü Devrede...")
#          delivery_manager.start_service()
    
#     # NLP ve Ajanlar
#     nlp_manager = NLPManager()
#     poyraz_agent = PoyrazAgent(nlp_manager)
    
#     sidar_tools = {'code': code_manager, 'system': sys_health_manager, 'security': security_manager}
#     sidar_agent = SidarAgent(sidar_tools)
    
#     tools = {
#         "camera": camera_manager, "code": code_manager, "system": sys_health_manager,
#         "finance": finance_manager, "operations": ops_manager, "accounting": accounting_manager, 
#         "messaging": RuntimeContext.messaging_manager, "delivery": delivery_manager,
#         "nlp": nlp_manager, "poyraz_special": poyraz_agent, "sidar_special": sidar_agent
#     }
    
#     if MEDIA_AVAILABLE:
#         tools['media'] = MediaManager()

#     RuntimeContext.engine = AgentEngine(memory_manager, tools)

#     # 3. WEB SUNUCUSU BAŞLATMA
#     if (Config.TEMPLATE_DIR / "index.html").exists():
#         flask_thread = threading.Thread(target=run_flask)
#         flask_thread.daemon = True
#         flask_thread.start()
#         logger.info(f"🌐 Web Arayüzü Hazır: http://localhost:5000")
#         threading.Timer(2.0, lambda: webbrowser.open("http://localhost:5000")).start()
#     else:
#         logger.error(f"HATA: {Config.TEMPLATE_DIR}/index.html bulunamadı.")

#     # Mikrofon Hazırlığı
#     try: mixer.init()
#     except Exception as e: logger.warning(f"Ses kartı erişim uyarısı: {e}")

#     r = sr.Recognizer()
#     mic = sr.Microphone()
    
#     with mic as source:
#          print("🎤 Ortam sesi kalibre ediliyor...")
#          r.adjust_for_ambient_noise(source, duration=1.5)

#     active_agent = None
#     print(f"{Colors.GREEN}✅ {Config.PROJECT_NAME.upper()} TAMAMEN HAZIR ({mode.upper()}){Colors.ENDC}")

#     # --- ANA DÖNGÜ ---
#     while state_manager.is_running():
#         try:
#             current_time = time.time()

#             # Paket Servis Kontrolü (30 sn'de bir)
#             if delivery_manager.is_selenium_active and int(current_time) % 30 == 0:
#                 order_alerts = delivery_manager.check_new_orders()
#                 if order_alerts:
#                     for alert in order_alerts:
#                         threading.Thread(target=play_voice, args=(f"Yeni bildirim: {alert}", "GAYA", state_manager)).start()

#             # MİKROFON DİNLEME
#             if RuntimeContext.voice_mode_active and state_manager.should_listen():
#                 state_manager.set_state(SystemState.LISTENING)
#                 user_input = ""
#                 audio_data = None 
                
#                 try:
#                     with mic as source:
#                         audio_data = await asyncio.to_thread(r.listen, source, timeout=3, phrase_time_limit=8)
#                         user_input = await asyncio.to_thread(r.recognize_google, audio_data, language="tr-TR")
#                 except (sr.WaitTimeoutError, sr.UnknownValueError): pass 
                
#                 if user_input:
#                     print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")
#                     sec_result = security_manager.analyze_situation(audio_data=audio_data)

#                     if "hafızayı sil" in user_input.lower():
#                         memory_manager.clear_history()
#                         threading.Thread(target=play_voice, args=("Hafızayı temizledim.", "ATLAS", state_manager)).start()
#                         continue

#                     # Ajan Tespiti
#                     new_agent = RuntimeContext.engine.determine_agent(user_input)
#                     active_agent = new_agent if new_agent else (active_agent if active_agent else "ATLAS")
                    
#                     state_manager.set_state(SystemState.THINKING)
#                     resp_data = await RuntimeContext.engine.get_response(active_agent, user_input, sec_result)

#                     print(f"🤖 {resp_data['agent']}: {resp_data['content']}")
#                     threading.Thread(target=play_voice, args=(resp_data['content'], resp_data['agent'], state_manager)).start()
            
#             else:
#                 if state_manager.get_state() not in [SystemState.THINKING, SystemState.SPEAKING]:
#                      state_manager.set_state(SystemState.IDLE)
#                 await asyncio.sleep(0.3)

#             await asyncio.sleep(0.05)

#         except Exception as main_err:
#             logger.error(f"ANA DÖNGÜ HATASI: {main_err}")
#             await asyncio.sleep(1)

#     # Kapanış
#     camera_manager.stop()
#     delivery_manager.stop_service()

# def start_lotus_system(mode="online"):
#     """Sistemi başlatan ana giriş noktası."""
#     try:
#         if sys.platform == 'win32':
#              asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
#         asyncio.run(main_loop(mode))
#     except KeyboardInterrupt:
#         print("\n[!] LotusAI güvenli bir şekilde kapatılıyor.")
#     except Exception as e:
#         logger.critical(f"BAŞLATMA SIRASINDA KRİTİK HATA: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     start_lotus_system("online")








# # import asyncio
# # import time
# # import speech_recognition as sr
# # from pygame import mixer
# # import keyboard
# # import threading
# # import os
# # import sys
# # import re
# # import io
# # import queue
# # import webbrowser
# # import cv2
# # import numpy as np
# # import logging
# # from flask import Flask, request, jsonify, render_template
# # from werkzeug.utils import secure_filename

# # # --- YAPILANDIRMA VE MODÜLLER ---
# # # Logging ayarları
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger("LotusSystem")

# # try:
# #     from config import Config
# #     from core.system_state import SystemState
# #     from core.memory import MemoryManager
# #     from core.security import SecurityManager

# #     from managers.camera import CameraManager
# #     from managers.code_manager import CodeManager
# #     from managers.system_health import SystemHealthManager 
# #     from managers.finance import FinanceManager 
# #     from managers.operations import OperationsManager
# #     from managers.accounting import AccountingManager
# #     from managers.messaging import MessagingManager
# #     from managers.delivery import DeliveryManager
    
# #     from managers.nlp import NLPManager
# #     from agents.definitions import AGENTS_CONFIG
# #     from agents.engine import AgentEngine
# #     from agents.poyraz import PoyrazAgent
# #     from agents.sidar import SidarAgent

# # except ImportError as e:
# #     logger.critical(f"KRİTİK HATA: Modüller yüklenirken sorun oluştu. Eksik dosya olabilir.\nHata: {e}")
# #     sys.exit(1)

# # # Media Manager Opsiyonel
# # try:
# #     from managers.media import MediaManager
# #     MEDIA_AVAILABLE = True
# # except ImportError:
# #     MEDIA_AVAILABLE = False
# #     logger.info("MediaManager bulunamadı, medya özellikleri devre dışı.")

# # # --- RENKLER ---
# # class Colors:
# #     HEADER = '\033[95m'
# #     BLUE = '\033[94m'
# #     GREEN = '\033[92m'
# #     FAIL = '\033[91m'
# #     ENDC = '\033[0m'
# #     YELLOW = '\033[93m'
# #     CYAN = '\033[96m'

# # # --- GLOBAL CONTEXT (RUNTIME) ---
# # class RuntimeContext:
# #     """Tüm global değişkenlerin merkezi yönetimi."""
# #     msg_queue = queue.Queue()
# #     messaging_manager = MessagingManager()
# #     engine = None 
# #     loop = None
# #     security_instance = None 
# #     state_manager = None
    
# #     # Web Durumları
# #     active_web_agent = "ATLAS"
# #     voice_mode_active = False

# # # --- FLASK VE WEBHOOK AYARLARI ---
# # template_dir = os.path.abspath(os.path.join(os.getcwd(), 'templates'))
# # static_dir = os.path.abspath(os.path.join(os.getcwd(), 'static'))
# # upload_folder = Config.UPLOAD_DIR if hasattr(Config, 'UPLOAD_DIR') else os.path.abspath(os.path.join(os.getcwd(), 'uploads'))

# # if not os.path.exists(upload_folder):
# #     os.makedirs(upload_folder)

# # app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
# # app.config['UPLOAD_FOLDER'] = upload_folder

# # # --- FLASK ROUTE TANIMLARI ---
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/api/toggle_voice', methods=['POST'])
# # def toggle_voice_api():
# #     """Web arayüzünden sesli dinlemeyi açıp kapatır."""
# #     data = request.json
# #     if data and 'active' in data:
# #         RuntimeContext.voice_mode_active = data['active']
# #     else:
# #         RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active
        
# #     status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
# #     print(f"{Colors.YELLOW}🎙️ Mikrofon Modu: {status_msg}{Colors.ENDC}")
# #     return jsonify({"status": "success", "active": RuntimeContext.voice_mode_active})

# # @app.route('/api/chat_history', methods=['GET'])
# # def get_chat_history():
# #     agent_name = request.args.get('agent', 'ATLAS')
# #     if RuntimeContext.engine and RuntimeContext.engine.memory:
# #         try:
# #             history = RuntimeContext.engine.memory.get_agent_history_for_web(agent_name, limit=20)
# #             return jsonify({"status": "success", "history": history})
# #         except Exception as e:
# #             return jsonify({"status": "error", "message": str(e)})
# #     return jsonify({"status": "error", "message": "Hafıza modülü hazır değil."})

# # @app.route('/api/chat', methods=['POST'])
# # def web_chat():
# #     user_msg = request.form.get('message', '')
# #     target_agent_req = request.form.get('target_agent', 'GENEL') 
    
# #     uploaded_file = request.files.get('file')
# #     auth_file = request.files.get('auth_frame') 
# #     file_path = None

# #     if uploaded_file and uploaded_file.filename != '':
# #         filename = secure_filename(uploaded_file.filename)
# #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         uploaded_file.save(file_path)

# #     if not user_msg and not file_path:
# #         return jsonify({"status": "error", "reply": "Mesaj içeriği boş."})

# #     # Hafıza Temizleme Komutu
# #     if "hafızayı sil" in user_msg.lower():
# #         if RuntimeContext.engine and RuntimeContext.engine.memory:
# #             RuntimeContext.engine.memory.clear_history()
# #             return jsonify({"status": "success", "agent": "SİSTEM", "reply": "Hafıza başarıyla temizlendi."})

# #     # --- KİMLİK DOĞRULAMA ---
# #     identified_user = None
# #     frame_present = False
    
# #     if auth_file and RuntimeContext.security_instance:
# #         try:
# #             file_bytes = np.frombuffer(auth_file.read(), np.uint8)
# #             frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# #             if frame is not None:
# #                 frame_present = True
# #                 identified_user = RuntimeContext.security_instance.check_static_frame(frame)
# #         except Exception as e:
# #             logger.error(f"Web Auth İşleme Hatası: {e}")

# #     # Güvenlik Kararı
# #     if identified_user:
# #         sec_result = ("ONAYLI", identified_user, None)
# #     elif frame_present:
# #         sec_result = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
# #     else:
# #         sec_result = ("SORGULAMA", {"name": "Bilinmiyor", "level": 0}, "KAMERA_YOK")

# #     try:
# #         # Grup Sohbeti Kontrolü
# #         group_triggers = ["millet", "ekip", "herkes", "gençler", "arkadaşlar", "team", "tüm ekip", "hepiniz"]
# #         is_group_call = target_agent_req == "GENEL" and any(t in user_msg.lower() for t in group_triggers)

# #         if is_group_call and RuntimeContext.engine and RuntimeContext.loop:
# #             future = asyncio.run_coroutine_threadsafe(
# #                 RuntimeContext.engine.get_team_response(user_msg, sec_result),
# #                 RuntimeContext.loop
# #             )
# #             replies_list = future.result(timeout=120)
# #             return jsonify({"status": "success", "replies": replies_list})
            
# #         # Tekil Ajan Belirleme
# #         final_agent = RuntimeContext.active_web_agent 
# #         if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
# #             final_agent = target_agent_req
# #         else:
# #             if RuntimeContext.engine:
# #                 detected_agent = RuntimeContext.engine.determine_agent(user_msg)
# #                 if detected_agent: final_agent = detected_agent
# #                 else: final_agent = "ATLAS"

# #         RuntimeContext.active_web_agent = final_agent 

# #         if RuntimeContext.loop and RuntimeContext.loop.is_running():
# #             future = asyncio.run_coroutine_threadsafe(
# #                 RuntimeContext.engine.get_response(final_agent, user_msg, sec_result, file_path=file_path), 
# #                 RuntimeContext.loop
# #             )
# #             response_data = future.result(timeout=90)
# #             return jsonify({
# #                 "status": "success", 
# #                 "agent": response_data['agent'],
# #                 "reply": response_data['content']
# #             })
# #         else:
# #             return jsonify({"status": "error", "reply": "Lotus motoru şu an asenkron döngüde değil."})
            
# #     except Exception as e:
# #         logger.error(f"Web Chat İşlem Hatası: {e}")
# #         return jsonify({"status": "error", "reply": f"Sistem hatası oluştu: {str(e)}"})

# # @app.route('/webhook', methods=['GET', 'POST'])
# # def webhook_handler():
# #     if request.method == 'GET':
# #         verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "lotus_ai_guvenlik_tokeni")
# #         mode = request.args.get("hub.mode")
# #         token = request.args.get("hub.verify_token")
# #         challenge = request.args.get("hub.challenge")
# #         if mode == "subscribe" and token == verify_token:
# #             return challenge, 200
# #         return "Verification failed", 403
    
# #     elif request.method == 'POST':
# #         try:
# #             data = request.json
# #             parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
# #             if parsed:
# #                 RuntimeContext.msg_queue.put(parsed)
# #                 return jsonify({"status": "ok"}), 200
# #             return jsonify({"status": "ignored"}), 200
# #         except Exception as e:
# #             logger.error(f"Webhook Mesaj Hatası: {e}")
# #             return jsonify({"status": "error"}), 500

# # def run_flask():
# #     try:
# #         import logging
# #         log = logging.getLogger('werkzeug')
# #         log.setLevel(logging.ERROR)
# #         app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
# #     except Exception as e:
# #         logger.error(f"Flask Sunucu Hatası: {e}")

# # # --- SES İŞLEMLERİ (TTS) ---
# # try:
# #     import edge_tts
# # except ImportError:
# #     logger.warning("edge_tts modülü bulunamadı.")

# # tts_model = None
# # if Config.USE_XTTS:
# #     try:
# #         from TTS.api import TTS
# #         import torch
# #         if torch.cuda.is_available():
# #             print(f"{Colors.BLUE}🔊 XTTS (GPU) Modeli Yükleniyor...{Colors.ENDC}")
# #             tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
# #             print(f"{Colors.BLUE}🔊 XTTS Kullanıma Hazır.{Colors.ENDC}")
# #         else:
# #             print(f"{Colors.YELLOW}⚠️ CUDA Desteklenmiyor, XTTS Devre Dışı. EdgeTTS kullanılacak.{Colors.ENDC}")
# #     except Exception as e:
# #         logger.error(f"XTTS Başlatılamadı: {e}")

# # async def edge_stream(text, voice):
# #     """EdgeTTS ile bulut tabanlı asenkron ses sentezi."""
# #     try:
# #         comm = edge_tts.Communicate(text, voice)
# #         data = b""
# #         async for chunk in comm.stream():
# #             if chunk["type"] == "audio":
# #                 data += chunk["data"]
# #         return data
# #     except Exception as e:
# #         logger.error(f"EdgeTTS Stream Hatası: {e}")
# #         return None

# # def play_voice(text, agent_name, state_mgr):
# #     """
# #     Sesi fiziksel olarak çalan fonksiyon (Threading içinde çalışır).
# #     Çalma sırasında sistem durumunu SPEAKING yapar.
# #     """
# #     if not text or not state_mgr: return
# #     clean = re.sub(r'#.*', '', text).replace('*', '').strip()
# #     if not clean: return
    
# #     state_mgr.set_state(SystemState.SPEAKING)
    
# #     try:
# #         if mixer.get_init() is None:
# #              mixer.init()

# #         mixer.music.unload()
# #         agent_data = AGENTS_CONFIG.get(agent_name, AGENTS_CONFIG.get("ATLAS", {}))
# #         wav_path = agent_data.get("voice_ref", "voices/atlas.wav")
# #         edge_voice = agent_data.get("edge", "tr-TR-AhmetNeural")
        
# #         use_xtts_now = Config.USE_XTTS and tts_model and os.path.exists(wav_path)
        
# #         # 1. XTTS (Yerel)
# #         if use_xtts_now:
# #             try:
# #                 output_path = "out.wav"
# #                 tts_model.tts_to_file(text=clean, speaker_wav=wav_path, language="tr", file_path=output_path)
# #                 mixer.music.load(output_path)
# #             except Exception as e:
# #                 logger.error(f"XTTS Hatası (EdgeTTS'e geçiliyor): {e}")
# #                 use_xtts_now = False 
        
# #         # 2. EdgeTTS (Bulut Fallback)
# #         if not use_xtts_now:
# #             try:
# #                 audio = asyncio.run(edge_stream(clean, edge_voice))
# #                 if audio:
# #                     mixer.music.load(io.BytesIO(audio))
# #                 else:
# #                     return
# #             except Exception as e:
# #                 logger.error(f"EdgeTTS Fallback Hatası: {e}")
# #                 return
            
# #         mixer.music.play()
        
# #         # Çalma bitene kadar kontrol döngüsü
# #         while mixer.music.get_busy():
# #             # Kullanıcı konuşmayı kesmek isterse (SPACE veya ESC)
# #             if keyboard.is_pressed('space') or keyboard.is_pressed('esc'): 
# #                 mixer.music.stop()
# #                 break
# #             time.sleep(0.05)
            
# #     except Exception as e:
# #         logger.error(f"Ses Çalma İşlemi Başarısız: {e}")
# #     finally:
# #         state_mgr.set_state(SystemState.IDLE)

# # # --- ANA ASYNC DÖNGÜSÜ ---
# # async def main_loop(mode):
# #     RuntimeContext.loop = asyncio.get_running_loop()

# #     # 1. MOD BİLGİLENDİRME
# #     if mode == "online":
# #         Config.AI_PROVIDER = "gemini"
# #         print(f"{Colors.HEADER}--- LOTUS AI: ONLINE MOD (GEMINI) AKTİF ---{Colors.ENDC}")
# #     else:
# #         Config.AI_PROVIDER = "ollama"
# #         print(f"{Colors.HEADER}--- LOTUS AI: LOCAL MOD (OLLAMA) AKTİF ---{Colors.ENDC}")

# #     hw_info = "GPU (CUDA)" if Config.USE_GPU else "CPU (Standart)"
# #     print(f"{Colors.CYAN}⚙️ Donanım: {hw_info} | Yüz Tanıma: {Config.FACE_REC_MODEL.upper()}{Colors.ENDC}")

# #     # 2. SERVİSLERİ BAŞLAT
# #     print(f"{Colors.BLUE}🛠️ Yöneticiler Başlatılıyor...{Colors.ENDC}")
    
# #     state_manager = SystemState()
# #     RuntimeContext.state_manager = state_manager
# #     memory_manager = MemoryManager()
    
# #     camera_manager = CameraManager()
# #     camera_manager.start()
    
# #     security_manager = SecurityManager(camera_manager)
# #     RuntimeContext.security_instance = security_manager
    
# #     code_manager = CodeManager(Config.WORK_DIR)
# #     sys_health_manager = SystemHealthManager()
# #     finance_manager = FinanceManager()
# #     accounting_manager = AccountingManager()
# #     ops_manager = OperationsManager()
    
# #     delivery_manager = DeliveryManager()
# #     if Config.FINANCE_MODE:
# #          print(f"{Colors.BLUE}🛵 Paket Servis Modülü Devrede...{Colors.ENDC}")
# #          delivery_manager.start_service()
    
# #     # NLP ve Ajanlar
# #     nlp_manager = NLPManager()
# #     print(f"{Colors.BLUE}🧠 Ajan Motoru (Engine) Yapılandırılıyor...{Colors.ENDC}")

# #     poyraz_agent = PoyrazAgent(nlp_manager)
    
# #     sidar_tools = {
# #         'code': code_manager,
# #         'system': sys_health_manager,
# #         'security': security_manager
# #     }
# #     sidar_agent = SidarAgent(sidar_tools)
    
# #     tools = {
# #         "camera": camera_manager, 
# #         "code": code_manager, 
# #         "system": sys_health_manager,
# #         "finance": finance_manager, 
# #         "operations": ops_manager,
# #         "accounting": accounting_manager, 
# #         "messaging": RuntimeContext.messaging_manager,
# #         "delivery": delivery_manager,
# #         "nlp": nlp_manager,
# #         "poyraz_special": poyraz_agent,
# #         "sidar_special": sidar_agent
# #     }
    
# #     if MEDIA_AVAILABLE:
# #         tools['media'] = MediaManager()

# #     RuntimeContext.engine = AgentEngine(memory_manager, tools)

# #     # 3. WEB SUNUCUSU BAŞLATMA
# #     if os.path.exists(os.path.join(template_dir, "index.html")):
# #         flask_thread = threading.Thread(target=run_flask)
# #         flask_thread.daemon = True
# #         flask_thread.start()
# #         print(f"{Colors.GREEN}🌐 Web Arayüzü Hazır: http://localhost:5000 {Colors.ENDC}")
# #         threading.Timer(2.0, lambda: webbrowser.open("http://localhost:5000")).start()
# #     else:
# #         logger.error("HATA: templates/index.html bulunamadı. Web arayüzü başlatılamıyor.")

# #     # Mikrofon Hazırlığı
# #     try:
# #         mixer.init()
# #     except Exception as e:
# #         logger.warning(f"Ses kartı erişim uyarısı: {e}")

# #     r = sr.Recognizer()
# #     mic = sr.Microphone()
    
# #     with mic as source:
# #          print("🎤 Ortam sesi kalibre ediliyor, lütfen bekleyin...")
# #          r.adjust_for_ambient_noise(source, duration=1.5)

# #     active_agent = None
# #     sec_result = ("BEKLEME", None, None)
    
# #     print(f"{Colors.GREEN}✅ LOTUS SİSTEMİ TAMAMEN HAZIR ({mode.upper()}){Colors.ENDC}")
# #     print(f"{Colors.YELLOW}🛑 Mikrofon Web Arayüzünden Aktifleştirilmeyi Bekliyor...{Colors.ENDC}")

# #     # --- ANA DÖNGÜ ---
# #     while state_manager.is_running():
# #         try:
# #             current_time = time.time()

# #             # Kuyruktaki Mesajlar (Webhook vb.)
# #             try:
# #                 incoming_api_msg = RuntimeContext.msg_queue.get_nowait()
# #                 # Gelecekte burada API mesajları işlenebilir
# #             except queue.Empty: pass

# #             # Paket Servis Kontrolü (30 sn'de bir)
# #             if delivery_manager.is_selenium_active and int(current_time) % 30 == 0:
# #                 order_alerts = delivery_manager.check_new_orders()
# #                 if order_alerts:
# #                     for alert in order_alerts:
# #                         threading.Thread(target=play_voice, args=(f"Yeni bildirim: {alert}", "GAYA", state_manager)).start()

# #             # MİKROFON DİNLEME (Sadece Voice Mode Aktifse)
# #             if RuntimeContext.voice_mode_active and state_manager.should_listen():
# #                 state_manager.set_state(SystemState.LISTENING)
# #                 user_input = ""
# #                 audio_data = None 
                
# #                 try:
# #                     with mic as source:
# #                         # Dinleme Limitleri
# #                         audio_data = await asyncio.to_thread(r.listen, source, timeout=3, phrase_time_limit=8)
# #                         user_input = await asyncio.to_thread(r.recognize_google, audio_data, language="tr-TR")
# #                 except (sr.WaitTimeoutError, sr.UnknownValueError):
# #                     pass 
# #                 except Exception as e:
# #                     logger.debug(f"Dinleme hatası: {e}")
                
# #                 if user_input:
# #                     print(f"{Colors.CYAN}>> KULLANICI: {user_input}{Colors.ENDC}")

# #                     # Güvenlik Analizi
# #                     sec_result = security_manager.analyze_situation(audio_data=audio_data)

# #                     # Hızlı Komut Kontrolü
# #                     if "hafızayı sil" in user_input.lower():
# #                         memory_manager.clear_history()
# #                         threading.Thread(target=play_voice, args=("Tüm hafızayı temizledim.", "ATLAS", state_manager)).start()
# #                         continue

# #                     # Ajan Tespiti ve Yanıt Üretimi
# #                     new_agent = RuntimeContext.engine.determine_agent(user_input)
# #                     if new_agent: active_agent = new_agent
# #                     elif not active_agent: active_agent = "ATLAS"
                    
# #                     state_manager.set_state(SystemState.THINKING)
# #                     resp_data = await RuntimeContext.engine.get_response(active_agent, user_input, sec_result)

# #                     print(f"🤖 {resp_data['agent']}: {resp_data['content']}")
                    
# #                     # Yanıtı Seslendir
# #                     threading.Thread(target=play_voice, args=(resp_data['content'], resp_data['agent'], state_manager)).start()
            
# #             else:
# #                 # Sistem Boşta (Idle) Durumu
# #                 if state_manager.get_state() not in [SystemState.THINKING, SystemState.SPEAKING]:
# #                      state_manager.set_state(SystemState.IDLE)
                
# #                 await asyncio.sleep(0.3)

# #             await asyncio.sleep(0.05)

# #         except Exception as main_err:
# #             logger.error(f"ANA DÖNGÜ HATASI: {main_err}")
# #             await asyncio.sleep(1)

# #     # Kapanış
# #     print("Lotus kapatılıyor, servisler durduruluyor...")
# #     camera_manager.stop()
# #     ops_manager.stop_service()
# #     delivery_manager.stop_service()

# # def start_lotus_system(mode="online"):
# #     """Sistemi başlatan ana giriş noktası."""
# #     try:
# #         # Windows Asyncio Uyumluluğu
# #         if sys.platform == 'win32':
# #              asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
# #         asyncio.run(main_loop(mode))
# #     except KeyboardInterrupt:
# #         print("\n[!] Kullanıcı kesmesi algılandı. LotusAI güvenli bir şekilde kapatılıyor.")
# #     except Exception as e:
# #         logger.critical(f"BAŞLATMA SIRASINDA KRİTİK HATA: {e}")
# #         import traceback
# #         traceback.print_exc()

# # if __name__ == "__main__":
# #     # Varsayılan olarak online (Gemini) modunda başla
# #     start_lotus_system("online")