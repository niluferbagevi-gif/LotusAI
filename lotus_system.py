"""
LotusAI Ana Sistem Motoru
Sürüm: 2.6.0 (Erişim seviyesi ve sürüm senkronu)
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
from config import Config, AccessLevel
from core.utils import setup_logging, patch_transformers, ignore_stderr
from core.runtime import RuntimeContext
from core.audio import init_audio_system, play_voice
from core.system_state import SystemState, SystemStateManager
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
from managers.github_manager import GithubManager  # <--- EKLENDİ

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
    SPEECH_LANGUAGE: str = os.getenv("STT_LANGUAGE", "tr-TR")
    
    # Sipariş kontrolü
    ORDER_CHECK_INTERVAL: int = 30  # saniye
    
    # Döngü bekleme süreleri
    IDLE_SLEEP: float = 0.5
    LOOP_SLEEP: float = 0.05
    ERROR_SLEEP: float = 2.0
    
    # Web dashboard
    DASHBOARD_PORT: int = int(os.getenv("FLASK_PORT", 5000))
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
    - Erişim seviyesi kontrolü (OpenClaw stili)
    """
    
    def __init__(self, mode: str = "online", access_level: str = "sandbox"):
        """
        Sistem başlatıcı
        
        Args:
            mode: 'online' (gemini) veya 'local' (ollama)
            access_level: 'restricted', 'sandbox', 'full'
        """
        self.mode = mode
        self.access_level = access_level
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Managerlar
        self.state_manager: Optional[SystemStateManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.delivery_manager: Optional[DeliveryManager] = None
        self.github_manager: Optional[GithubManager] = None # <--- EKLENDİ
        
        # Agents
        self.engine: Optional[AgentEngine] = None
        
        # Ses tanıma
        self.recognizer: Optional[sr.Recognizer] = None
        self.microphone: Optional[sr.Microphone] = None
        
        logger.info(f"LotusSystem başlatılıyor - Mod: {mode}, Erişim: {access_level}")
    
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
        access_display = {
            AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece Bilgi Alma)",
            AccessLevel.SANDBOX: "📦 Sandbox (Güvenli Dosya Yazma)",
            AccessLevel.FULL: "⚡ Tam Erişim (Terminal & Komut)"
        }.get(self.access_level, self.access_level)
        
        print(f"\n{Colors.HEADER}{'═' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}  {Config.PROJECT_NAME.upper()} SİSTEMİ v{Config.VERSION}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'═' * 70}{Colors.ENDC}")
        print(f"{Colors.CYAN}  ⚙️  Donanım     : {Config.GPU_INFO}{Colors.ENDC}")
        print(f"{Colors.CYAN}  🖥️  Cihaz       : {device.upper()}{Colors.ENDC}")
        print(f"{Colors.CYAN}  🧠 Sağlayıcı  : {Config.AI_PROVIDER.upper()}{Colors.ENDC}")
        print(f"{Colors.CYAN}  🔐 Erişim      : {access_display}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'═' * 70}{Colors.ENDC}\n")
    
    def _initialize_managers(self) -> Tuple[Dict[str, Any], Any]:
        """
        Tüm managerleri başlat
        
        Returns:
            Tuple[Manager dictionary, NLP Manager]
        """
        logger.info("📦 Managerlar başlatılıyor...")
        
        # Core managerlar
        self.state_manager = SystemStateManager()
        RuntimeContext.set_state_manager(self.state_manager)
        
        self.memory_manager = MemoryManager()
        
        # Kamera
        self.camera_manager = CameraManager()
        with ignore_stderr():
            self.camera_manager.start()
        logger.info("📷 Kamera sistemi aktif")
        
        # Güvenlik
        self.security_manager = SecurityManager(self.camera_manager)
        RuntimeContext.set_security_instance(self.security_manager)
        logger.info("🔒 Güvenlik sistemi aktif")
        
        # Diğer managerlar
        code_manager = CodeManager(Config.WORK_DIR)
        sys_health_manager = SystemHealthManager()
        finance_manager = FinanceManager()
        accounting_manager = AccountingManager()
        ops_manager = OperationsManager()
        messaging_manager = MessagingManager()
        RuntimeContext.set_messaging_manager(messaging_manager)
        
        # GitHub Manager
        self.github_manager = GithubManager(access_level=self.access_level) # <--- EKLENDİ

        # Delivery manager
        self.delivery_manager = DeliveryManager()
        if getattr(Config, "FINANCE_MODE", False):
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
            "state": self.state_manager,
            "github": self.github_manager  # <--- EKLENDİ
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
            'memory': self.memory_manager,
            'github': tools['github'] # <--- EKLENDİ
        }
        sidar_agent = SidarAgent(sidar_tools, access_level=self.access_level)
        
        # Agent'ları tools'a ekle
        tools['poyraz_special'] = poyraz_agent
        tools['sidar_special'] = sidar_agent
        
        # Poyraz'a tüm toolları ver
        poyraz_agent.update_tools(tools)
        
        # Engine'i oluştur
        self.engine = AgentEngine(self.memory_manager, tools, access_level=self.access_level)
        RuntimeContext.set_engine(self.engine)
        
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
        Mikrofonu yapılandır (Hata Korumalı)
        
        Returns:
            Başarılı ise True
        """
        # Ses sistemi Config üzerinden kapalıysa doğrudan devre dışı bırak
        if not getattr(Config, "VOICE_ENABLED", True):
            logger.info("🎙️ Mikrofon kullanımı .env ayarları ile devre dışı bırakıldı (VOICE_ENABLED=False)")
            RuntimeContext.set_voice_mode(False)
            return False

        self.recognizer = sr.Recognizer()
        
        try:
            with ignore_stderr():
                try:
                    self.microphone = sr.Microphone()
                except OSError:
                    logger.warning("⚠️ Sistemde mikrofon bulunamadı (Input Device Error)")
                    RuntimeContext.set_voice_mode(False)
                    return False

                with self.microphone as source:
                    print(f"{Colors.YELLOW}🎤 Ortam sesi kalibre ediliyor...{Colors.ENDC}")
                    self.recognizer.adjust_for_ambient_noise(
                        source,
                        duration=SystemConfig.AMBIENT_NOISE_DURATION
                    )
            
            logger.info("✅ Mikrofon kalibre edildi")
            RuntimeContext.set_voice_mode(True)
            return True
        
        except Exception as e:
            # Hata mesajını temizle ve yönet
            err_msg = str(e)
            if "generator" in err_msg or "throw" in err_msg:
                err_msg = "Mikrofon başlatılamadı (Teknik Hata: Generator Exit)"
            
            logger.warning(f"⚠️ Mikrofon hatası: {err_msg}")
            RuntimeContext.set_voice_mode(False)
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
        
        # Seslendirme (erişim seviyesine göre kısıtlama yok)
        RuntimeContext.submit_task(
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
                    RuntimeContext.submit_task(
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
                if (getattr(Config, "FINANCE_MODE", False) and 
                    current_time - last_order_check >= SystemConfig.ORDER_CHECK_INTERVAL):
                    await self._check_delivery_orders()
                    last_order_check = current_time
                
                # Ses dinleme modu
                if RuntimeContext.is_voice_mode_active() and self.state_manager.should_listen():
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
        
        # RuntimeContext'i temizle
        with suppress(Exception):
            RuntimeContext.shutdown(wait=True, timeout=5.0)
            logger.info("✓ RuntimeContext kapatıldı")
        
        logger.info("✅ LotusAI temiz bir şekilde kapatıldı")
    
    async def run(self) -> None:
        """Sistemi başlat ve çalıştır"""
        try:
            # RuntimeContext'i başlat
            RuntimeContext.initialize(max_workers=SystemConfig.MAX_WORKERS)
            
            # Event loop'u kaydet
            self.loop = asyncio.get_running_loop()
            RuntimeContext.set_loop(self.loop)
            
            # Provider modunu ayarla (Config'e işle)
            Config.set_provider_mode(self.mode)
            # Erişim seviyesini de Config'e set et (launcher'dan gelmişti, pekiştirme)
            Config.set_access_level(self.access_level)
            
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
# BAŞLATMA FONKSİYONU (GÜNCELLENMİŞ)
# ═══════════════════════════════════════════════════════════════
def start_lotus_system(mode: str = "online", access_level: str = "sandbox") -> None:
    """
    LotusAI sistemini başlatır
    
    Args:
        mode: 'online' (Gemini) veya 'local' (Ollama)
        access_level: 'restricted', 'sandbox', 'full'
    """
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        system = LotusSystem(mode=mode, access_level=access_level)
        asyncio.run(system.run())
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!] LotusAI kullanıcı tarafından kapatıldı{Colors.ENDC}")
    
    except Exception as e:
        logger.critical(f"Kritik sistem hatası: {e}", exc_info=True)
        print(f"\n{Colors.FAIL}❌ Sistem başlatılamadı: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    # Doğrudan çalıştırılırsa varsayılan değerler
    mode = os.getenv("AI_PROVIDER", "online")
    # Erişim seviyesi .env'den okunabilir, yoksa sandbox
    access = os.getenv("ACCESS_LEVEL", "sandbox")
    start_lotus_system(mode, access)