"""
LotusAI Delivery Manager
Sürüm: 2.5.4 (Hotfix: Selenium Flag Fix)
Açıklama: Paket servis entegrasyon yönetimi

Özellikler:
- Selenium automation
- GPU hızlandırmalı tarayıcı
- Çoklu platform desteği (Yemeksepeti, Getir, Trendyol)
- Otomatik kurtarma
- Akıllı filtreleme
- Ekran görüntüsü
- Thread-safe operasyonlar
"""

import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config

logger = logging.getLogger("LotusAI.Delivery")


# ═══════════════════════════════════════════════════════════════
# SELENIUM
# ═══════════════════════════════════════════════════════════════
SELENIUM_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import (
        WebDriverException,
        NoSuchWindowException,
        TimeoutException
    )
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    logger.error(
        "❌ Selenium yok. "
        "'pip install selenium webdriver-manager' çalıştırın"
    )


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class Platform(Enum):
    """Paket servis platformları"""
    YEMEKSEPETI = "yemeksepeti"
    GETIR = "getir"
    TRENDYOL = "trendyol"


class ServiceStatus(Enum):
    """Servis durumları"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RECOVERING = "recovering"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class PlatformConfig:
    """Platform yapılandırması"""
    platform: Platform
    name: str
    url: str
    keywords: List[str]


@dataclass
class OrderAlert:
    """Sipariş uyarısı"""
    platform: str
    message: str
    timestamp: datetime
    screenshot_path: Optional[str] = None


@dataclass
class DeliveryMetrics:
    """Delivery manager metrikleri"""
    alerts_generated: int = 0
    screenshots_taken: int = 0
    tab_recoveries: int = 0
    errors_encountered: int = 0
    total_checks: int = 0


# ═══════════════════════════════════════════════════════════════
# DELIVERY MANAGER
# ═══════════════════════════════════════════════════════════════
class DeliveryManager:
    """
    LotusAI Paket Servis Entegrasyon Yöneticisi
    
    Yetenekler:
    - Selenium automation: Web tabanlı platform kontrolü
    - GPU hızlandırma: Tarayıcı render için GPU kullanımı
    - Çoklu platform: Yemeksepeti, Getir, Trendyol
    - Otomatik kurtarma: Çöken sekmeleri yeniden açar
    - Akıllı filtreleme: Yanlış alarmları engeller
    - Screenshot: Sipariş kanıtı için görüntü
    
    Selenium ile tarayıcı kontrolü yaparak paket servis platformlarındaki
    yeni siparişleri otomatik tespit eder.
    """
    
    # Platform configurations
    PLATFORM_CONFIGS = {
        Platform.YEMEKSEPETI: PlatformConfig(
            platform=Platform.YEMEKSEPETI,
            name="Yemeksepeti",
            url=getattr(Config, 'YEMEKSEPETI_URL', "https://partner.yemeksepeti.com"),
            keywords=["yeni sipariş", "zil çalıyor", "sipariş var", "bekleyen ("]
        ),
        Platform.GETIR: PlatformConfig(
            platform=Platform.GETIR,
            name="Getir",
            url=getattr(Config, 'GETIR_URL', "https://restoran.getir.com"),
            keywords=["yeni sipariş", "sipariş geldi", "onay bekleyen"]
        ),
        Platform.TRENDYOL: PlatformConfig(
            platform=Platform.TRENDYOL,
            name="Trendyol",
            url=getattr(Config, 'TRENDYOL_URL', "https://partner.trendyol.com"),
            keywords=["yeni sipariş", "aktif sipariş", "bekleyen ("]
        )
    }
    
    # Ignore phrases (false positives)
    IGNORE_PHRASES = [
        "bekleyen sipariş yok",
        "aktif siparişiniz bulunmamaktadır",
        "sipariş bulunmamaktadır",
        "0 bekleyen",
        "(0)",
        "yok"
    ]
    
    # Alert cooldown (seconds)
    ALERT_COOLDOWN = 120  # 2 dakika
    
    # Tab recovery delay
    TAB_LOAD_DELAY = 1.0
    
    # Check delay
    CHECK_DELAY = 0.3
    
    def __init__(self):
        """Delivery manager başlatıcı"""
        # Selenium
        self.driver: Optional[webdriver.Chrome] = None
        self.status = ServiceStatus.INACTIVE
        self.is_selenium_active = False  # [FIX] Bu değişken eklendi
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Paths
        self.work_dir = Config.WORK_DIR
        self.user_data_dir = self.work_dir / "chrome_user_data"
        self.screenshots_dir = self.work_dir / "static" / "delivery_previews"
        
        # Create directories
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # Alert tracking
        self.last_alerts: Dict[str, float] = {}
        
        # Metrics
        self.metrics = DeliveryMetrics()
    
    # ───────────────────────────────────────────────────────────
    # SERVICE CONTROL
    # ───────────────────────────────────────────────────────────
    
    def start_service(self, headless: bool = False) -> bool:
        """
        Selenium tarayıcısını başlat
        
        Args:
            headless: Headless mode
        
        Returns:
            Başarılı ise True
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium mevcut değil")
            return False
        
        with self.lock:
            # Already active
            if self.status == ServiceStatus.ACTIVE and self.driver:
                self.is_selenium_active = True
                return True
            
            mode = "GPU" if Config.USE_GPU else "CPU"
            logger.info(f"🛵 Paket servis başlatılıyor ({mode} modu)...")
            
            try:
                # Chrome options
                chrome_options = self._create_chrome_options(headless)
                
                # Service
                service = Service(ChromeDriverManager().install())
                
                # Create driver
                self.driver = webdriver.Chrome(
                    service=service,
                    options=chrome_options
                )
                
                # Hide webdriver
                self.driver.execute_script(
                    "Object.defineProperty(navigator, 'webdriver', "
                    "{get: () => undefined})"
                )
                
                # Load panels
                self._load_initial_panels()
                
                self.status = ServiceStatus.ACTIVE
                self.is_selenium_active = True  # [FIX] Durum güncellemesi
                logger.info(f"✅ Paket servis aktif ({mode})")
                
                return True
            
            except Exception as e:
                logger.critical(f"❌ Tarayıcı başlatma hatası: {e}")
                self.status = ServiceStatus.ERROR
                self.is_selenium_active = False  # [FIX] Hata durumunda false
                self.metrics.errors_encountered += 1
                return False
    
    def _create_chrome_options(self, headless: bool) -> Options:
        """Chrome seçenekleri oluştur"""
        chrome_options = Options()
        
        # User data
        chrome_options.add_argument(
            f"--user-data-dir={self.user_data_dir}"
        )
        chrome_options.add_argument("--start-maximized")
        
        # GPU acceleration (Config controlled)
        if Config.USE_GPU:
            chrome_options.add_argument("--enable-gpu")
            chrome_options.add_argument("--enable-software-rasterizer")
            chrome_options.add_argument("--ignore-gpu-blocklist")
            chrome_options.add_argument("--num-raster-threads=4")
        else:
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-software-rasterizer")
        
        # Headless
        if headless:
            chrome_options.add_argument("--headless=new")
            if not Config.USE_GPU:
                chrome_options.add_argument("--disable-gpu")
        
        # Anti-bot
        chrome_options.add_argument(
            "--disable-blink-features=AutomationControlled"
        )
        chrome_options.add_experimental_option(
            "excludeSwitches",
            ["enable-automation"]
        )
        chrome_options.add_experimental_option(
            "useAutomationExtension",
            False
        )
        
        # Performance
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--silent")
        chrome_options.add_argument("--disable-notifications")
        
        return chrome_options
    
    def _load_initial_panels(self) -> None:
        """Platform panellerini yükle"""
        if not self.driver:
            return
        
        try:
            configs = list(self.PLATFORM_CONFIGS.values())
            
            if not configs:
                return
            
            # First platform in main tab
            first_config = configs[0]
            self.driver.get(first_config.url)
            
            # Others in new tabs
            for config in configs[1:]:
                self.driver.execute_script(
                    f"window.open('{config.url}', '_blank');"
                )
                time.sleep(self.TAB_LOAD_DELAY)
            
            logger.info(f"🌐 {len(configs)} panel hazırlandı")
        
        except Exception as e:
            logger.error(f"Panel yükleme hatası: {e}")
            self.metrics.errors_encountered += 1
    
    def stop_service(self) -> None:
        """Tarayıcıyı kapat"""
        with self.lock:
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("🔌 Paket servis kapatıldı")
                except Exception:
                    pass
                finally:
                    self.driver = None
                    self.status = ServiceStatus.INACTIVE
                    self.is_selenium_active = False  # [FIX] Durum sıfırlama
    
    # ───────────────────────────────────────────────────────────
    # ORDER CHECKING
    # ───────────────────────────────────────────────────────────
    
    def check_new_orders(self) -> List[OrderAlert]:
        """
        Yeni siparişleri kontrol et
        
        Returns:
            OrderAlert listesi
        """
        alerts = []
        
        # Eğer aktif değilse boş dön
        if self.status != ServiceStatus.ACTIVE or not self.driver or not self.is_selenium_active:
            return alerts
        
        with self.lock:
            try:
                handles = self.driver.window_handles
                
                # Tab recovery check
                if len(handles) < len(self.PLATFORM_CONFIGS):
                    logger.warning("⚠️ Eksik panel, kurtarılıyor...")
                    self.status = ServiceStatus.RECOVERING
                    self._recover_missing_tabs()
                    self.status = ServiceStatus.ACTIVE
                    return alerts
                
                # Check each tab
                for handle in handles:
                    try:
                        self.driver.switch_to.window(handle)
                        time.sleep(self.CHECK_DELAY)
                        
                        # Identify platform
                        current_url = self.driver.current_url.lower()
                        platform_config = self._identify_platform(current_url)
                        
                        if platform_config:
                            alert = self._check_platform_for_orders(
                                platform_config,
                                current_url
                            )
                            
                            if alert:
                                alerts.append(alert)
                    
                    except (NoSuchWindowException, WebDriverException) as e:
                        logger.debug(f"Sekme hatası (göz ardı): {e}")
                        continue
                
                self.metrics.total_checks += 1
            
            except Exception as e:
                logger.error(f"❌ Sipariş kontrolü hatası: {e}")
                self.metrics.errors_encountered += 1
        
        return alerts
    
    def _check_platform_for_orders(
        self,
        config: PlatformConfig,
        current_url: str
    ) -> Optional[OrderAlert]:
        """Platform'da sipariş kontrolü"""
        try:
            # Get page content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            page_title = self.driver.title.lower()
            
            # Keyword matching
            found_trigger = (
                any(kw in body_text for kw in config.keywords) or
                any(kw in page_title for kw in config.keywords)
            )
            
            if not found_trigger:
                return None
            
            # Negative filtering
            if any(phrase in body_text for phrase in self.IGNORE_PHRASES):
                return None
            
            # Cooldown check
            now = time.time()
            last_alert_time = self.last_alerts.get(config.name, 0)
            
            if now - last_alert_time <= self.ALERT_COOLDOWN:
                return None
            
            # Create alert
            message = f"🔔 {config.name}: Yeni sipariş veya hareketlilik!"
            
            # Take screenshot
            screenshot_path = self.take_panel_screenshot(config.name)
            
            # Update last alert time
            self.last_alerts[config.name] = now
            
            # Refresh heavy platforms
            if any(x in current_url for x in ["yemeksepeti", "trendyol"]):
                self.driver.refresh()
                logger.debug(f"🔄 {config.name} paneli tazelendi")
            
            self.metrics.alerts_generated += 1
            logger.info(message)
            
            return OrderAlert(
                platform=config.name,
                message=message,
                timestamp=datetime.now(),
                screenshot_path=screenshot_path
            )
        
        except Exception as e:
            logger.error(f"Platform kontrol hatası ({config.name}): {e}")
            return None
    
    def _identify_platform(self, url: str) -> Optional[PlatformConfig]:
        """URL'den platform tespit et"""
        for config in self.PLATFORM_CONFIGS.values():
            domain_part = config.url.split("//")[-1].split(".")[0]
            if domain_part in url:
                return config
        
        return None
    
    # ───────────────────────────────────────────────────────────
    # TAB RECOVERY
    # ───────────────────────────────────────────────────────────
    
    def _recover_missing_tabs(self) -> None:
        """Eksik sekmeleri kurtar"""
        with self.lock:
            try:
                handles = self.driver.window_handles
                current_urls = []
                
                # Get current URLs
                for handle in handles:
                    try:
                        self.driver.switch_to.window(handle)
                        current_urls.append(self.driver.current_url.lower())
                    except Exception:
                        continue
                
                # Check missing platforms
                for config in self.PLATFORM_CONFIGS.values():
                    domain_part = config.url.split("//")[-1].split(".")[0]
                    
                    if not any(domain_part in url for url in current_urls):
                        logger.info(f"🔄 {config.name} kurtarılıyor...")
                        self.driver.execute_script(
                            f"window.open('{config.url}', '_blank');"
                        )
                        self.metrics.tab_recoveries += 1
            
            except Exception as e:
                logger.error(f"Tab kurtarma hatası: {e}")
                self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # SCREENSHOT
    # ───────────────────────────────────────────────────────────
    
    def take_panel_screenshot(self, platform_name: str) -> Optional[str]:
        """
        Panel ekran görüntüsü
        
        Args:
            platform_name: Platform adı
        
        Returns:
            Dosya yolu veya None
        """
        if not self.driver:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{platform_name.lower()}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            
            self.driver.save_screenshot(str(filepath))
            
            self.metrics.screenshots_taken += 1
            return str(filepath)
        
        except Exception as e:
            logger.warning(f"Screenshot hatası ({platform_name}): {e}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # STATUS & METRICS
    # ───────────────────────────────────────────────────────────
    
    def get_status_summary(self) -> str:
        """
        Durum özeti
        
        Returns:
            Formatlanmış durum
        """
        if self.status == ServiceStatus.INACTIVE:
            return "Paket Servis: 🔴 DEVRE DIŞI"
        
        if self.status == ServiceStatus.ERROR:
            return "Paket Servis: ⚠️ HATA"
        
        try:
            tab_count = len(self.driver.window_handles)
            gpu_status = "GPU" if Config.USE_GPU else "CPU"
            
            return (
                f"Paket Servis: 🟢 AKTİF "
                f"({tab_count} Panel - {gpu_status})"
            )
        
        except Exception:
            return "Paket Servis: ⚠️ BAĞLANTI SORUNU"
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Delivery metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "status": self.status.value,
            "is_selenium_active": self.is_selenium_active,  # [FIX] Metriklere eklendi
            "alerts_generated": self.metrics.alerts_generated,
            "screenshots_taken": self.metrics.screenshots_taken,
            "tab_recoveries": self.metrics.tab_recoveries,
            "errors_encountered": self.metrics.errors_encountered,
            "total_checks": self.metrics.total_checks,
            "selenium_available": SELENIUM_AVAILABLE,
            "gpu_enabled": Config.USE_GPU
        }