import logging
import time
import threading
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from config import Config

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Delivery")

# Selenium Kütüphane Kontrolleri
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import WebDriverException, NoSuchWindowException, TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.error("❌ Selenium eksik. 'pip install selenium webdriver-manager' çalıştırın.")

class DeliveryManager:
    """
    LotusAI Paket Servis Entegrasyon Yöneticisi (GPU Hızlandırmalı Versiyon).
    
    Yetenekler:
    - GPU Hızlandırma: Tarayıcı render işlemlerini GPU'ya aktararak CPU tasarrufu sağlar.
    - Çoklu Panel Yönetimi: Yemeksepeti, Getir, Trendyol takibi.
    - Akıllı Filtreleme: Yanlış alarmları eleyen gelişmiş kontrol mekanizması.
    - Otomatik Onarım: Çöken sekmeleri veya tarayıcıyı tespit edip yeniden başlatır.
    """
    
    def __init__(self):
        self.driver = None 
        self.is_selenium_active = False
        self.lock = threading.RLock()
        
        # Dizin Yapılandırması
        self.work_dir = Path(getattr(Config, 'WORK_DIR', Path.cwd()))
        self.user_data_dir = self.work_dir / "chrome_user_data"
        self.screenshots_dir = self.work_dir / "static" / "delivery_previews"
        
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_alerts = {} 
        
        # Platform Konfigürasyonu
        self.platforms = {
            "YEMEKSEPETI": {
                "name": "Yemeksepeti",
                "url": getattr(Config, 'YEMEKSEPETI_URL', "https://partner.yemeksepeti.com"),
                "keywords": ["yeni sipariş", "zil çalıyor", "sipariş var", "bekleyen ("]
            },
            "GETIR": {
                "name": "Getir",
                "url": getattr(Config, 'GETIR_URL', "https://restoran.getir.com"),
                "keywords": ["yeni sipariş", "sipariş geldi", "onay bekleyen"]
            },
            "TRENDYOL": {
                "name": "Trendyol",
                "url": getattr(Config, 'TRENDYOL_URL', "https://partner.trendyol.com"),
                "keywords": ["yeni sipariş", "aktif sipariş", "bekleyen ("]
            }
        }

        self.ignore_phrases = [
            "bekleyen sipariş yok", "aktif siparişiniz bulunmamaktadır",
            "sipariş bulunmamaktadır", "0 bekleyen", "(0)", "yok"
        ]

    def start_service(self, headless: bool = False) -> bool:
        """Selenium tarayıcısını GPU donanım hızlandırma ve anti-bot ayarlarıyla başlatır."""
        if not SELENIUM_AVAILABLE:
            return False

        with self.lock:
            if self.is_selenium_active and self.driver:
                return True

            logger.info("🛵 Paket Servis Tarayıcısı (GPU Hızlandırmalı) başlatılıyor...")
            
            try:
                chrome_options = Options()
                chrome_options.add_argument(f"--user-data-dir={self.user_data_dir}")
                chrome_options.add_argument("--start-maximized")
                
                # --- GPU VE DONANIM HIZLANDIRMA AYARLARI ---
                chrome_options.add_argument("--enable-gpu") # GPU kullanımını zorla
                chrome_options.add_argument("--enable-software-rasterizer")
                chrome_options.add_argument("--ignore-gpu-blocklist") # Desteklenmeyen GPU'larda bile dene
                chrome_options.add_argument("--num-raster-threads=4") # Render işlemini hızlandır
                
                if headless:
                    # Yeni headless modu GPU desteğini daha iyi yönetir
                    chrome_options.add_argument("--headless=new") 
                    chrome_options.add_argument("--disable-gpu") # Eski headless modda bazen gerekir ama 'new' ile kullanılmaz
                
                # --- ANTİ-BOT VE PERFORMANS AYARLARI ---
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--log-level=3")
                chrome_options.add_argument("--silent")
                chrome_options.add_argument("--disable-notifications") # Bildirim pencerelerini engelle

                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                
                # WebDriver izlerini gizle (JavaScript seviyesinde)
                self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
                self.is_selenium_active = True
                self._load_initial_panels()
                
                logger.info("✅ Paket Servis servisi GPU desteğiyle aktif edildi.")
                return True
                
            except Exception as e:
                logger.critical(f"❌ Tarayıcı Başlatma Hatası: {e}")
                self.is_selenium_active = False
                return False

    def _load_initial_panels(self):
        """Platformları sekmelerde açar."""
        if not self.driver: return
        
        try:
            platform_keys = list(self.platforms.keys())
            if not platform_keys: return

            # İlk platformu ana sekmede aç
            first_key = platform_keys[0]
            self.driver.get(self.platforms[first_key]["url"])
            
            # Diğerlerini yeni sekmelerde aç
            for key in platform_keys[1:]:
                data = self.platforms[key]
                self.driver.execute_script(f"window.open('{data['url']}', '_blank');")
                time.sleep(1) # Sekmeler arası yük dengelemesi
            
            logger.info(f"🌐 {len(self.platforms)} panel sekmesi GPU üzerinde hazırlandı.")
        except Exception as e:
            logger.error(f"❌ Panel yükleme hatası: {e}")

    def check_new_orders(self) -> List[str]:
        """GPU üzerinden render edilen sekmeleri tarayarak sipariş kontrolü yapar."""
        alerts = []
        if not self.is_selenium_active or not self.driver: 
            return alerts
            
        with self.lock:
            try:
                handles = self.driver.window_handles
                
                # Sekme kaybı durumunda kurtarma
                if len(handles) < len(self.platforms):
                    logger.warning("⚠️ Eksik panel tespit edildi, kurtarılıyor...")
                    self._recover_missing_tabs()
                    return alerts

                for handle in handles:
                    try:
                        self.driver.switch_to.window(handle)
                        # GPU render'ın tamamlanması için çok kısa bir es
                        time.sleep(0.3) 
                        
                        current_url = self.driver.current_url.lower()
                        active_platform = self._identify_platform(current_url)
                        
                        if active_platform:
                            p_name = active_platform['name']
                            
                            # DOM ve Metin Analizi
                            body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                            page_title = self.driver.title.lower()
                            
                            # Akıllı Kelime Eşleştirme
                            found_trigger = any(kw in body_text for kw in active_platform['keywords']) or \
                                           any(kw in page_title for kw in active_platform['keywords'])
                            
                            if found_trigger:
                                # Negatif Filtreleme (Sipariş yok mesajlarını ele)
                                if not any(ip in body_text for ip in self.ignore_phrases):
                                    # Cooldown: Aynı platform için 2 dakika (120 sn) bekle
                                    now = time.time()
                                    if now - self.last_alerts.get(p_name, 0) > 120:
                                        msg = f"🔔 {p_name}: Yeni bir sipariş veya hareketlilik algılandı!"
                                        alerts.append(msg)
                                        logger.info(msg)
                                        self.last_alerts[p_name] = now
                                        
                                        # Kanıt için ekran görüntüsü al
                                        self.take_panel_screenshot(p_name)
                                        
                                        # Bellek sızıntısını ve donmaları önlemek için ağır panelleri tazele
                                        if any(x in current_url for x in ["yemeksepeti", "trendyol"]):
                                            self.driver.refresh()
                                            logger.debug(f"🔄 {p_name} paneli tazelendi.")
                            
                    except (NoSuchWindowException, WebDriverException) as e:
                        logger.debug(f"Sekme geçiş hatası (Göz ardı edilebilir): {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Sipariş tarama döngüsünde kritik hata: {e}")
                
        return alerts

    def _identify_platform(self, url: str) -> Optional[Dict]:
        """URL içeriğinden platformu teşhis eder."""
        for data in self.platforms.values():
            domain_part = data['url'].split("//")[-1].split(".")[0]
            if domain_part in url:
                return data
        return None

    def _recover_missing_tabs(self):
        """Kapanan sekmeleri tespit eder ve GPU desteğiyle yeniden açar."""
        with self.lock:
            try:
                handles = self.driver.window_handles
                current_urls = []
                for h in handles:
                    try:
                        self.driver.switch_to.window(h)
                        current_urls.append(self.driver.current_url.lower())
                    except: continue

                for key, data in self.platforms.items():
                    domain_part = data['url'].split("//")[-1].split(".")[0]
                    if not any(domain_part in url for url in current_urls):
                        logger.info(f"🔄 {data['name']} sekmesi kurtarılıyor...")
                        self.driver.execute_script(f"window.open('{data['url']}', '_blank');")
            except Exception as e:
                logger.error(f"Tab kurtarma sırasında hata: {e}")

    def take_panel_screenshot(self, platform_name: str) -> Optional[str]:
        """GPU tarafından render edilen güncel görüntüyü diske kaydeder."""
        if not self.driver: return None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{platform_name.lower()}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            self.driver.save_screenshot(str(filepath))
            return str(filepath)
        except Exception as e:
            logger.warning(f"📸 Ekran görüntüsü alma başarısız ({platform_name}): {e}")
            return None

    def stop_service(self):
        """Tarayıcıyı ve tüm GPU kaynaklarını güvenli bir şekilde serbest bırakır."""
        with self.lock:
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("🔌 Paket servis tarayıcısı ve GPU kaynakları kapatıldı.")
                except: pass
                finally:
                    self.driver = None
                    self.is_selenium_active = False

    def get_status_summary(self) -> str:
        """GAYA ve sistem geneli için durum bilgisi üretir."""
        if not self.is_selenium_active:
            return "Paket Servis Takibi: 🔴 DEVRE DIŞI"
        try:
            tab_count = len(self.driver.window_handles)
            gpu_status = "GPU Aktif" if self.is_selenium_active else "CPU Modu"
            return f"Paket Servis Takibi: 🟢 AKTİF ({tab_count} Panel - {gpu_status})"
        except:
            return "Paket Servis Takibi: ⚠️ BAĞLANTI SORUNU"


# import logging
# import time
# import threading
# import os
# from pathlib import Path
# from datetime import datetime
# from config import Config

# # --- LOGLAMA ---
# # LotusAI merkezi log sistemine entegre named logger
# logger = logging.getLogger("LotusAI.Delivery")

# # Paket servis takibi için Selenium kütüphaneleri
# try:
#     from selenium import webdriver
#     from selenium.webdriver.chrome.service import Service
#     from selenium.webdriver.chrome.options import Options
#     from selenium.webdriver.common.by import By
#     from selenium.common.exceptions import WebDriverException, NoSuchWindowException, TimeoutException
#     from webdriver_manager.chrome import ChromeDriverManager
#     SELENIUM_AVAILABLE = True
# except ImportError:
#     SELENIUM_AVAILABLE = False
#     logger.error("Selenium kütüphaneleri eksik. 'pip install selenium webdriver-manager' çalıştırın.")

# class DeliveryManager:
#     """
#     LotusAI Paket Servis Entegrasyon Yöneticisi.
#     Yemeksepeti, Getir ve Trendyol panellerini tek bir tarayıcıda yönetir.
#     GAYA ajanı bu modülden gelen verileri işleyerek rapor sunar.
#     """
#     def __init__(self):
#         self.driver = None 
#         self.is_selenium_active = False
#         self.lock = threading.Lock()
        
#         # Kullanıcı veri dizini (Oturumların açık kalması için kritik)
#         work_dir = getattr(Config, 'WORK_DIR', Path.cwd())
#         self.user_data_dir = work_dir / "chrome_user_data"
#         self.user_data_dir.mkdir(parents=True, exist_ok=True)
        
#         # Ekran görüntüleri için klasör
#         self.screenshots_dir = work_dir / "static" / "delivery_previews"
#         self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
#         # Son tespit edilen siparişlerin kaydı (Mükerrer uyarıyı önlemek için)
#         self.last_alerts = {} 
        
#         # Restoran Panelleri Konfigürasyonu
#         self.platforms = {
#             "YEMEKSEPETI": {
#                 "name": "Yemeksepeti",
#                 "url": getattr(Config, 'YEMEKSEPETI_URL', "https://partner.yemeksepeti.com"),
#                 "keywords": ["yeni sipariş", "new order", "zil çalıyor", "sipariş var", "bekleyen ("]
#             },
#             "GETIR": {
#                 "name": "Getir",
#                 "url": getattr(Config, 'GETIR_URL', "https://restoran.getir.com"),
#                 "keywords": ["yeni sipariş", "sipariş geldi", "onay bekleyen"]
#             },
#             "TRENDYOL": {
#                 "name": "Trendyol",
#                 "url": getattr(Config, 'TRENDYOL_URL', "https://partner.trendyol.com"),
#                 "keywords": ["yeni sipariş", "aktif sipariş", "bekleyen ("]
#             }
#         }

#         # Yanlış alarmı önlemek için göz ardı edilecek metinler
#         self.ignore_phrases = [
#             "bekleyen sipariş yok", 
#             "aktif siparişiniz bulunmamaktadır",
#             "sipariş bulunmamaktadır",
#             "yeni sipariş nasıl alınır",
#             "0 bekleyen",
#             "(0)",
#             "yok"
#         ]

#     def start_service(self):
#         """Selenium tarayıcısını gelişmiş anti-detection ve performans ayarlarıyla başlatır."""
#         if not SELENIUM_AVAILABLE:
#             logger.error("Selenium modülü yüklü değil, servis başlatılamıyor.")
#             return False

#         with self.lock:
#             if self.is_selenium_active and self.driver:
#                 logger.info("Paket Servis servisi zaten aktif.")
#                 return True

#             logger.info("🛵 Paket Servis Tarayıcısı Hazırlanıyor...")
            
#             try:
#                 chrome_options = Options()
                
#                 # --- OTURUM VE PERSISTENCE ---
#                 chrome_options.add_argument(f"--user-data-dir={self.user_data_dir}")
#                 chrome_options.add_argument("--start-maximized")
                
#                 # --- GİZLİLİK VE ANTİ-BOT (Detection Prevention) ---
#                 chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#                 chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
#                 chrome_options.add_experimental_option('useAutomationExtension', False)
#                 chrome_options.add_argument("--no-sandbox")
#                 chrome_options.add_argument("--disable-dev-shm-usage")
                
#                 # Gereksiz konsol kirliliğini önle
#                 chrome_options.add_argument("--log-level=3")
#                 chrome_options.add_argument("--silent")

#                 # Otomatik Driver Kurulumu
#                 service = Service(ChromeDriverManager().install())
#                 self.driver = webdriver.Chrome(service=service, options=chrome_options)
                
#                 # Webdriver olduğunu gizle
#                 self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
#                 self.is_selenium_active = True
                
#                 # İlk açılışta panelleri yükle
#                 self._load_initial_panels()
                
#                 logger.info("✅ Paket Servis tarayıcısı ve paneller başarıyla açıldı.")
#                 return True
                
#             except Exception as e:
#                 logger.critical(f"Tarayıcı Başlatma Hatası: {e}")
#                 self.is_selenium_active = False
#                 return False

#     def _load_initial_panels(self):
#         """Tüm tanımlı platformları sekmelerde açar."""
#         if not self.driver: return
        
#         try:
#             # İlk platform (Yemeksepeti)
#             self.driver.get(self.platforms["YEMEKSEPETI"]["url"])
            
#             # Diğerlerini yeni sekmelerde aç
#             for key, data in self.platforms.items():
#                 if key == "YEMEKSEPETI": continue
#                 self.driver.execute_script(f"window.open('{data['url']}', '_blank');")
            
#             logger.info("Tüm servis sekmeleri oluşturuldu.")
#         except Exception as e:
#             logger.error(f"Panel yükleme hatası: {e}")

#     def take_panel_screenshot(self, platform_name):
#         """Gaya'nın raporuna eklemesi için mevcut panelin görüntüsünü kaydeder."""
#         if not self.driver: return None
#         try:
#             filename = f"{platform_name.lower()}_{datetime.now().strftime('%H%M%S')}.png"
#             filepath = self.screenshots_dir / filename
#             self.driver.save_screenshot(str(filepath))
#             return str(filepath)
#         except Exception as e:
#             logger.warning(f"Ekran görüntüsü alınamadı ({platform_name}): {e}")
#             return None

#     def check_new_orders(self):
#         """
#         Sekmeleri dolaşarak sipariş kontrolü yapar. 
#         GAYA ajanı bu fonksiyonu döngüsel olarak çağırır.
#         """
#         alerts = []
#         if not self.is_selenium_active or not self.driver: 
#             return alerts
            
#         with self.lock:
#             try:
#                 handles = self.driver.window_handles
                
#                 for handle in handles:
#                     try:
#                         self.driver.switch_to.window(handle)
#                         time.sleep(0.5) # Sayfanın odağa alınması ve render için kısa bekleme
                        
#                         current_url = self.driver.current_url.lower()
#                         active_platform = None
                        
#                         # URL'den hangi platformda olduğumuzu anla
#                         for key, data in self.platforms.items():
#                             if data['url'].split("//")[-1].split(".")[0] in current_url:
#                                 active_platform = data
#                                 break
                        
#                         if active_platform:
#                             p_name = active_platform['name']
                            
#                             # Sayfa içeriğini analiz et
#                             body_element = self.driver.find_element(By.TAG_NAME, "body")
#                             body_text = body_element.text.lower()
#                             page_title = self.driver.title.lower()
                            
#                             # Sipariş var mı kontrolü
#                             found_trigger = any(kw in body_text for kw in active_platform['keywords']) or \
#                                            any(kw in page_title for kw in active_platform['keywords'])
                            
#                             if found_trigger:
#                                 # Yanlış alarmları ele (Filtreleme)
#                                 is_false_alarm = any(ip in body_text for ip in self.ignore_phrases)
                                
#                                 if not is_false_alarm:
#                                     # Aynı platform için son 2 dakikada uyarı verilmiş mi kontrol et
#                                     last_alert_time = self.last_alerts.get(p_name, 0)
#                                     if time.time() - last_alert_time > 120: # 2 dakika cooldown
#                                         alert_msg = f"🔔 {p_name}: Yeni sipariş veya hareketlilik tespit edildi!"
                                        
#                                         if alert_msg not in alerts:
#                                             alerts.append(alert_msg)
#                                             logger.info(alert_msg)
#                                             self.last_alerts[p_name] = time.time()
                                            
#                                             # Görüntü kanıtı al
#                                             self.take_panel_screenshot(p_name)
                                            
#                                             # Otomatik Sayfa Yenileme (Paneli güncel tutmak için)
#                                             # Bazı paneller uzun süre dokunulmazsa bağlantıyı koparır.
#                                             if "yemeksepeti" in current_url:
#                                                 self.driver.refresh()
                            
#                     except (NoSuchWindowException, WebDriverException):
#                         # Eğer bir pencere kapandıysa veya hata verdiyse servisi yeniden canlandırmayı dene
#                         logger.warning("Bir sekme ulaşılamaz durumda, kontrol atlanıyor.")
#                         continue
                        
#             except Exception as e:
#                 logger.error(f"Sipariş tarama genel hatası: {e}")
                
#         return alerts

#     def stop_service(self):
#         """Kaynakları temizleyerek tarayıcıyı kapatır."""
#         with self.lock:
#             if self.driver:
#                 try:
#                     self.driver.quit()
#                     logger.info("Paket servis tarayıcısı kapatıldı.")
#                 except Exception as e:
#                     logger.warning(f"Kapatma hatası: {e}")
#                 finally:
#                     self.driver = None
#                     self.is_selenium_active = False

#     def restart_service(self):
#         """Kritik hatalarda sistemi ayağa kaldırır."""
#         logger.warning("🔄 Paket Servis servisi yeniden başlatılıyor...")
#         self.stop_service()
#         time.sleep(2)
#         return self.start_service()

#     def get_status_summary(self):
#         """Gaya'nın sistem durumu raporuna eklemesi için detaylı özet."""
#         if not self.is_selenium_active:
#             return "Paket Servis Takibi: 🔴 DEVRE DIŞI"
        
#         try:
#             tab_count = len(self.driver.window_handles)
#             platforms_monitored = []
            
#             # Hangi platformlar açık kontrol et
#             current_handles = self.driver.window_handles
#             for handle in current_handles:
#                 self.driver.switch_to.window(handle)
#                 url = self.driver.current_url
#                 for key, data in self.platforms.items():
#                     if data['url'].split("//")[-1].split(".")[0] in url:
#                         platforms_monitored.append(data['name'])
            
#             p_list = ", ".join(set(platforms_monitored))
#             return f"Paket Servis Takibi: 🟢 AKTİF ({tab_count} Panel: {p_list})"
#         except:
#             return "Paket Servis Takibi: ⚠️ BAĞLANTI SORUNU"