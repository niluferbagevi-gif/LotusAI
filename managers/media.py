import wikipedia
import logging
import locale
import random
import requests
import time
import threading
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    class Config:
        WORK_DIR = os.getcwd()
        STATIC_DIR = Path("static")
        USE_GPU = False
        INSTAGRAM_ACCOUNT_ID = "lotusbagevi"
        FACEBOOK_PAGE_ID = "niluferbagevi"
        COMPETITORS = []
        GEMINI_MODEL = "gemini-1.5-flash"
        _MAIN_KEY = ""

# --- LOGGING SETUP ---
logger = logging.getLogger("LotusAI.Media")

# --- KÜTÜPHANE YÜKLEMELERİ ---

# 1. Google Search
try:
    from googlesearch import search
    SEARCH_AVAILABLE = True
except ImportError as e:
    SEARCH_AVAILABLE = False
    logger.warning(f"⚠️ MediaManager: 'googlesearch-python' yüklenemedi. Detay: {e}")

# 2. Instagram (Instaloader)
try:
    import instaloader
    INSTAGRAM_AVAILABLE = True
except ImportError as e:
    INSTAGRAM_AVAILABLE = False
    logger.warning(f"⚠️ MediaManager: 'instaloader' yüklenemedi. Detay: {e}")

# 3. Facebook Scraper
try:
    from facebook_scraper import get_posts
    FACEBOOK_AVAILABLE = True
except Exception as e:
    FACEBOOK_AVAILABLE = False
    error_msg = str(e)
    if "lxml.html.clean" in error_msg:
        logger.warning("⚠️ MediaManager: 'lxml_html_clean' eksik. (pip install lxml_html_clean)")
    else:
        logger.warning(f"⚠️ MediaManager: 'facebook-scraper' yüklenemedi. Detay: {e}")

# 4. Google Trends
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError as e:
    TRENDS_AVAILABLE = False
    logger.warning(f"⚠️ MediaManager: 'pytrends' yüklenemedi. Detay: {e}")

# --- GPU / TORCH ENTEGRASYONU (CONFIG KONTROLLÜ) ---
HAS_TORCH = False
DEVICE = "cpu"
USE_GPU_CONFIG = getattr(Config, "USE_GPU", False)

if USE_GPU_CONFIG:
    try:
        import torch
        HAS_TORCH = True
        if torch.cuda.is_available():
            DEVICE = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 MediaManager GPU Aktif: {gpu_name}")
            except:
                logger.info("🚀 MediaManager GPU Aktif")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            logger.info("🚀 MediaManager Apple Silicon GPU (MPS) Aktif")
        else:
            logger.info("ℹ️ Config GPU açık ancak uygun donanım bulunamadı. CPU kullanılacak.")
    except ImportError:
        logger.info("ℹ️ PyTorch yüklü değil, GPU hızlandırma devre dışı.")
else:
    logger.info("ℹ️ Medya işlemleri CPU modunda (Config ayarı).")


class MediaManager:
    """
    LotusAI Medya, İçerik ve Sosyal Medya Yöneticisi.
    v2.6.6 - Tam Sürüm, Donanım Hızlandırmalı ve Hata Yönetimli
    """
    
    def __init__(self):
        self.lock = threading.RLock()
        self.is_search_active = SEARCH_AVAILABLE
        self.is_insta_active = INSTAGRAM_AVAILABLE
        self.is_fb_active = FACEBOOK_AVAILABLE
        self.is_trends_active = TRENDS_AVAILABLE
        
        # Donanım Yapılandırması (Global değişkenden)
        self.device = DEVICE
        
        # Yapılandırma verileri
        self.target_insta = getattr(Config, 'INSTAGRAM_ACCOUNT_ID', "lotusbagevi")
        self.target_fb = getattr(Config, 'FACEBOOK_PAGE_ID', "niluferbagevi")
        self.competitors = getattr(Config, 'COMPETITORS', [])
        
        # API Key Yönetimi
        self.api_key = getattr(Config, '_MAIN_KEY', "")
        
        # Dizinler
        self.static_dir = Path(getattr(Config, 'STATIC_DIR', './static'))
        self.ai_images_dir = self.static_dir / "ai_images"
        try:
            self.ai_images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")

        # Pazarlama Takvimi (Türkiye odaklı)
        self.marketing_calendar = {
            "01-01": "Yılbaşı (Yeni Yılın İlk Günü) ✨",
            "02-14": "Sevgililer Günü ❤️",
            "03-08": "Dünya Kadınlar Günü 💐",
            "03-21": "Nevruz / Baharın Başlangıcı 🌱",
            "04-23": "23 Nisan Ulusal Egemenlik ve Çocuk Bayramı 🇹🇷",
            "05-01": "1 Mayıs Emek ve Dayanışma Günü 🛠️",
            "05-19": "19 Mayıs Atatürk'ü Anma, Gençlik ve Spor Bayramı 🇹🇷",
            "07-15": "15 Temmuz Demokrasi ve Milli Birlik Günü 🇹🇷",
            "08-30": "30 Ağustos Zafer Bayramı 🇹🇷",
            "10-29": "29 Ekim Cumhuriyet Bayramı 🇹🇷",
            "11-10": "10 Kasım Atatürk'ü Anma Günü 🇹🇷",
            "11-24": "Öğretmenler Günü 📚",
            "12-05": "Dünya Türk Kahvesi Günü ☕"
        }

        self._setup_environment()
        if self.is_insta_active:
            self._init_instagram()

    def _setup_environment(self):
        """Dil ve yerel ayarları yapılandırır."""
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
        except:
            try: locale.setlocale(locale.LC_ALL, 'turkish')
            except: logger.debug("MediaManager: Yerel dil ayarı varsayılanda kaldı.")
        
        try:
            wikipedia.set_lang("tr")
        except:
            pass

    def _init_instagram(self):
        """Instagram istemcisini başlatır."""
        try:
            self.L = instaloader.Instaloader()
            # Bot algılanmasını önlemek için User-Agent güncellemesi
            self.L.context._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
        except Exception as e:
            logger.error(f"Instagram başlatma hatası: {e}")
            self.is_insta_active = False

    def ai_content_advisor(self, context_data: str) -> str:
        """
        Gemini API kullanarak profesyonel içerik stratejisi önerir.
        """
        if not self.api_key:
            return "⚠️ Gemini API anahtarı yapılandırılmamış."

        try:
            system_prompt = "Sen profesyonel bir dijital pazarlama danışmanısın. Verilen güncel verilere (tarih, trendler, özel günler) göre en etkili Instagram paylaşım fikrini, caption metnini ve hashtag listesini öner."
            user_query = f"Günün Verileri: {context_data}. Bu bilgilere göre dikkat çekici bir içerik planı hazırla."
            
            payload = {
                "contents": [{"parts": [{"text": user_query}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]}
            }
            
            model = getattr(Config, 'GEMINI_MODEL', 'gemini-1.5-flash')
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            
            # Üstel geri çekilme (Exponential backoff) ile retry mekanizması
            for delay in [1, 2, 4, 8, 16]:
                try:
                    response = requests.post(url, json=payload, timeout=20)
                    if response.status_code == 200:
                        result = response.json()
                        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Öneri oluşturulamadı.")
                    elif response.status_code == 429:
                        time.sleep(delay)
                    else:
                        logger.debug(f"Gemini API Hatası ({response.status_code}): {response.text}")
                        break
                except Exception as e:
                    logger.debug(f"Gemini API Denemesi Başarısız: {e}")
                    time.sleep(delay)
            
            return "AI servisinden şu an yanıt alınamıyor."
        except Exception as e:
            logger.error(f"AI Advisor hatası: {e}")
            return "İçerik analizi yapılamadı."

    def universal_search(self, query: str) -> str:
        """
        Kullanıcı sorgusuna göre tüm dijital kaynakları tarar.
        """
        with self.lock:
            query_lower = query.lower()
            report = [f"🌐 '{query.upper()}' MEDYA VE BİLGİ RAPORU"]
            if self.device != "cpu":
                report[0] += f" [Hızlandırma: {self.device.upper()}]"

            # 1. Wikipedia Sorgusu
            if self.is_search_active:
                try:
                    wiki_sum = wikipedia.summary(query, sentences=2)
                    report.append(f"\n[BİLGİ BANKASI]:\n{wiki_sum}")
                except: pass

            # 2. Gündem ve Trendler
            if any(k in query_lower for k in ["gündem", "trend", "ne var", "popüler"]):
                report.append(f"\n[TÜRKİYE GÜNDEMİ]:\n{self.get_turkey_trends()}")

            # 3. Görsel Üretim Tetikleyicisi
            visual_triggers = ["çiz", "tasarla", "oluştur", "görsel", "resim"]
            if any(k in query_lower for k in visual_triggers):
                prompt = query
                for word in visual_triggers + ["bana", "bir", "tane"]:
                    prompt = prompt.replace(word, "")
                img_res = self.generate_concept_image(prompt.strip())
                report.append(f"\n[TASARIM]:\n{img_res}")

            # 4. Sosyal Medya İstatistikleri
            if "instagram" in query_lower or "sosyal medya" in query_lower:
                report.append(f"\n[INSTAGRAM]:\n{self.get_instagram_stats()}")
                if self.competitors:
                    report.append(f"\n[RAKİP DURUMU]:\n{self.check_competitors()}")

            if "facebook" in query_lower:
                report.append(f"\n[FACEBOOK]:\n{self.get_facebook_stats()}")

            # 5. Web Sonuçları (Google)
            if self.is_search_active and len(report) < 3:
                try:
                    google_links = []
                    # search() fonksiyonu generator döndürür
                    for j in search(query, num_results=3, lang="tr", advanced=True):
                        google_links.append(f"- {j.title}: {j.url}")
                    if google_links:
                        report.append(f"\n[WEB BAĞLANTILARI]:\n" + "\n".join(google_links))
                except Exception as e:
                    logger.debug(f"Google Search hatası: {e}")

            return "\n".join(report)

    def get_turkey_trends(self) -> str:
        """Google Trends verilerini çeker."""
        if not self.is_trends_active: return "Trends modülü pasif."
        try:
            # TrendReq nesnesi oluşturulurken timeout ve retry eklenebilir
            pytrends = TrendReq(hl='tr-TR', tz=180)
            trending = pytrends.trending_searches(pn='turkey')
            top_5 = trending.head(5)[0].tolist()
            return "🔥 " + ", ".join(top_5)
        except Exception as e:
            logger.error(f"Trends çekme hatası: {e}")
            return "Gündem verilerine şu an erişilemiyor."

    def generate_concept_image(self, prompt: str) -> str:
        """AI görseli oluşturur ve kaydeder."""
        try:
            # Daha kaliteli sonuç için prompt zenginleştirme
            styled_prompt = f"professional commercial photography, hyperrealistic, 8k, bokeh, elegant lighting, {prompt}"
            safe_prompt = requests.utils.quote(styled_prompt)
            # Pollinations AI ücretsiz ve hızlı bir görsel üretim servisidir
            url = f"https://image.pollinations.ai/prompt/{safe_prompt}?nologo=true&width=1024&height=1024&seed={random.randint(1,9999)}"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = f"concept_{int(time.time())}.jpg"
                save_path = self.ai_images_dir / filename
                save_path.write_bytes(response.content)
                return f"✅ Görsel başarıyla oluşturuldu: {filename}"
            return "❌ Görsel sunucusu şu an meşgul."
        except Exception as e:
            logger.error(f"Görsel üretim hatası: {e}")
            return f"❌ Hata: {str(e)}"

    def get_instagram_stats(self) -> str:
        """Instagram verilerini çeker."""
        if not self.is_insta_active: return "Instagram modülü eksik."
        try:
            # Oturum verileri ile profil yükleme
            profile = instaloader.Profile.from_username(self.L.context, self.target_insta)
            return f"📸 @{profile.username} | 👥 Takipçi: {profile.followers:,} | 📝 Gönderi: {profile.mediacount}"
        except Exception as e:
            logger.warning(f"Instagram istatistik hatası: {e}")
            return "Instagram verileri alınamadı (Gizlilik veya Limit)."

    def get_facebook_stats(self) -> str:
        """Facebook sayfa özetini getirir."""
        if not self.is_fb_active: return "Facebook modülü eksik."
        try:
            # facebook_scraper üzerinden gönderileri çekme
            posts = get_posts(self.target_fb, pages=1)
            for post in posts:
                text = (post.get('text') or "Görsel paylaşım")[:80]
                return f"📝 En Son: {text}..."
            return "Paylaşım bulunamadı."
        except Exception as e:
            logger.warning(f"Facebook istatistik hatası: {e}")
            return f"Facebook verilerine ulaşılamadı. ({str(e)[:50]}...)"

    def check_competitors(self) -> str:
        """Rakip analiz özeti döner."""
        if not self.is_insta_active or not self.competitors or not hasattr(self, 'L'): 
            return "Rakip takibi yapılamıyor."
        summary = []
        for comp in self.competitors:
            try:
                profile = instaloader.Profile.from_username(self.L.context, comp)
                summary.append(f"🏁 @{comp}: {profile.followers:,} takipçi")
            except: continue
        return "\n".join(summary) if summary else "Rakip verisi yok."

    def get_daily_context(self) -> str:
        """Gaya için günlük dijital brifing hazırlar."""
        now = datetime.now()
        month_day = now.strftime("%m-%d")
        
        briefing = [
            f"📅 BUGÜN: {now.strftime('%d %B %Y, %A')}",
            f"📍 LOKASYON: Bursa / Nilüfer",
            f"⚡ DONANIM: {self.device.upper()} Hızlandırma Aktif" if self.device != "cpu" else "⚡ DONANIM: CPU Modu"
        ]
        
        special = self.marketing_calendar.get(month_day)
        if special:
            briefing.append(f"🚩 ÖNEMLİ GÜN: {special}")
        
        trends = self.get_turkey_trends()
        context_str = f"Tarih: {now.strftime('%d %m')}, Özel Gün: {special if special else 'Yok'}, Trendler: {trends}"
        ai_advice = self.ai_content_advisor(context_str)
        
        briefing.append(f"\n💡 AI PAZARLAMA ÖNERİSİ:\n{ai_advice}")
        return "\n".join(briefing)

    def trigger_delivery_interface(self):
        """DeliveryManager üzerinden paket servis panellerini tetikler."""
        try:
            # Circular import önlemek için içeride import
            from managers.delivery import DeliveryManager
            dm = DeliveryManager()
            if hasattr(dm, 'start_service'):
                return dm.start_service()
            return "⚠️ Delivery servisi hazır değil."
        except Exception as e:
            logger.error(f"Delivery tetikleme hatası: {e}")
            return f"⚠️ Hata: {str(e)}"

    def get_hardware_info(self) -> Dict[str, Any]:
        """Sistem donanım bilgilerini raporlar."""
        info = {"device": self.device}
        if HAS_TORCH and torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            except: pass
        return info