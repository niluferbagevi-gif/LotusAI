"""
LotusAI Media Manager
Sürüm: 2.5.3
Açıklama: Medya, içerik ve sosyal medya yönetimi

Özellikler:
- Sosyal medya entegrasyonu (Instagram, Facebook)
- Google Trends takibi
- AI içerik önerileri (Gemini)
- Görsel oluşturma (Pollinations AI)
- Pazarlama takvimi
- Web arama
- GPU hızlandırma
"""

import wikipedia
import logging
import locale
import random
import requests
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config

logger = logging.getLogger("LotusAI.Media")


# ═══════════════════════════════════════════════════════════════
# OPTIONAL LIBRARIES
# ═══════════════════════════════════════════════════════════════
SEARCH_AVAILABLE = False
INSTAGRAM_AVAILABLE = False
FACEBOOK_AVAILABLE = False
TRENDS_AVAILABLE = False

# Google Search
try:
    from googlesearch import search
    SEARCH_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ googlesearch-python yok")

# Instagram
try:
    import instaloader
    INSTAGRAM_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ instaloader yok")

# Facebook
try:
    from facebook_scraper import get_posts
    FACEBOOK_AVAILABLE = True
except Exception as e:
    if "lxml.html.clean" in str(e):
        logger.warning("⚠️ lxml_html_clean eksik")
    else:
        logger.warning(f"⚠️ facebook-scraper yok: {e}")

# Google Trends
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ pytrends yok")


# ═══════════════════════════════════════════════════════════════
# GPU (PyTorch)
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
DEVICE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        HAS_TORCH = True
        
        if torch.cuda.is_available():
            DEVICE = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 Media GPU aktif: {gpu_name}")
            except Exception:
                logger.info("🚀 Media GPU aktif")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            logger.info("🚀 Media Apple Silicon (MPS) aktif")
    except ImportError:
        logger.info("ℹ️ PyTorch yok, GPU devre dışı")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class Platform(Enum):
    """Sosyal medya platformları"""
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    TIKTOK = "tiktok"


class ContentType(Enum):
    """İçerik tipleri"""
    POST = "post"
    STORY = "story"
    REEL = "reel"
    IMAGE = "image"
    VIDEO = "video"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class SocialStats:
    """Sosyal medya istatistikleri"""
    platform: Platform
    username: str
    followers: int
    posts: int
    engagement_rate: float = 0.0


@dataclass
class ContentSuggestion:
    """İçerik önerisi"""
    title: str
    description: str
    hashtags: List[str]
    best_time: str
    special_occasion: Optional[str] = None


@dataclass
class MediaMetrics:
    """Media manager metrikleri"""
    searches_performed: int = 0
    trends_checked: int = 0
    images_generated: int = 0
    ai_suggestions: int = 0
    social_stats_fetched: int = 0
    errors_encountered: int = 0


# ═══════════════════════════════════════════════════════════════
# MEDIA MANAGER
# ═══════════════════════════════════════════════════════════════
class MediaManager:
    """
    LotusAI Medya, İçerik ve Sosyal Medya Yöneticisi
    
    Yetenekler:
    - Sosyal medya: Instagram ve Facebook istatistikleri
    - Trend takibi: Google Trends entegrasyonu
    - AI içerik: Gemini ile içerik önerileri
    - Görsel üretim: Pollinations AI ile görsel oluşturma
    - Web arama: Google search entegrasyonu
    - Pazarlama takvimi: Türkiye özel günleri
    - GPU hızlandırma: PyTorch desteği
    
    Sosyal medya ve dijital pazarlama için merkezi yönetim noktası.
    """
    
    # Marketing calendar (Turkey focused)
    MARKETING_CALENDAR = {
        "01-01": "Yılbaşı ✨",
        "02-14": "Sevgililer Günü ❤️",
        "03-08": "Dünya Kadınlar Günü 💐",
        "03-21": "Nevruz 🌱",
        "04-23": "23 Nisan 🇹🇷",
        "05-01": "1 Mayıs 🛠️",
        "05-19": "19 Mayıs 🇹🇷",
        "07-15": "15 Temmuz 🇹🇷",
        "08-30": "30 Ağustos 🇹🇷",
        "10-29": "29 Ekim 🇹🇷",
        "11-10": "10 Kasım 🇹🇷",
        "11-24": "Öğretmenler Günü 📚",
        "12-05": "Türk Kahvesi Günü ☕"
    }
    
    # Image generation settings
    IMAGE_WIDTH = 1024
    IMAGE_HEIGHT = 1024
    
    # API retry settings
    MAX_RETRIES = 5
    RETRY_DELAYS = [1, 2, 4, 8, 16]
    
    def __init__(self):
        """Media manager başlatıcı"""
        # Thread safety
        self.lock = threading.RLock()
        
        # Feature flags
        self.is_search_active = SEARCH_AVAILABLE
        self.is_insta_active = INSTAGRAM_AVAILABLE
        self.is_fb_active = FACEBOOK_AVAILABLE
        self.is_trends_active = TRENDS_AVAILABLE
        
        # Hardware
        self.device = DEVICE
        
        # Config
        self.target_insta = getattr(Config, 'INSTAGRAM_ACCOUNT_ID', "lotusbagevi")
        self.target_fb = getattr(Config, 'FACEBOOK_PAGE_ID', "niluferbagevi")
        self.competitors = getattr(Config, 'COMPETITORS', [])
        self.api_key = getattr(Config, '_MAIN_KEY', "")
        
        # Paths
        self.static_dir = Config.STATIC_DIR
        self.ai_images_dir = self.static_dir / "ai_images"
        
        try:
            self.ai_images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # Metrics
        self.metrics = MediaMetrics()
        
        # Initialize
        self._setup_environment()
        
        if self.is_insta_active:
            self._init_instagram()
    
    def _setup_environment(self) -> None:
        """Dil ve yerel ayarları yapılandır"""
        # Locale
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
        except Exception:
            try:
                locale.setlocale(locale.LC_ALL, 'turkish')
            except Exception:
                logger.debug("Locale varsayılanda kaldı")
        
        # Wikipedia
        try:
            wikipedia.set_lang("tr")
        except Exception:
            pass
    
    def _init_instagram(self) -> None:
        """Instagram istemcisi başlat"""
        try:
            self.L = instaloader.Instaloader()
            
            # Anti-bot user agent
            self.L.context._session.headers.update({
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                )
            })
        except Exception as e:
            logger.error(f"Instagram başlatma hatası: {e}")
            self.is_insta_active = False
            self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # AI CONTENT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def ai_content_advisor(self, context_data: str) -> str:
        """
        AI içerik önerisi (Gemini)
        
        Args:
            context_data: Bağlam verisi
        
        Returns:
            İçerik önerisi
        """
        if not self.api_key:
            return "⚠️ Gemini API anahtarı yok"
        
        try:
            system_prompt = (
                "Sen profesyonel bir dijital pazarlama danışmanısın. "
                "Verilen güncel verilere göre en etkili Instagram paylaşım "
                "fikrini, caption metnini ve hashtag listesini öner."
            )
            
            user_query = (
                f"Günün Verileri: {context_data}. "
                "Bu bilgilere göre dikkat çekici bir içerik planı hazırla."
            )
            
            payload = {
                "contents": [{"parts": [{"text": user_query}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]}
            }
            
            model = getattr(Config, 'GEMINI_MODEL', 'gemini-1.5-flash')
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{model}:generateContent?key={self.api_key}"
            )
            
            # Exponential backoff
            for delay in self.RETRY_DELAYS:
                try:
                    response = requests.post(url, json=payload, timeout=20)
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = (
                            result.get('candidates', [{}])[0]
                            .get('content', {})
                            .get('parts', [{}])[0]
                            .get('text', "Öneri oluşturulamadı")
                        )
                        
                        self.metrics.ai_suggestions += 1
                        return text
                    
                    elif response.status_code == 429:
                        time.sleep(delay)
                    else:
                        logger.debug(
                            f"Gemini API hatası ({response.status_code}): "
                            f"{response.text[:100]}"
                        )
                        break
                
                except Exception as e:
                    logger.debug(f"Gemini denemesi başarısız: {e}")
                    time.sleep(delay)
            
            return "AI servisinden yanıt alınamıyor"
        
        except Exception as e:
            logger.error(f"AI advisor hatası: {e}")
            self.metrics.errors_encountered += 1
            return "İçerik analizi yapılamadı"
    
    # ───────────────────────────────────────────────────────────
    # UNIVERSAL SEARCH
    # ───────────────────────────────────────────────────────────
    
    def universal_search(self, query: str) -> str:
        """
        Evrensel arama
        
        Args:
            query: Arama sorgusu
        
        Returns:
            Arama raporu
        """
        with self.lock:
            query_lower = query.lower()
            report = [f"🌐 '{query.upper()}' MEDYA RAPORU"]
            
            if self.device != "cpu":
                report[0] += f" [Hızlandırma: {self.device.upper()}]"
            
            # Wikipedia
            if self.is_search_active:
                try:
                    wiki_sum = wikipedia.summary(query, sentences=2)
                    report.append(f"\n[BİLGİ BANKASI]:\n{wiki_sum}")
                except Exception:
                    pass
            
            # Trends
            if any(k in query_lower for k in ["gündem", "trend", "popüler"]):
                report.append(f"\n[TÜRKİYE GÜNDEMİ]:\n{self.get_turkey_trends()}")
            
            # Image generation
            visual_triggers = ["çiz", "tasarla", "oluştur", "görsel", "resim"]
            if any(k in query_lower for k in visual_triggers):
                prompt = query
                for word in visual_triggers + ["bana", "bir", "tane"]:
                    prompt = prompt.replace(word, "")
                
                img_res = self.generate_concept_image(prompt.strip())
                report.append(f"\n[TASARIM]:\n{img_res}")
            
            # Social media
            if "instagram" in query_lower or "sosyal medya" in query_lower:
                report.append(f"\n[INSTAGRAM]:\n{self.get_instagram_stats()}")
                
                if self.competitors:
                    report.append(f"\n[RAKİP]:\n{self.check_competitors()}")
            
            if "facebook" in query_lower:
                report.append(f"\n[FACEBOOK]:\n{self.get_facebook_stats()}")
            
            # Web search
            if self.is_search_active and len(report) < 3:
                try:
                    google_links = []
                    for j in search(query, num_results=3, lang="tr", advanced=True):
                        google_links.append(f"- {j.title}: {j.url}")
                    
                    if google_links:
                        report.append(
                            f"\n[WEB]:\n" + "\n".join(google_links)
                        )
                except Exception as e:
                    logger.debug(f"Google search hatası: {e}")
            
            self.metrics.searches_performed += 1
            
            return "\n".join(report)
    
    # ───────────────────────────────────────────────────────────
    # TRENDS
    # ───────────────────────────────────────────────────────────
    
    def get_turkey_trends(self) -> str:
        """
        Türkiye trendleri
        
        Returns:
            Trend listesi
        """
        if not self.is_trends_active:
            return "Trends modülü pasif"
        
        try:
            pytrends = TrendReq(hl='tr-TR', tz=180)
            trending = pytrends.trending_searches(pn='turkey')
            top_5 = trending.head(5)[0].tolist()
            
            self.metrics.trends_checked += 1
            
            return "🔥 " + ", ".join(top_5)
        
        except Exception as e:
            logger.error(f"Trends hatası: {e}")
            self.metrics.errors_encountered += 1
            return "Gündem verilerine erişilemiyor"
    
    # ───────────────────────────────────────────────────────────
    # IMAGE GENERATION
    # ───────────────────────────────────────────────────────────
    
    def generate_concept_image(self, prompt: str) -> str:
        """
        AI görsel oluştur
        
        Args:
            prompt: Görsel tanımı
        
        Returns:
            Sonuç mesajı
        """
        try:
            # Enhance prompt
            styled_prompt = (
                f"professional commercial photography, "
                f"hyperrealistic, 8k, bokeh, elegant lighting, {prompt}"
            )
            
            safe_prompt = requests.utils.quote(styled_prompt)
            
            # Pollinations AI
            url = (
                f"https://image.pollinations.ai/prompt/{safe_prompt}"
                f"?nologo=true"
                f"&width={self.IMAGE_WIDTH}"
                f"&height={self.IMAGE_HEIGHT}"
                f"&seed={random.randint(1, 9999)}"
            )
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                filename = f"concept_{int(time.time())}.jpg"
                save_path = self.ai_images_dir / filename
                save_path.write_bytes(response.content)
                
                self.metrics.images_generated += 1
                
                return f"✅ Görsel oluşturuldu: {filename}"
            
            return "❌ Görsel sunucusu meşgul"
        
        except Exception as e:
            logger.error(f"Görsel üretim hatası: {e}")
            self.metrics.errors_encountered += 1
            return f"❌ Hata: {str(e)[:50]}"
    
    # ───────────────────────────────────────────────────────────
    # SOCIAL MEDIA STATS
    # ───────────────────────────────────────────────────────────
    
    def get_instagram_stats(self) -> str:
        """
        Instagram istatistikleri
        
        Returns:
            İstatistik metni
        """
        if not self.is_insta_active:
            return "Instagram modülü eksik"
        
        try:
            profile = instaloader.Profile.from_username(
                self.L.context,
                self.target_insta
            )
            
            self.metrics.social_stats_fetched += 1
            
            return (
                f"📸 @{profile.username} | "
                f"👥 Takipçi: {profile.followers:,} | "
                f"📝 Gönderi: {profile.mediacount}"
            )
        
        except Exception as e:
            logger.warning(f"Instagram istatistik hatası: {e}")
            self.metrics.errors_encountered += 1
            return "Instagram verileri alınamadı (Gizlilik/Limit)"
    
    def get_facebook_stats(self) -> str:
        """
        Facebook istatistikleri
        
        Returns:
            İstatistik metni
        """
        if not self.is_fb_active:
            return "Facebook modülü eksik"
        
        try:
            posts = get_posts(self.target_fb, pages=1)
            
            for post in posts:
                text = (post.get('text') or "Görsel paylaşım")[:80]
                
                self.metrics.social_stats_fetched += 1
                
                return f"📝 En Son: {text}..."
            
            return "Paylaşım bulunamadı"
        
        except Exception as e:
            logger.warning(f"Facebook istatistik hatası: {e}")
            self.metrics.errors_encountered += 1
            return f"Facebook verilerine ulaşılamadı ({str(e)[:50]})"
    
    def check_competitors(self) -> str:
        """
        Rakip analizi
        
        Returns:
            Rakip özeti
        """
        if not self.is_insta_active or not self.competitors:
            return "Rakip takibi yapılamıyor"
        
        if not hasattr(self, 'L'):
            return "Instagram istemcisi yok"
        
        summary = []
        
        for comp in self.competitors:
            try:
                profile = instaloader.Profile.from_username(
                    self.L.context,
                    comp
                )
                summary.append(f"🏁 @{comp}: {profile.followers:,} takipçi")
            except Exception:
                continue
        
        return "\n".join(summary) if summary else "Rakip verisi yok"
    
    # ───────────────────────────────────────────────────────────
    # DAILY CONTEXT
    # ───────────────────────────────────────────────────────────
    
    def get_daily_context(self) -> str:
        """
        Günlük brifing
        
        Returns:
            Formatlanmış brifing
        """
        now = datetime.now()
        month_day = now.strftime("%m-%d")
        
        briefing = [
            f"📅 BUGÜN: {now.strftime('%d %B %Y, %A')}",
            f"📍 LOKASYON: Bursa / Nilüfer"
        ]
        
        # Hardware
        if self.device != "cpu":
            briefing.append(f"⚡ DONANIM: {self.device.upper()} Aktif")
        else:
            briefing.append("⚡ DONANIM: CPU Modu")
        
        # Special day
        special = self.MARKETING_CALENDAR.get(month_day)
        if special:
            briefing.append(f"🚩 ÖNEMLİ GÜN: {special}")
        
        # Trends
        trends = self.get_turkey_trends()
        
        # AI suggestion
        context_str = (
            f"Tarih: {now.strftime('%d %m')}, "
            f"Özel Gün: {special if special else 'Yok'}, "
            f"Trendler: {trends}"
        )
        
        ai_advice = self.ai_content_advisor(context_str)
        briefing.append(f"\n💡 AI PAZARLAMA ÖNERİSİ:\n{ai_advice}")
        
        return "\n".join(briefing)
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Donanım bilgisi
        
        Returns:
            Donanım dict
        """
        info = {"device": self.device}
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["memory_allocated"] = (
                    f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )
            except Exception:
                pass
        
        return info
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Media metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "searches_performed": self.metrics.searches_performed,
            "trends_checked": self.metrics.trends_checked,
            "images_generated": self.metrics.images_generated,
            "ai_suggestions": self.metrics.ai_suggestions,
            "social_stats_fetched": self.metrics.social_stats_fetched,
            "errors_encountered": self.metrics.errors_encountered,
            "device": self.device,
            "search_available": self.is_search_active,
            "instagram_available": self.is_insta_active,
            "facebook_available": self.is_fb_active,
            "trends_available": self.is_trends_active
        }