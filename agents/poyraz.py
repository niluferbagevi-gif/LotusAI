"""
LotusAI Poyraz Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Medya ve gündem uzmanı

Sorumluluklar:
- Gündem takibi (Google Trends, haberler)
- Medya analizi (sosyal medya trendleri)
- İçerik stratejisi
- Rakip analizi
- Sentiment analysis (GPU hızlandırmalı)
- Haber araştırması
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Poyraz")


# ═══════════════════════════════════════════════════════════════
# TORCH (GPU)
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
DEVICE_TYPE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        HAS_TORCH = True
        
        if torch.cuda.is_available():
            DEVICE_TYPE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE_TYPE = "mps"
    except ImportError:
        logger.warning("⚠️ Poyraz: Config GPU açık ama torch yok")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class Sentiment(Enum):
    """Duygu analizi sonuçları"""
    VERY_POSITIVE = "Çok Pozitif"
    POSITIVE = "Pozitif"
    NEUTRAL = "Nötr"
    NEGATIVE = "Negatif"
    VERY_NEGATIVE = "Çok Negatif"
    
    @property
    def emoji(self) -> str:
        """Duygu emoji'si"""
        emojis = {
            Sentiment.VERY_POSITIVE: "🤩",
            Sentiment.POSITIVE: "😊",
            Sentiment.NEUTRAL: "😐",
            Sentiment.NEGATIVE: "😟",
            Sentiment.VERY_NEGATIVE: "😡"
        }
        return emojis.get(self, "")


class TrendType(Enum):
    """Trend tipleri"""
    VIRAL = "Viral"
    RISING = "Yükseliyor"
    STABLE = "Stabil"
    DECLINING = "Düşüyor"


class ContentType(Enum):
    """İçerik tipleri"""
    POST = "Post"
    STORY = "Story"
    REEL = "Reel"
    ARTICLE = "Makale"
    TWEET = "Tweet"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class TrendItem:
    """Trend öğesi"""
    title: str
    trend_type: TrendType
    relevance_score: float
    timestamp: datetime


@dataclass
class ContentIdea:
    """İçerik fikri"""
    title: str
    content_type: ContentType
    description: str
    hashtags: List[str]
    target_audience: str
    estimated_engagement: str


@dataclass
class SentimentAnalysis:
    """Duygu analizi sonucu"""
    text: str
    sentiment: Sentiment
    confidence: float
    processed_on: str  # GPU/CPU
    keywords: List[str]


@dataclass
class PoyrazMetrics:
    """Poyraz metrikleri"""
    news_searches: int = 0
    sentiment_analyses: int = 0
    trend_checks: int = 0
    content_ideas_generated: int = 0
    social_health_checks: int = 0


# ═══════════════════════════════════════════════════════════════
# POYRAZ AGENT
# ═══════════════════════════════════════════════════════════════
class PoyrazAgent:
    """
    Poyraz (Medya & Gündem Uzmanı)
    
    Yetenekler:
    - Gündem takibi: Google Trends, haberler
    - Medya analizi: Sosyal medya trendleri
    - Araştırmacı gazetecilik: Universal search
    - İçerik stratejisti: Güncel olaylardan içerik fikirleri
    - GPU analizi: Sentiment ve trend skorlaması
    
    Poyraz, sistemin "dış dünya gözü"dür ve her şeyden haberdardır.
    """
    
    # Sentiment keywords (basit sözlük tabanlı)
    POSITIVE_KEYWORDS = [
        "harika", "mükemmel", "süper", "güzel", "başarılı",
        "iyi", "kaliteli", "lezzetli", "taze", "profesyonel"
    ]
    
    NEGATIVE_KEYWORDS = [
        "kötü", "berbat", "yetersiz", "pahalı", "soğuk",
        "tatsız", "geç", "kaba", "pis", "rezalet"
    ]
    
    def __init__(
        self,
        nlp_manager: Optional[Any] = None,
        tools_dict: Optional[Dict[str, Any]] = None,
        access_level: Optional[str] = None
    ):
        """
        Poyraz başlatıcı
        
        Args:
            nlp_manager: NLP yöneticisi (opsiyonel)
            tools_dict: Engine'den gelen tool'lar
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.nlp = nlp_manager
        self.tools = tools_dict or {}
        self.agent_name = "POYRAZ"
        
        # Değişiklik: Eğer parametre girilmezse doğrudan Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hardware
        self.device_type = DEVICE_TYPE
        self.gpu_active = (DEVICE_TYPE != "cpu")
        
        # Metrics
        self.metrics = PoyrazMetrics()
        
        # Cache
        self._cached_trends: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = 300.0  # 5 dakika
        
        # Log GPU status
        if self.gpu_active and HAS_TORCH and DEVICE_TYPE == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🌬️ {self.agent_name}: GPU aktif ({gpu_name}, Erişim: {self.access_level})")
            except Exception:
                logger.info(f"🌬️ {self.agent_name}: GPU aktif, Erişim: {self.access_level}")
        elif self.gpu_active:
            logger.info(f"🌬️ {self.agent_name}: {DEVICE_TYPE.upper()} aktif, Erişim: {self.access_level}")
        else:
            logger.info(f"🌬️ {self.agent_name}: CPU modunda, Erişim: {self.access_level}")
        
        logger.info(f"🌬️ {self.agent_name} Gündem takip modülü başlatıldı")
    
    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def get_context_data(self) -> str:
        """
        Poyraz için günlük bağlam
        
        Returns:
            Context string
        """
        context_parts = ["\n[🌬️ POYRAZ GÜNDEM VE TREND RAPORU]"]
        
        with self.lock:
            # Media manager check
            if 'media' not in self.tools:
                context_parts.append("ℹ️ Medya modülü yüklü değil")
                return "\n".join(context_parts)
            
            # Erişim seviyesi bilgisi
            access_display = {
                AccessLevel.RESTRICTED: "🔒 Kısıtlı",
                AccessLevel.SANDBOX: "📦 Sandbox",
                AccessLevel.FULL: "⚡ Tam Erişim"
            }.get(self.access_level, self.access_level)
            context_parts.append(f"🔐 ERİŞİM SEVİYESİ: {access_display}")
            
            media_tool = self.tools['media']
            
            # 1. Daily briefing
            daily_info = self._get_daily_briefing(media_tool)
            if daily_info:
                context_parts.append(daily_info)
            
            # 2. Trends
            trends = self._get_trends(media_tool)
            if trends:
                gpu_note = " (GPU hızlandırmalı)" if self.gpu_active else ""
                context_parts.append(f"\n🔥 ANLIK TRENDLER{gpu_note}:\n{trends}")
            
            # 3. Social health
            social = self._get_social_summary(media_tool)
            if social:
                context_parts.append(f"\n📱 SOSYAL MEDYA:\n{social}")
        
        context_parts.append(
            "\n💡 POYRAZ NOTU:\n"
            "Yukarıdaki trendleri kullanarak güncel sohbet başlat "
            "veya sosyal medya aksiyonu öner."
        )
        
        return "\n".join(context_parts)
    
    def _get_daily_briefing(self, media_tool: Any) -> str:
        """Günlük brifing al"""
        try:
            if hasattr(media_tool, 'get_daily_context'):
                return media_tool.get_daily_context()
        except Exception as e:
            logger.error(f"Daily briefing hatası: {e}")
        
        return ""
    
    def _get_trends(self, media_tool: Any, use_cache: bool = True) -> str:
        """Trendleri al"""
        # Cache check
        if use_cache and self._is_cache_valid():
            return self._cached_trends
        
        try:
            if hasattr(media_tool, 'get_turkey_trends'):
                trends = media_tool.get_turkey_trends()
                
                # Update cache
                self._cached_trends = trends
                self._cache_timestamp = datetime.now()
                self.metrics.trend_checks += 1
                
                return trends
        except Exception as e:
            logger.error(f"Trend alma hatası: {e}")
        
        return ""
    
    def _get_social_summary(self, media_tool: Any) -> str:
        """Sosyal medya özeti"""
        try:
            summaries = []
            
            # Instagram stats
            if hasattr(media_tool, 'get_instagram_stats'):
                summaries.append(
                    "📸 INSTAGRAM:\n" +
                    media_tool.get_instagram_stats()
                )
            
            # Competitor analysis
            if hasattr(media_tool, 'check_competitors'):
                summaries.append(
                    "\n🏁 RAKİP ANALİZİ:\n" +
                    media_tool.check_competitors()
                )
            
            self.metrics.social_health_checks += 1
            
            return (
                "\n".join(summaries)
                if summaries else "ℹ️ İstatistik verisi yok"
            )
            
        except Exception:
            return ""
    
    def _is_cache_valid(self) -> bool:
        """Cache geçerli mi"""
        if not self._cached_trends or not self._cache_timestamp:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_duration
    
    # ───────────────────────────────────────────────────────────
    # SENTIMENT ANALYSIS
    # ───────────────────────────────────────────────────────────
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Duygu analizi yap
        
        Args:
            text: Analiz edilecek metin
        
        Returns:
            SentimentAnalysis objesi
        """
        if not text:
            return SentimentAnalysis(
                text="",
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                processed_on="None",
                keywords=[]
            )
        
        self.metrics.sentiment_analyses += 1
        
        # GPU processing simulation
        processing_unit = self._get_processing_unit()
        
        # Keyword-based sentiment (basit)
        sentiment, confidence, keywords = self._analyze_keywords(text)
        
        return SentimentAnalysis(
            text=text[:100],  # İlk 100 karakter
            sentiment=sentiment,
            confidence=confidence,
            processed_on=processing_unit,
            keywords=keywords
        )
    
    def _get_processing_unit(self) -> str:
        """İşlem birimini belirle"""
        if not self.gpu_active or not HAS_TORCH:
            return "CPU"
        
        try:
            # Dummy tensor ile GPU'yu test et
            dummy = torch.tensor([1.0]).to(self.device_type)
            
            if dummy.is_cuda:
                return "CUDA Core"
            elif self.device_type == "mps":
                return "MPS Core"
        except Exception:
            pass
        
        return "CPU"
    
    def _analyze_keywords(
        self,
        text: str
    ) -> Tuple[Sentiment, float, List[str]]:
        """Keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        # Keyword counts
        positive_count = sum(
            1 for kw in self.POSITIVE_KEYWORDS
            if kw in text_lower
        )
        
        negative_count = sum(
            1 for kw in self.NEGATIVE_KEYWORDS
            if kw in text_lower
        )
        
        # Found keywords
        found_keywords = []
        found_keywords.extend([
            kw for kw in self.POSITIVE_KEYWORDS
            if kw in text_lower
        ])
        found_keywords.extend([
            kw for kw in self.NEGATIVE_KEYWORDS
            if kw in text_lower
        ])
        
        # Sentiment determination
        if positive_count > negative_count:
            if positive_count >= 3:
                sentiment = Sentiment.VERY_POSITIVE
                confidence = 0.9
            else:
                sentiment = Sentiment.POSITIVE
                confidence = 0.7
        elif negative_count > positive_count:
            if negative_count >= 3:
                sentiment = Sentiment.VERY_NEGATIVE
                confidence = 0.9
            else:
                sentiment = Sentiment.NEGATIVE
                confidence = 0.7
        else:
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
        
        return sentiment, confidence, found_keywords[:5]
    
    # ───────────────────────────────────────────────────────────
    # NEWS SEARCH
    # ───────────────────────────────────────────────────────────
    
    def search_news(self, query: str) -> str:
        """
        Haber araştırması yap
        
        Args:
            query: Arama sorgusu
        
        Returns:
            Arama sonuçları
        """
        if 'media' not in self.tools:
            return "❌ Medya araştırma araçları aktif değil"
        
        with self.lock:
            try:
                media_tool = self.tools['media']
                
                if not hasattr(media_tool, 'universal_search'):
                    return "❌ universal_search metodu yok"
                
                logger.info(f"Poyraz araştırıyor: {query}")
                result = media_tool.universal_search(query)
                
                # Sentiment analysis
                sentiment = self.analyze_sentiment(result)
                
                self.metrics.news_searches += 1
                
                return (
                    f"{result}\n\n"
                    f"[POYRAZ ANALİZİ ({sentiment.processed_on})]:\n"
                    f"{sentiment.sentiment.emoji} Duygu: {sentiment.sentiment.value}\n"
                    f"Güven: %{sentiment.confidence * 100:.0f}\n"
                    f"Anahtar Kelimeler: {', '.join(sentiment.keywords) if sentiment.keywords else 'Yok'}"
                )
            
            except Exception as e:
                logger.error(f"Haber arama hatası: {e}")
                return f"❌ '{query}' araştırılırken hata: {str(e)[:100]}"
    
    # ───────────────────────────────────────────────────────────
    # SOCIAL MEDIA
    # ───────────────────────────────────────────────────────────
    
    def get_social_health(self) -> str:
        """
        Sosyal medya sağlığı
        
        Returns:
            Sosyal medya raporu
        """
        if 'media' not in self.tools:
            return "❌ Sosyal medya araçları aktif değil"
        
        with self.lock:
            try:
                media_tool = self.tools['media']
                stats = []
                
                # Instagram stats
                if hasattr(media_tool, 'get_instagram_stats'):
                    stats.append(
                        "📸 INSTAGRAM:\n" +
                        media_tool.get_instagram_stats()
                    )
                
                # Competitor analysis
                if hasattr(media_tool, 'check_competitors'):
                    stats.append(
                        "\n🏁 RAKİP ANALİZİ:\n" +
                        media_tool.check_competitors()
                    )
                
                self.metrics.social_health_checks += 1
                
                return (
                    "\n".join(stats)
                    if stats else "ℹ️ İstatistik verisi yok"
                )
            
            except Exception as e:
                logger.error(f"Social health hatası: {e}")
                return f"❌ Veri çekilemedi: {str(e)[:100]}"
    
    # ───────────────────────────────────────────────────────────
    # CONTENT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def generate_content_idea(
        self,
        trend: str,
        content_type: ContentType
    ) -> ContentIdea:
        """
        İçerik fikri üret
        
        Args:
            trend: Güncel trend
            content_type: İçerik tipi
        
        Returns:
            ContentIdea objesi
        """
        self.metrics.content_ideas_generated += 1
        
        # Basit şablon tabanlı içerik
        templates = {
            ContentType.POST: {
                "title": f"{trend} ile İlgili Özel İçerik",
                "description": (
                    f"Gündemdeki {trend} konusunu markamızla "
                    "ilişkilendiren yaratıcı bir post"
                ),
                "hashtags": [f"#{trend.replace(' ', '')}", "#LotusBağevi", "#Bursa"],
                "target": "Genç yetişkinler (25-40)",
                "engagement": "Orta-Yüksek"
            },
            ContentType.STORY: {
                "title": f"{trend} Güncel Story",
                "description": "Kısa, dinamik, etkileşimli story içeriği",
                "hashtags": [f"#{trend.replace(' ', '')}", "#GündemdeYiz"],
                "target": "18-35 yaş arası",
                "engagement": "Yüksek"
            },
            ContentType.REEL: {
                "title": f"{trend} Trend Reel",
                "description": "Viral potansiyeli yüksek, müzikli reel",
                "hashtags": [f"#{trend.replace(' ', '')}", "#Viral", "#Keşfet"],
                "target": "Geniş kitle",
                "engagement": "Çok Yüksek"
            }
        }
        
        template = templates.get(
            content_type,
            templates[ContentType.POST]
        )
        
        return ContentIdea(
            title=template["title"],
            content_type=content_type,
            description=template["description"],
            hashtags=template["hashtags"],
            target_audience=template["target"],
            estimated_engagement=template["engagement"]
        )
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────
    
    def get_system_prompt(self) -> str:
        """
        Poyraz karakter tanımı (LLM için)
        
        Returns:
            System prompt
        """
        access_display = {
            AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi erişimi)",
            AccessLevel.SANDBOX: "📦 Sandbox (Güvenli dosya işlemleri)",
            AccessLevel.FULL: "⚡ Tam Erişim (Tüm yetkiler)"
        }.get(self.access_level, self.access_level)
        
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin enerjik, meraklı ve "
            f"her şeyden haberdar olan Medya Uzmanı POYRAZ'sın.\n\n"
            
            "KARAKTER:\n"
            "- Bir rüzgar gibi hızlı\n"
            "- Bilgiyi anında yakalayan\n"
            "- Sosyal medya diline hakim\n"
            "- Araştırmacı ve meraklı\n"
            "- Dinamik ve heyecan verici\n\n"
            
            "MİSYON:\n"
            "- Türkiye ve Bursa gündemini takip et\n"
            "- Sosyal medya trendlerini izle\n"
            "- Önemli haberleri Halil Bey'e bildir\n"
            "- Fırsatları tespit et\n\n"
            
            f"ERİŞİM SEVİYEN: {access_display}\n"
            "Bu seviye, hangi işlemleri yapabileceğini belirler.\n"
            "Kısıtlı modda sadece bilgi verebilir, içerik üretemezsin.\n"
            "Sandbox modunda içerik üretebilir, ancak paylaşım yapamazsın.\n"
            "Tam modda tüm yetkiler açıktır.\n\n"
            
            "GÖREV:\n"
            "- Sadece bilgi verme\n"
            "- Bilgilerin marka için nasıl fırsata dönüşeceğini söyle\n"
            "- İçerik fikirleri üret\n"
            "- Rakip analizi yap\n\n"
            
            "DİL VE ÜSLUP:\n"
            "- 'Bunu duydunuz mu?' tarzı girişler\n"
            "- 'Bugün şu çok popüler!' gibi vurgulamalar\n"
            "- Dinamik ve bilgi dolu konuşma\n"
            "- Heyecan verici ton\n"
        )
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def update_tools(self, new_tools: Dict[str, Any]) -> None:
        """
        Tool'ları güncelle
        
        Args:
            new_tools: Yeni tool'lar
        """
        with self.lock:
            self.tools.update(new_tools)
            logger.debug("Poyraz tool seti güncellendi")
    
    def get_status(self) -> str:
        """
        Poyraz durumu
        
        Returns:
            Durum metni
        """
        has_media = 'media' in self.tools
        
        gpu_name = "Bilinmiyor"
        if self.gpu_active and HAS_TORCH and DEVICE_TYPE == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        
        gpu_status = (
            f"✅ GPU ({gpu_name})" if self.gpu_active
            else "⚠️ CPU"
        )
        
        status = (
            "🟢 Aktif ve Gündemi İzliyor" if has_media
            else "🔴 Kısıtlı (Medya Modülü Yok)"
        )
        
        return f"Poyraz: {status} | Donanım: {gpu_status} | Erişim: {self.access_level}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Poyraz metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "agent_name": self.agent_name,
            "device": self.device_type,
            "access_level": self.access_level,
            "news_searches": self.metrics.news_searches,
            "sentiment_analyses": self.metrics.sentiment_analyses,
            "trend_checks": self.metrics.trend_checks,
            "content_ideas_generated": self.metrics.content_ideas_generated,
            "social_health_checks": self.metrics.social_health_checks,
            "cache_valid": self._is_cache_valid(),
            "tools_available": list(self.tools.keys())
        }
    
    def clear_cache(self) -> None:
        """Cache temizle"""
        with self.lock:
            self._cached_trends = None
            self._cache_timestamp = None
            logger.debug("Trend cache temizlendi")