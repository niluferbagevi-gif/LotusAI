"""
LotusAI agents/poyraz.py - Poyraz Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu & NLP Entegrasyonu)
Açıklama: Medya ve gündem uzmanı

Sorumluluklar:
- Gündem takibi (Google Trends, haberler)
- Medya analizi (sosyal medya trendleri)
- İçerik stratejisi
- Rakip analizi
- Sentiment analysis (Merkezi NLP Manager ile entegre)
- Haber araştırması
- auto_handle: Medya ve gündem sorgularını LLM'e gitmeden anlık karşılama
"""

import logging
import threading
import re
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
    processed_on: str
    keywords: List[str]


@dataclass
class PoyrazMetrics:
    """Poyraz metrikleri"""
    news_searches: int = 0
    sentiment_analyses: int = 0
    trend_checks: int = 0
    content_ideas_generated: int = 0
    social_health_checks: int = 0
    auto_handle_count: int = 0


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
    - Duygu analizi: Merkezi NLP Yöneticisi ile gerçek yapay zeka analizi
    - auto_handle: Medya ve gündem sorgularını LLM'siz anlık karşılama

    Erişim Seviyesi Kuralları (OpenClaw):
    - restricted: Gündem ve trend bilgisi okuyabilir.
    - sandbox: İçerik fikirleri üretebilir.
    - full: Tüm yetkiler (sosyal medya paylaşımı dahil).

    Poyraz, sistemin "dış dünya gözü"dür ve her şeyden haberdardır.
    """

    # Sentiment keywords (NLP modülü hata verirse Fallback olarak kullanılır)
    POSITIVE_KEYWORDS = [
        "harika", "mükemmel", "süper", "güzel", "başarılı",
        "iyi", "kaliteli", "lezzetli", "taze", "profesyonel"
    ]

    NEGATIVE_KEYWORDS = [
        "kötü", "berbat", "yetersiz", "pahalı", "soğuk",
        "tatsız", "geç", "kaba", "pis", "rezalet"
    ]

    # auto_handle tetikleyicileri
    AUTO_TREND_TRIGGERS = [
        "trendler", "gündem", "ne trend", "ne konuşuluyor",
        "ne var ne yok", "gündemdekiler", "trending"
    ]
    AUTO_NEWS_TRIGGERS = [
        "haberler", "son dakika", "bugün ne oldu",
        "gelişmeler", "haber var mı"
    ]
    AUTO_SOCIAL_TRIGGERS = [
        "sosyal medya durumu", "instagram durumu", "rakip analizi",
        "sosyal sağlık", "takipçi"
    ]
    AUTO_CONTENT_TRIGGERS = [
        "içerik öner", "post fikri", "ne paylaşayım", "reel fikri",
        "story fikri", "içerik stratejisi"
    ]
    AUTO_METRIC_TRIGGERS = [
        "poyraz raporu", "analiz sayısı", "istatistik",
        "kaç haber", "performans"
    ]
    AUTO_SENTIMENT_TRIGGERS = [
        "duygu analizi", "sentiment", "yorumları analiz et",
        "ne düşünüyorlar", "yorum analizi"
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

        # Parametre girilmezse Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL

        # Thread safety
        self.lock = threading.RLock()

        # Metrics
        self.metrics = PoyrazMetrics()

        # Cache (trendler için — 5 dakika)
        self._cached_trends: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = 300.0

        logger.info(f"🌬️ {self.agent_name} Gündem takip modülü başlatıldı (Erişim: {self.access_level.upper()})")

    # ───────────────────────────────────────────────────────────
    # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
    # ───────────────────────────────────────────────────────────

    async def auto_handle(self, text: str) -> Optional[str]:
        """
        Kullanıcı metnini analiz ederek Poyraz'ın araçlarını
        otomatik çalıştırır. engine.py tarafından çağrılır.

        LLM'e gitmeden önce anlık, deterministik yanıt üretir.
        Eşleşme yoksa None döner ve akış LLM'e devam eder.

        Args:
            text: Kullanıcı metni (temizlenmiş)

        Returns:
            Yanıt string veya None
        """
        text_lower = text.lower()
        self.metrics.auto_handle_count += 1

        # 1. Trend / Gündem sorgulama
        if any(t in text_lower for t in self.AUTO_TREND_TRIGGERS):
            if 'media' not in self.tools:
                return "⚠️ Medya modülü aktif değil, trend verisi alınamıyor."
            trends = self._get_trends(self.tools['media'])
            return (
                f"🌬️ POYRAZ GÜNDEM RAPORU\n"
                f"{'═' * 35}\n"
                f"{trends if trends else 'Trend verisi alınamadı.'}\n"
                f"{'═' * 35}"
            )

        # 2. Haberler
        if any(t in text_lower for t in self.AUTO_NEWS_TRIGGERS):
            if 'media' not in self.tools:
                return "⚠️ Medya modülü aktif değil."
            daily = self._get_daily_briefing(self.tools['media'])
            return (
                f"🌬️ POYRAZ HABER BRİFİNGİ\n"
                f"{'═' * 35}\n"
                f"{daily if daily else 'Günlük brifing alınamadı.'}\n"
                f"{'═' * 35}"
            )

        # 3. Sosyal medya / Rakip analizi
        if any(t in text_lower for t in self.AUTO_SOCIAL_TRIGGERS):
            report = self.get_social_health()
            return (
                f"🌬️ POYRAZ SOSYAL MEDYA RAPORU\n"
                f"{'═' * 35}\n"
                f"{report}\n"
                f"{'═' * 35}"
            )

        # 4. İçerik önerisi
        if any(t in text_lower for t in self.AUTO_CONTENT_TRIGGERS):
            if self.access_level == AccessLevel.RESTRICTED:
                return (
                    "🔒 Kısıtlı modda içerik stratejisi üretemem.\n"
                    "Sandbox veya Tam Erişim modu gereklidir."
                )
            # Güncel trendi içerik fikrine dönüştür
            trend_text = "Güncel Trend"
            if 'media' in self.tools:
                cached = self._get_trends(self.tools['media'])
                if cached:
                    # İlk satırı trend olarak kullan
                    trend_text = cached.split('\n')[0].strip()[:50]

            # Hangi içerik tipi isteniyor?
            content_type = ContentType.POST
            if "reel" in text_lower:
                content_type = ContentType.REEL
            elif "story" in text_lower:
                content_type = ContentType.STORY
            elif "makale" in text_lower:
                content_type = ContentType.ARTICLE

            idea = self.generate_content_idea(trend_text, content_type)
            hashtags = " ".join(idea.hashtags)

            return (
                f"🌬️ POYRAZ İÇERİK FİKRİ\n"
                f"{'═' * 35}\n"
                f"📌 Başlık: {idea.title}\n"
                f"📋 Tip: {idea.content_type.value}\n"
                f"📝 Açıklama: {idea.description}\n"
                f"🏷️ Hashtagler: {hashtags}\n"
                f"👥 Hedef Kitle: {idea.target_audience}\n"
                f"📊 Tahmini Etkileşim: {idea.estimated_engagement}\n"
                f"{'═' * 35}"
            )

        # 5. Duygu / Sentiment analizi
        if any(t in text_lower for t in self.AUTO_SENTIMENT_TRIGGERS):
            # Analiz edilecek metin tırnaklar içinde mi?
            match = re.search(r'["\'](.+?)["\']', text)
            if match:
                sample_text = match.group(1)
                result = self.analyze_sentiment(sample_text)
                return (
                    f"🌬️ POYRAZ DUYGU ANALİZİ ({result.processed_on})\n"
                    f"{'═' * 35}\n"
                    f"📝 Metin: {result.text[:80]}...\n"
                    f"{result.sentiment.emoji} Duygu: {result.sentiment.value}\n"
                    f"📊 Güven: %{result.confidence * 100:.0f}\n"
                    f"🔑 Anahtar: {', '.join(result.keywords) if result.keywords else 'Yok'}\n"
                    f"{'═' * 35}"
                )
            return (
                "🌬️ Duygu analizi için metni tırnak içinde yaz.\n"
                "Örnek: duygu analizi yap 'harika bir yer, çok lezzetli'"
            )

        # 6. Metrik / İstatistik
        if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
            return self._format_metrics_report()

        # Eşleşme yok — LLM devralır
        return None

    def _format_metrics_report(self) -> str:
        """Metrik raporunu formatla"""
        return (
            f"📊 POYRAZ MEDYA METRİKLERİ\n"
            f"{'═' * 35}\n"
            f"🔍 Haber Araması      : {self.metrics.news_searches}\n"
            f"😊 Duygu Analizi     : {self.metrics.sentiment_analyses}\n"
            f"🔥 Trend Kontrolü    : {self.metrics.trend_checks}\n"
            f"💡 İçerik Fikri      : {self.metrics.content_ideas_generated}\n"
            f"📱 Sosyal Kontrol    : {self.metrics.social_health_checks}\n"
            f"⚡ Auto-Handle       : {self.metrics.auto_handle_count}\n"
            f"{'═' * 35}"
        )

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

        # Erişim seviyesi bilgisi
        access_display = {
            AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi)",
            AccessLevel.SANDBOX: "📦 Sandbox (İçerik üretimi)",
            AccessLevel.FULL: "⚡ Tam Erişim"
        }.get(self.access_level, self.access_level)
        context_parts.append(f"🔐 ERİŞİM SEVİYESİ: {access_display}")

        with self.lock:
            if 'media' not in self.tools:
                context_parts.append("ℹ️ Medya modülü yüklü değil")
                return "\n".join(context_parts)

            media_tool = self.tools['media']

            # 1. Daily briefing
            daily_info = self._get_daily_briefing(media_tool)
            if daily_info:
                context_parts.append(daily_info)

            # 2. Trends
            trends = self._get_trends(media_tool)
            if trends:
                context_parts.append(f"\n🔥 ANLIK TRENDLER:\n{trends}")

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
        """Trendleri al (5 dk cache)"""
        if use_cache and self._is_cache_valid():
            return self._cached_trends

        try:
            if hasattr(media_tool, 'get_turkey_trends'):
                trends = media_tool.get_turkey_trends()
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

            if hasattr(media_tool, 'get_instagram_stats'):
                summaries.append(
                    "📸 INSTAGRAM:\n" + media_tool.get_instagram_stats()
                )

            if hasattr(media_tool, 'check_competitors'):
                summaries.append(
                    "\n🏁 RAKİP ANALİZİ:\n" + media_tool.check_competitors()
                )

            self.metrics.social_health_checks += 1

            return "\n".join(summaries) if summaries else "ℹ️ İstatistik verisi yok"

        except Exception:
            return ""

    def _is_cache_valid(self) -> bool:
        """Cache geçerli mi"""
        if not self._cached_trends or not self._cache_timestamp:
            return False
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_duration

    # ───────────────────────────────────────────────────────────
    # SENTIMENT ANALYSIS (Merkezi NLP Entegrasyonu)
    # ───────────────────────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Duygu analizi yap.
        Varsa merkezi NLP Yöneticisindeki (BERT Modeli) AI'yı kullanır.
        Yoksa kelime bazlı (Fallback) yönteme düşer.

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

        # Merkezi NLP modülü ile gerçek Yapay Zeka (AI) analizi
        if self.nlp and hasattr(self.nlp, "detect_emotion"):
            try:
                nlp_result = self.nlp.detect_emotion(text)
                
                # NLP Enum'ını Poyraz Enum'ına Map Et
                sentiment_map = {
                    "POZITIF": Sentiment.POSITIVE,
                    "NEGATIF": Sentiment.NEGATIVE,
                    "NOTR": Sentiment.NEUTRAL
                }
                mapped_sentiment = sentiment_map.get(nlp_result.sentiment.value, Sentiment.NEUTRAL)
                
                # Context için yine de keyword çıkart
                _, _, keywords = self._analyze_keywords(text)
                
                return SentimentAnalysis(
                    text=text[:100],
                    sentiment=mapped_sentiment,
                    confidence=nlp_result.confidence,
                    processed_on="Merkezi NLP Modülü (AI)",
                    keywords=keywords
                )
            except Exception as e:
                logger.error(f"NLP Modülü üzerinden analiz hatası: {e}")

        # Eğer NLP modülü çalışmazsa veya yoksa: Fallback Keyword Analysis
        sentiment, confidence, keywords = self._analyze_keywords(text)

        return SentimentAnalysis(
            text=text[:100],
            sentiment=sentiment,
            confidence=confidence,
            processed_on="CPU (Keyword Fallback)",
            keywords=keywords
        )

    def _analyze_keywords(
        self,
        text: str
    ) -> Tuple[Sentiment, float, List[str]]:
        """Keyword-based sentiment analysis (Fallback)"""
        text_lower = text.lower()

        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

        found_keywords = []
        found_keywords.extend([kw for kw in self.POSITIVE_KEYWORDS if kw in text_lower])
        found_keywords.extend([kw for kw in self.NEGATIVE_KEYWORDS if kw in text_lower])

        if positive_count > negative_count:
            sentiment = Sentiment.VERY_POSITIVE if positive_count >= 3 else Sentiment.POSITIVE
            confidence = 0.9 if positive_count >= 3 else 0.7
        elif negative_count > positive_count:
            sentiment = Sentiment.VERY_NEGATIVE if negative_count >= 3 else Sentiment.NEGATIVE
            confidence = 0.9 if negative_count >= 3 else 0.7
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

                sentiment = self.analyze_sentiment(result)

                self.metrics.news_searches += 1

                return (
                    f"{result}\n\n"
                    f"[POYRAZ ANALİZİ ({sentiment.processed_on})]:\n"
                    f"{sentiment.sentiment.emoji} Duygu: {sentiment.sentiment.value}\n"
                    f"Güven: %{sentiment.confidence * 100:.0f}\n"
                    f"Anahtar Kelimeler: "
                    f"{', '.join(sentiment.keywords) if sentiment.keywords else 'Yok'}"
                )

            except Exception as e:
                logger.error(f"Haber arama hatası: {e}")
                return f"❌ '{query}' araştırılırken hata: {str(e)[:100]}"

    # ───────────────────────────────────────────────────────────
    # SOCIAL MEDIA
    # ───────────────────────────────────────────────────────────

    def get_social_health(self) -> str:
        """Sosyal medya sağlığı raporu"""
        if 'media' not in self.tools:
            return "❌ Sosyal medya araçları aktif değil"

        with self.lock:
            try:
                media_tool = self.tools['media']
                stats = []

                if hasattr(media_tool, 'get_instagram_stats'):
                    stats.append(
                        "📸 INSTAGRAM:\n" + media_tool.get_instagram_stats()
                    )

                if hasattr(media_tool, 'check_competitors'):
                    stats.append(
                        "\n🏁 RAKİP ANALİZİ:\n" + media_tool.check_competitors()
                    )

                self.metrics.social_health_checks += 1

                return "\n".join(stats) if stats else "ℹ️ İstatistik verisi yok"

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
            },
            ContentType.ARTICLE: {
                "title": f"{trend} Hakkında Kapsamlı İnceleme",
                "description": "SEO odaklı, derinlemesine blog yazısı",
                "hashtags": [f"#{trend.replace(' ', '')}", "#Blog", "#İnceleme"],
                "target": "Araştırmacı okuyucular",
                "engagement": "Orta"
            },
            ContentType.TWEET: {
                "title": f"{trend} — Gündem Tweeti",
                "description": "Kısa, etkili, anında reaksiyon yaratan tweet",
                "hashtags": [f"#{trend.replace(' ', '')}", "#Gündem"],
                "target": "Twitter kullanıcıları",
                "engagement": "Değişken"
            }
        }

        template = templates.get(content_type, templates[ContentType.POST])

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
        """Poyraz karakter tanımı (LLM için)"""
        # Erişim seviyesi notu — diğer agent'larla tutarlı
        access_note = {
            AccessLevel.RESTRICTED: (
                "DİKKAT: Kısıtlı moddasın. "
                "Gündem ve trend bilgisi okuyabilirsin, içerik üretemezsin."
            ),
            AccessLevel.SANDBOX: (
                "DİKKAT: Sandbox modundasın. "
                "İçerik fikirleri üretebilirsin, sosyal medyaya gönderemezsin."
            ),
            AccessLevel.FULL: (
                "Tam erişim yetkin var. "
                "İçerik üretimi ve sosyal medya aksiyonlarını koordine edebilirsin."
            )
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

            f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

            "GÖREV:\n"
            "- Sadece bilgi verme, fırsata nasıl dönüşeceğini söyle\n"
            "- İçerik fikirleri üret\n"
            "- Rakip analizi yap\n"
            "- Sentiment analizi sunmak için veriyi yorumla\n\n"

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
        """Tool'ları güncelle"""
        with self.lock:
            self.tools.update(new_tools)
            logger.debug("Poyraz tool seti güncellendi")

    def get_status(self) -> str:
        """Poyraz durumu"""
        has_media = 'media' in self.tools

        # Donanım durumu SystemHealth'ten çekilebilir, ancak Poyraz için NLP durumu yeterlidir
        nlp_status = "✅ NLP (AI) Aktif" if self.nlp else "⚠️ Sadece Kelime (Fallback)"

        status = (
            "🟢 Aktif ve Gündemi İzliyor" if has_media
            else "🔴 Kısıtlı (Medya Modülü Yok)"
        )

        return (
            f"Poyraz: {status} | Analiz: {nlp_status} | "
            f"Erişim: {self.access_level.upper()}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Poyraz metrikleri"""
        return {
            "agent_name": self.agent_name,
            "access_level": self.access_level,
            "news_searches": self.metrics.news_searches,
            "sentiment_analyses": self.metrics.sentiment_analyses,
            "trend_checks": self.metrics.trend_checks,
            "content_ideas_generated": self.metrics.content_ideas_generated,
            "social_health_checks": self.metrics.social_health_checks,
            "auto_handle_count": self.metrics.auto_handle_count,
            "cache_valid": self._is_cache_valid(),
            "tools_available": list(self.tools.keys())
        }

    def clear_cache(self) -> None:
        """Cache temizle"""
        with self.lock:
            self._cached_trends = None
            self._cache_timestamp = None
            logger.debug("Trend cache temizlendi")


# """
# LotusAI agents/poyraz.py - Poyraz Agent
# Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
# Açıklama: Medya ve gündem uzmanı

# Sorumluluklar:
# - Gündem takibi (Google Trends, haberler)
# - Medya analizi (sosyal medya trendleri)
# - İçerik stratejisi
# - Rakip analizi
# - Sentiment analysis (GPU hızlandırmalı)
# - Haber araştırması
# - auto_handle: Medya ve gündem sorgularını LLM'e gitmeden anlık karşılama
# """

# import logging
# import threading
# from typing import Dict, Any, Optional, List, Tuple
# from dataclasses import dataclass
# from enum import Enum
# from datetime import datetime

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Poyraz")


# # ═══════════════════════════════════════════════════════════════
# # TORCH (GPU)
# # ═══════════════════════════════════════════════════════════════
# HAS_TORCH = False
# DEVICE_TYPE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
#         HAS_TORCH = True

#         if torch.cuda.is_available():
#             DEVICE_TYPE = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             DEVICE_TYPE = "mps"
#     except ImportError:
#         logger.warning("⚠️ Poyraz: Config GPU açık ama torch yok")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class Sentiment(Enum):
#     """Duygu analizi sonuçları"""
#     VERY_POSITIVE = "Çok Pozitif"
#     POSITIVE = "Pozitif"
#     NEUTRAL = "Nötr"
#     NEGATIVE = "Negatif"
#     VERY_NEGATIVE = "Çok Negatif"

#     @property
#     def emoji(self) -> str:
#         emojis = {
#             Sentiment.VERY_POSITIVE: "🤩",
#             Sentiment.POSITIVE: "😊",
#             Sentiment.NEUTRAL: "😐",
#             Sentiment.NEGATIVE: "😟",
#             Sentiment.VERY_NEGATIVE: "😡"
#         }
#         return emojis.get(self, "")


# class TrendType(Enum):
#     """Trend tipleri"""
#     VIRAL = "Viral"
#     RISING = "Yükseliyor"
#     STABLE = "Stabil"
#     DECLINING = "Düşüyor"


# class ContentType(Enum):
#     """İçerik tipleri"""
#     POST = "Post"
#     STORY = "Story"
#     REEL = "Reel"
#     ARTICLE = "Makale"
#     TWEET = "Tweet"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class TrendItem:
#     """Trend öğesi"""
#     title: str
#     trend_type: TrendType
#     relevance_score: float
#     timestamp: datetime


# @dataclass
# class ContentIdea:
#     """İçerik fikri"""
#     title: str
#     content_type: ContentType
#     description: str
#     hashtags: List[str]
#     target_audience: str
#     estimated_engagement: str


# @dataclass
# class SentimentAnalysis:
#     """Duygu analizi sonucu"""
#     text: str
#     sentiment: Sentiment
#     confidence: float
#     processed_on: str
#     keywords: List[str]


# @dataclass
# class PoyrazMetrics:
#     """Poyraz metrikleri"""
#     news_searches: int = 0
#     sentiment_analyses: int = 0
#     trend_checks: int = 0
#     content_ideas_generated: int = 0
#     social_health_checks: int = 0
#     auto_handle_count: int = 0


# # ═══════════════════════════════════════════════════════════════
# # POYRAZ AGENT
# # ═══════════════════════════════════════════════════════════════
# class PoyrazAgent:
#     """
#     Poyraz (Medya & Gündem Uzmanı)

#     Yetenekler:
#     - Gündem takibi: Google Trends, haberler
#     - Medya analizi: Sosyal medya trendleri
#     - Araştırmacı gazetecilik: Universal search
#     - İçerik stratejisti: Güncel olaylardan içerik fikirleri
#     - GPU analizi: Sentiment ve trend skorlaması
#     - auto_handle: Medya ve gündem sorgularını LLM'siz anlık karşılama

#     Erişim Seviyesi Kuralları (OpenClaw):
#     - restricted: Gündem ve trend bilgisi okuyabilir.
#     - sandbox: İçerik fikirleri üretebilir.
#     - full: Tüm yetkiler (sosyal medya paylaşımı dahil).

#     Poyraz, sistemin "dış dünya gözü"dür ve her şeyden haberdardır.
#     """

#     # Sentiment keywords
#     POSITIVE_KEYWORDS = [
#         "harika", "mükemmel", "süper", "güzel", "başarılı",
#         "iyi", "kaliteli", "lezzetli", "taze", "profesyonel"
#     ]

#     NEGATIVE_KEYWORDS = [
#         "kötü", "berbat", "yetersiz", "pahalı", "soğuk",
#         "tatsız", "geç", "kaba", "pis", "rezalet"
#     ]

#     # auto_handle tetikleyicileri
#     AUTO_TREND_TRIGGERS = [
#         "trendler", "gündem", "ne trend", "ne konuşuluyor",
#         "ne var ne yok", "gündemdekiler", "trending"
#     ]
#     AUTO_NEWS_TRIGGERS = [
#         "haberler", "son dakika", "bugün ne oldu",
#         "gelişmeler", "haber var mı"
#     ]
#     AUTO_SOCIAL_TRIGGERS = [
#         "sosyal medya durumu", "instagram durumu", "rakip analizi",
#         "sosyal sağlık", "takipçi"
#     ]
#     AUTO_CONTENT_TRIGGERS = [
#         "içerik öner", "post fikri", "ne paylaşayım", "reel fikri",
#         "story fikri", "içerik stratejisi"
#     ]
#     AUTO_METRIC_TRIGGERS = [
#         "poyraz raporu", "analiz sayısı", "istatistik",
#         "kaç haber", "performans"
#     ]
#     AUTO_SENTIMENT_TRIGGERS = [
#         "duygu analizi", "sentiment", "yorumları analiz et",
#         "ne düşünüyorlar", "yorum analizi"
#     ]

#     def __init__(
#         self,
#         nlp_manager: Optional[Any] = None,
#         tools_dict: Optional[Dict[str, Any]] = None,
#         access_level: Optional[str] = None
#     ):
#         """
#         Poyraz başlatıcı

#         Args:
#             nlp_manager: NLP yöneticisi (opsiyonel)
#             tools_dict: Engine'den gelen tool'lar
#             access_level: Erişim seviyesi (restricted, sandbox, full)
#         """
#         self.nlp = nlp_manager
#         self.tools = tools_dict or {}
#         self.agent_name = "POYRAZ"

#         # Parametre girilmezse Config'den oku
#         self.access_level = access_level or Config.ACCESS_LEVEL

#         # Thread safety
#         self.lock = threading.RLock()

#         # Hardware
#         self.device_type = DEVICE_TYPE
#         self.gpu_active = (DEVICE_TYPE != "cpu")

#         # Metrics
#         self.metrics = PoyrazMetrics()

#         # Cache (trendler için — 5 dakika)
#         self._cached_trends: Optional[str] = None
#         self._cache_timestamp: Optional[datetime] = None
#         self._cache_duration = 300.0

#         # Başlatma logu
#         if self.gpu_active and HAS_TORCH and DEVICE_TYPE == "cuda":
#             try:
#                 gpu_name = torch.cuda.get_device_name(0)
#                 logger.info(
#                     f"🌬️ {self.agent_name}: GPU aktif "
#                     f"({gpu_name}, Erişim: {self.access_level})"
#                 )
#             except Exception:
#                 logger.info(
#                     f"🌬️ {self.agent_name}: GPU aktif, Erişim: {self.access_level}"
#                 )
#         elif self.gpu_active:
#             logger.info(
#                 f"🌬️ {self.agent_name}: {DEVICE_TYPE.upper()} aktif, "
#                 f"Erişim: {self.access_level}"
#             )
#         else:
#             logger.info(
#                 f"🌬️ {self.agent_name}: CPU modunda, Erişim: {self.access_level}"
#             )

#         logger.info(f"🌬️ {self.agent_name} Gündem takip modülü başlatıldı")

#     # ───────────────────────────────────────────────────────────
#     # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
#     # ───────────────────────────────────────────────────────────

#     async def auto_handle(self, text: str) -> Optional[str]:
#         """
#         Kullanıcı metnini analiz ederek Poyraz'ın araçlarını
#         otomatik çalıştırır. engine.py tarafından çağrılır.

#         LLM'e gitmeden önce anlık, deterministik yanıt üretir.
#         Eşleşme yoksa None döner ve akış LLM'e devam eder.

#         Args:
#             text: Kullanıcı metni (temizlenmiş)

#         Returns:
#             Yanıt string veya None
#         """
#         text_lower = text.lower()
#         self.metrics.auto_handle_count += 1

#         # 1. Trend / Gündem sorgulama
#         if any(t in text_lower for t in self.AUTO_TREND_TRIGGERS):
#             if 'media' not in self.tools:
#                 return "⚠️ Medya modülü aktif değil, trend verisi alınamıyor."
#             trends = self._get_trends(self.tools['media'])
#             gpu_note = f" ({DEVICE_TYPE.upper()} hızlandırmalı)" if self.gpu_active else ""
#             return (
#                 f"🌬️ POYRAZ GÜNDEM RAPORU{gpu_note}\n"
#                 f"{'═' * 35}\n"
#                 f"{trends if trends else 'Trend verisi alınamadı.'}\n"
#                 f"{'═' * 35}"
#             )

#         # 2. Haberler
#         if any(t in text_lower for t in self.AUTO_NEWS_TRIGGERS):
#             if 'media' not in self.tools:
#                 return "⚠️ Medya modülü aktif değil."
#             daily = self._get_daily_briefing(self.tools['media'])
#             return (
#                 f"🌬️ POYRAZ HABER BRİFİNGİ\n"
#                 f"{'═' * 35}\n"
#                 f"{daily if daily else 'Günlük brifing alınamadı.'}\n"
#                 f"{'═' * 35}"
#             )

#         # 3. Sosyal medya / Rakip analizi
#         if any(t in text_lower for t in self.AUTO_SOCIAL_TRIGGERS):
#             report = self.get_social_health()
#             return (
#                 f"🌬️ POYRAZ SOSYAL MEDYA RAPORU\n"
#                 f"{'═' * 35}\n"
#                 f"{report}\n"
#                 f"{'═' * 35}"
#             )

#         # 4. İçerik önerisi
#         if any(t in text_lower for t in self.AUTO_CONTENT_TRIGGERS):
#             if self.access_level == AccessLevel.RESTRICTED:
#                 return (
#                     "🔒 Kısıtlı modda içerik stratejisi üretemem.\n"
#                     "Sandbox veya Tam Erişim modu gereklidir."
#                 )
#             # Güncel trendi içerik fikrine dönüştür
#             trend_text = "Güncel Trend"
#             if 'media' in self.tools:
#                 cached = self._get_trends(self.tools['media'])
#                 if cached:
#                     # İlk satırı trend olarak kullan
#                     trend_text = cached.split('\n')[0].strip()[:50]

#             # Hangi içerik tipi isteniyor?
#             content_type = ContentType.POST
#             if "reel" in text_lower:
#                 content_type = ContentType.REEL
#             elif "story" in text_lower:
#                 content_type = ContentType.STORY
#             elif "makale" in text_lower:
#                 content_type = ContentType.ARTICLE

#             idea = self.generate_content_idea(trend_text, content_type)
#             hashtags = " ".join(idea.hashtags)

#             return (
#                 f"🌬️ POYRAZ İÇERİK FİKRİ\n"
#                 f"{'═' * 35}\n"
#                 f"📌 Başlık: {idea.title}\n"
#                 f"📋 Tip: {idea.content_type.value}\n"
#                 f"📝 Açıklama: {idea.description}\n"
#                 f"🏷️ Hashtagler: {hashtags}\n"
#                 f"👥 Hedef Kitle: {idea.target_audience}\n"
#                 f"📊 Tahmini Etkileşim: {idea.estimated_engagement}\n"
#                 f"{'═' * 35}"
#             )

#         # 5. Duygu / Sentiment analizi
#         if any(t in text_lower for t in self.AUTO_SENTIMENT_TRIGGERS):
#             # Analiz edilecek metin tırnaklar içinde mi?
#             import re
#             match = re.search(r'["\'](.+?)["\']', text)
#             if match:
#                 sample_text = match.group(1)
#                 result = self.analyze_sentiment(sample_text)
#                 return (
#                     f"🌬️ POYRAZ DUYGU ANALİZİ ({result.processed_on})\n"
#                     f"{'═' * 35}\n"
#                     f"📝 Metin: {result.text[:80]}...\n"
#                     f"{result.sentiment.emoji} Duygu: {result.sentiment.value}\n"
#                     f"📊 Güven: %{result.confidence * 100:.0f}\n"
#                     f"🔑 Anahtar: {', '.join(result.keywords) if result.keywords else 'Yok'}\n"
#                     f"{'═' * 35}"
#                 )
#             return (
#                 "🌬️ Duygu analizi için metni tırnak içinde yaz.\n"
#                 "Örnek: duygu analizi yap 'harika bir yer, çok lezzetli'"
#             )

#         # 6. Metrik / İstatistik
#         if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
#             return self._format_metrics_report()

#         # Eşleşme yok — LLM devralır
#         return None

#     def _format_metrics_report(self) -> str:
#         """Metrik raporunu formatla"""
#         return (
#             f"📊 POYRAZ MEDYA METRİKLERİ\n"
#             f"{'═' * 35}\n"
#             f"🔍 Haber Araması      : {self.metrics.news_searches}\n"
#             f"😊 Duygu Analizi     : {self.metrics.sentiment_analyses}\n"
#             f"🔥 Trend Kontrolü    : {self.metrics.trend_checks}\n"
#             f"💡 İçerik Fikri      : {self.metrics.content_ideas_generated}\n"
#             f"📱 Sosyal Kontrol    : {self.metrics.social_health_checks}\n"
#             f"⚡ Auto-Handle       : {self.metrics.auto_handle_count}\n"
#             f"{'═' * 35}"
#         )

#     # ───────────────────────────────────────────────────────────
#     # CONTEXT GENERATION
#     # ───────────────────────────────────────────────────────────

#     def get_context_data(self) -> str:
#         """
#         Poyraz için günlük bağlam

#         Returns:
#             Context string
#         """
#         context_parts = ["\n[🌬️ POYRAZ GÜNDEM VE TREND RAPORU]"]

#         # Erişim seviyesi bilgisi
#         access_display = {
#             AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi)",
#             AccessLevel.SANDBOX: "📦 Sandbox (İçerik üretimi)",
#             AccessLevel.FULL: "⚡ Tam Erişim"
#         }.get(self.access_level, self.access_level)
#         context_parts.append(f"🔐 ERİŞİM SEVİYESİ: {access_display}")

#         with self.lock:
#             if 'media' not in self.tools:
#                 context_parts.append("ℹ️ Medya modülü yüklü değil")
#                 return "\n".join(context_parts)

#             media_tool = self.tools['media']

#             # 1. Daily briefing
#             daily_info = self._get_daily_briefing(media_tool)
#             if daily_info:
#                 context_parts.append(daily_info)

#             # 2. Trends
#             trends = self._get_trends(media_tool)
#             if trends:
#                 gpu_note = " (GPU hızlandırmalı)" if self.gpu_active else ""
#                 context_parts.append(f"\n🔥 ANLIK TRENDLER{gpu_note}:\n{trends}")

#             # 3. Social health
#             social = self._get_social_summary(media_tool)
#             if social:
#                 context_parts.append(f"\n📱 SOSYAL MEDYA:\n{social}")

#         context_parts.append(
#             "\n💡 POYRAZ NOTU:\n"
#             "Yukarıdaki trendleri kullanarak güncel sohbet başlat "
#             "veya sosyal medya aksiyonu öner."
#         )

#         return "\n".join(context_parts)

#     def _get_daily_briefing(self, media_tool: Any) -> str:
#         """Günlük brifing al"""
#         try:
#             if hasattr(media_tool, 'get_daily_context'):
#                 return media_tool.get_daily_context()
#         except Exception as e:
#             logger.error(f"Daily briefing hatası: {e}")
#         return ""

#     def _get_trends(self, media_tool: Any, use_cache: bool = True) -> str:
#         """Trendleri al (5 dk cache)"""
#         if use_cache and self._is_cache_valid():
#             return self._cached_trends

#         try:
#             if hasattr(media_tool, 'get_turkey_trends'):
#                 trends = media_tool.get_turkey_trends()
#                 self._cached_trends = trends
#                 self._cache_timestamp = datetime.now()
#                 self.metrics.trend_checks += 1
#                 return trends
#         except Exception as e:
#             logger.error(f"Trend alma hatası: {e}")

#         return ""

#     def _get_social_summary(self, media_tool: Any) -> str:
#         """Sosyal medya özeti"""
#         try:
#             summaries = []

#             if hasattr(media_tool, 'get_instagram_stats'):
#                 summaries.append(
#                     "📸 INSTAGRAM:\n" + media_tool.get_instagram_stats()
#                 )

#             if hasattr(media_tool, 'check_competitors'):
#                 summaries.append(
#                     "\n🏁 RAKİP ANALİZİ:\n" + media_tool.check_competitors()
#                 )

#             self.metrics.social_health_checks += 1

#             return "\n".join(summaries) if summaries else "ℹ️ İstatistik verisi yok"

#         except Exception:
#             return ""

#     def _is_cache_valid(self) -> bool:
#         """Cache geçerli mi"""
#         if not self._cached_trends or not self._cache_timestamp:
#             return False
#         elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
#         return elapsed < self._cache_duration

#     # ───────────────────────────────────────────────────────────
#     # SENTIMENT ANALYSIS
#     # ───────────────────────────────────────────────────────────

#     def analyze_sentiment(self, text: str) -> SentimentAnalysis:
#         """
#         Duygu analizi yap

#         Args:
#             text: Analiz edilecek metin

#         Returns:
#             SentimentAnalysis objesi
#         """
#         if not text:
#             return SentimentAnalysis(
#                 text="",
#                 sentiment=Sentiment.NEUTRAL,
#                 confidence=0.0,
#                 processed_on="None",
#                 keywords=[]
#             )

#         self.metrics.sentiment_analyses += 1
#         processing_unit = self._get_processing_unit()
#         sentiment, confidence, keywords = self._analyze_keywords(text)

#         return SentimentAnalysis(
#             text=text[:100],
#             sentiment=sentiment,
#             confidence=confidence,
#             processed_on=processing_unit,
#             keywords=keywords
#         )

#     def _get_processing_unit(self) -> str:
#         """İşlem birimini belirle"""
#         if not self.gpu_active or not HAS_TORCH:
#             return "CPU"

#         try:
#             dummy = torch.tensor([1.0]).to(self.device_type)
#             if dummy.is_cuda:
#                 return "CUDA Core"
#             elif self.device_type == "mps":
#                 return "MPS Core"
#         except Exception:
#             pass

#         return "CPU"

#     def _analyze_keywords(
#         self,
#         text: str
#     ) -> Tuple[Sentiment, float, List[str]]:
#         """Keyword-based sentiment analysis"""
#         text_lower = text.lower()

#         positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
#         negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

#         found_keywords = []
#         found_keywords.extend([kw for kw in self.POSITIVE_KEYWORDS if kw in text_lower])
#         found_keywords.extend([kw for kw in self.NEGATIVE_KEYWORDS if kw in text_lower])

#         if positive_count > negative_count:
#             sentiment = Sentiment.VERY_POSITIVE if positive_count >= 3 else Sentiment.POSITIVE
#             confidence = 0.9 if positive_count >= 3 else 0.7
#         elif negative_count > positive_count:
#             sentiment = Sentiment.VERY_NEGATIVE if negative_count >= 3 else Sentiment.NEGATIVE
#             confidence = 0.9 if negative_count >= 3 else 0.7
#         else:
#             sentiment = Sentiment.NEUTRAL
#             confidence = 0.5

#         return sentiment, confidence, found_keywords[:5]

#     # ───────────────────────────────────────────────────────────
#     # NEWS SEARCH
#     # ───────────────────────────────────────────────────────────

#     def search_news(self, query: str) -> str:
#         """
#         Haber araştırması yap

#         Args:
#             query: Arama sorgusu

#         Returns:
#             Arama sonuçları
#         """
#         if 'media' not in self.tools:
#             return "❌ Medya araştırma araçları aktif değil"

#         with self.lock:
#             try:
#                 media_tool = self.tools['media']

#                 if not hasattr(media_tool, 'universal_search'):
#                     return "❌ universal_search metodu yok"

#                 logger.info(f"Poyraz araştırıyor: {query}")
#                 result = media_tool.universal_search(query)

#                 sentiment = self.analyze_sentiment(result)

#                 self.metrics.news_searches += 1

#                 return (
#                     f"{result}\n\n"
#                     f"[POYRAZ ANALİZİ ({sentiment.processed_on})]:\n"
#                     f"{sentiment.sentiment.emoji} Duygu: {sentiment.sentiment.value}\n"
#                     f"Güven: %{sentiment.confidence * 100:.0f}\n"
#                     f"Anahtar Kelimeler: "
#                     f"{', '.join(sentiment.keywords) if sentiment.keywords else 'Yok'}"
#                 )

#             except Exception as e:
#                 logger.error(f"Haber arama hatası: {e}")
#                 return f"❌ '{query}' araştırılırken hata: {str(e)[:100]}"

#     # ───────────────────────────────────────────────────────────
#     # SOCIAL MEDIA
#     # ───────────────────────────────────────────────────────────

#     def get_social_health(self) -> str:
#         """Sosyal medya sağlığı raporu"""
#         if 'media' not in self.tools:
#             return "❌ Sosyal medya araçları aktif değil"

#         with self.lock:
#             try:
#                 media_tool = self.tools['media']
#                 stats = []

#                 if hasattr(media_tool, 'get_instagram_stats'):
#                     stats.append(
#                         "📸 INSTAGRAM:\n" + media_tool.get_instagram_stats()
#                     )

#                 if hasattr(media_tool, 'check_competitors'):
#                     stats.append(
#                         "\n🏁 RAKİP ANALİZİ:\n" + media_tool.check_competitors()
#                     )

#                 self.metrics.social_health_checks += 1

#                 return "\n".join(stats) if stats else "ℹ️ İstatistik verisi yok"

#             except Exception as e:
#                 logger.error(f"Social health hatası: {e}")
#                 return f"❌ Veri çekilemedi: {str(e)[:100]}"

#     # ───────────────────────────────────────────────────────────
#     # CONTENT GENERATION
#     # ───────────────────────────────────────────────────────────

#     def generate_content_idea(
#         self,
#         trend: str,
#         content_type: ContentType
#     ) -> ContentIdea:
#         """
#         İçerik fikri üret

#         Args:
#             trend: Güncel trend
#             content_type: İçerik tipi

#         Returns:
#             ContentIdea objesi
#         """
#         self.metrics.content_ideas_generated += 1

#         templates = {
#             ContentType.POST: {
#                 "title": f"{trend} ile İlgili Özel İçerik",
#                 "description": (
#                     f"Gündemdeki {trend} konusunu markamızla "
#                     "ilişkilendiren yaratıcı bir post"
#                 ),
#                 "hashtags": [f"#{trend.replace(' ', '')}", "#LotusBağevi", "#Bursa"],
#                 "target": "Genç yetişkinler (25-40)",
#                 "engagement": "Orta-Yüksek"
#             },
#             ContentType.STORY: {
#                 "title": f"{trend} Güncel Story",
#                 "description": "Kısa, dinamik, etkileşimli story içeriği",
#                 "hashtags": [f"#{trend.replace(' ', '')}", "#GündemdeYiz"],
#                 "target": "18-35 yaş arası",
#                 "engagement": "Yüksek"
#             },
#             ContentType.REEL: {
#                 "title": f"{trend} Trend Reel",
#                 "description": "Viral potansiyeli yüksek, müzikli reel",
#                 "hashtags": [f"#{trend.replace(' ', '')}", "#Viral", "#Keşfet"],
#                 "target": "Geniş kitle",
#                 "engagement": "Çok Yüksek"
#             },
#             ContentType.ARTICLE: {
#                 "title": f"{trend} Hakkında Kapsamlı İnceleme",
#                 "description": "SEO odaklı, derinlemesine blog yazısı",
#                 "hashtags": [f"#{trend.replace(' ', '')}", "#Blog", "#İnceleme"],
#                 "target": "Araştırmacı okuyucular",
#                 "engagement": "Orta"
#             },
#             ContentType.TWEET: {
#                 "title": f"{trend} — Gündem Tweeti",
#                 "description": "Kısa, etkili, anında reaksiyon yaratan tweet",
#                 "hashtags": [f"#{trend.replace(' ', '')}", "#Gündem"],
#                 "target": "Twitter kullanıcıları",
#                 "engagement": "Değişken"
#             }
#         }

#         template = templates.get(content_type, templates[ContentType.POST])

#         return ContentIdea(
#             title=template["title"],
#             content_type=content_type,
#             description=template["description"],
#             hashtags=template["hashtags"],
#             target_audience=template["target"],
#             estimated_engagement=template["engagement"]
#         )

#     # ───────────────────────────────────────────────────────────
#     # SYSTEM PROMPT
#     # ───────────────────────────────────────────────────────────

#     def get_system_prompt(self) -> str:
#         """Poyraz karakter tanımı (LLM için)"""
#         # Erişim seviyesi notu — diğer agent'larla tutarlı
#         access_note = {
#             AccessLevel.RESTRICTED: (
#                 "DİKKAT: Kısıtlı moddasın. "
#                 "Gündem ve trend bilgisi okuyabilirsin, içerik üretemezsin."
#             ),
#             AccessLevel.SANDBOX: (
#                 "DİKKAT: Sandbox modundasın. "
#                 "İçerik fikirleri üretebilirsin, sosyal medyaya gönderemezsin."
#             ),
#             AccessLevel.FULL: (
#                 "Tam erişim yetkin var. "
#                 "İçerik üretimi ve sosyal medya aksiyonlarını koordine edebilirsin."
#             )
#         }.get(self.access_level, "")

#         return (
#             f"Sen {Config.PROJECT_NAME} sisteminin enerjik, meraklı ve "
#             f"her şeyden haberdar olan Medya Uzmanı POYRAZ'sın.\n\n"

#             "KARAKTER:\n"
#             "- Bir rüzgar gibi hızlı\n"
#             "- Bilgiyi anında yakalayan\n"
#             "- Sosyal medya diline hakim\n"
#             "- Araştırmacı ve meraklı\n"
#             "- Dinamik ve heyecan verici\n\n"

#             "MİSYON:\n"
#             "- Türkiye ve Bursa gündemini takip et\n"
#             "- Sosyal medya trendlerini izle\n"
#             "- Önemli haberleri Halil Bey'e bildir\n"
#             "- Fırsatları tespit et\n\n"

#             f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

#             "GÖREV:\n"
#             "- Sadece bilgi verme, fırsata nasıl dönüşeceğini söyle\n"
#             "- İçerik fikirleri üret\n"
#             "- Rakip analizi yap\n"
#             "- Sentiment analizi sunmak için veriyi yorumla\n\n"

#             "DİL VE ÜSLUP:\n"
#             "- 'Bunu duydunuz mu?' tarzı girişler\n"
#             "- 'Bugün şu çok popüler!' gibi vurgulamalar\n"
#             "- Dinamik ve bilgi dolu konuşma\n"
#             "- Heyecan verici ton\n"
#         )

#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────

#     def update_tools(self, new_tools: Dict[str, Any]) -> None:
#         """Tool'ları güncelle"""
#         with self.lock:
#             self.tools.update(new_tools)
#             logger.debug("Poyraz tool seti güncellendi")

#     def get_status(self) -> str:
#         """Poyraz durumu"""
#         has_media = 'media' in self.tools

#         gpu_name = "Bilinmiyor"
#         if self.gpu_active and HAS_TORCH and DEVICE_TYPE == "cuda":
#             try:
#                 gpu_name = torch.cuda.get_device_name(0)
#             except Exception:
#                 pass

#         gpu_status = f"✅ GPU ({gpu_name})" if self.gpu_active else "⚠️ CPU"

#         status = (
#             "🟢 Aktif ve Gündemi İzliyor" if has_media
#             else "🔴 Kısıtlı (Medya Modülü Yok)"
#         )

#         return (
#             f"Poyraz: {status} | Donanım: {gpu_status} | "
#             f"Erişim: {self.access_level}"
#         )

#     def get_metrics(self) -> Dict[str, Any]:
#         """Poyraz metrikleri"""
#         return {
#             "agent_name": self.agent_name,
#             "device": self.device_type,
#             "access_level": self.access_level,
#             "news_searches": self.metrics.news_searches,
#             "sentiment_analyses": self.metrics.sentiment_analyses,
#             "trend_checks": self.metrics.trend_checks,
#             "content_ideas_generated": self.metrics.content_ideas_generated,
#             "social_health_checks": self.metrics.social_health_checks,
#             "auto_handle_count": self.metrics.auto_handle_count,
#             "cache_valid": self._is_cache_valid(),
#             "tools_available": list(self.tools.keys())
#         }

#     def clear_cache(self) -> None:
#         """Cache temizle"""
#         with self.lock:
#             self._cached_trends = None
#             self._cache_timestamp = None
#             logger.debug("Trend cache temizlendi")