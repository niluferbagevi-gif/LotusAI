"""
LotusAI agents/kurt.py - Kurt Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu & Mimari Optimizasyon)
Açıklama: Finans ve borsa stratejisti

Sorumluluklar:
- Piyasa analizi (kripto, borsa)
- Likidite yönetimi
- Risk analizi
- Yatırım önerileri
- Kasa izleme
- Teknik analiz (Merkezi FinanceManager üzerinden)
- auto_handle: Finansal sorguları LLM'e gitmeden anlık karşılama
"""

import re
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Kurt")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class MarketTrend(Enum):
    """Piyasa trendleri"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"

    @property
    def emoji(self) -> str:
        emojis = {
            MarketTrend.BULLISH: "📈",
            MarketTrend.BEARISH: "📉",
            MarketTrend.SIDEWAYS: "➡️",
            MarketTrend.VOLATILE: "🎢",
            MarketTrend.UNKNOWN: "❓"
        }
        return emojis.get(self, "")


class LiquidityLevel(Enum):
    """Likidite seviyeleri"""
    CRITICAL = "KRİTİK"
    LOW = "DÜŞÜK"
    MODERATE = "ORTA"
    HIGH = "YÜKSEK"
    EXCELLENT = "MÜKEMMEL"

    @property
    def color(self) -> str:
        colors = {
            LiquidityLevel.CRITICAL: "🔴",
            LiquidityLevel.LOW: "🟠",
            LiquidityLevel.MODERATE: "🟡",
            LiquidityLevel.HIGH: "🟢",
            LiquidityLevel.EXCELLENT: "💎"
        }
        return colors.get(self, "⚪")


class InvestmentAdvice(Enum):
    """Yatırım tavsiyeleri"""
    STRONG_BUY = "Güçlü Al"
    BUY = "Al"
    HOLD = "Tut"
    SELL = "Sat"
    STRONG_SELL = "Güçlü Sat"
    WAIT = "Bekle"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class LiquidityStatus:
    """Likidite durumu"""
    balance: float
    level: LiquidityLevel
    message: str
    risk_alert: bool = False


@dataclass
class MarketAnalysis:
    """Piyasa analizi"""
    asset: str
    trend: MarketTrend
    advice: InvestmentAdvice
    report: str
    chart_file: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class KurtMetrics:
    """Kurt metrikleri"""
    analyses_performed: int = 0
    market_checks: int = 0
    liquidity_alerts: int = 0
    investment_advices: int = 0
    total_balance_tracked: float = 0.0
    auto_handle_count: int = 0


# ═══════════════════════════════════════════════════════════════
# KURT AGENT
# ═══════════════════════════════════════════════════════════════
class KurtAgent:
    """
    Kurt (Finans & Borsa Stratejisti)

    Yetenekler:
    - Piyasa analizi: Kripto ve borsa trend tahmini
    - Kasa denetimi: Nakit akışı ve likidite yönetimi
    - Stratejik tavsiye: Kâr odaklı finansal yorumlar
    - Donanım farkındalığı: Ağır hesaplamaları merkezi Manager'lara bırakır
    - Risk yönetimi: Likidite ve volatilite takibi
    - auto_handle: Finansal sorguları LLM'siz anlık karşılama

    Erişim Seviyesi Kuralları (OpenClaw):
    - restricted: Piyasa ve kasa verilerini okuyabilir, işlem yapamaz.
    - sandbox: Analiz ve öneri yapabilir.
    - full: Tüm yetkiler (gelecekte işlem emri vb.).

    Kurt, sistemin "finans kurdu"dur ve para konusunda taviz vermez.
    """

    # Liquidity thresholds (TL)
    LIQUIDITY_THRESHOLDS = {
        LiquidityLevel.CRITICAL: 0.0,
        LiquidityLevel.LOW: 5000.0,
        LiquidityLevel.MODERATE: 10000.0,
        LiquidityLevel.HIGH: 20000.0,
        LiquidityLevel.EXCELLENT: 50000.0
    }

    # auto_handle tetikleyicileri
    AUTO_LIQUIDITY_TRIGGERS = [
        "kasa durumu", "likidite", "bakiye", "nakit", "para durumu",
        "kasada ne kadar", "bakiyemiz", "ne kadar param var"
    ]
    AUTO_MARKET_TRIGGERS = [
        "piyasa", "borsa", "kripto", "bitcoin", "btc", "eth",
        "dolar", "euro", "döviz", "piyasa durumu", "market"
    ]
    AUTO_METRIC_TRIGGERS = [
        "finansal özet", "kurt raporu", "analiz sayısı",
        "kaç analiz", "performans", "istatistik"
    ]
    AUTO_RISK_TRIGGERS = [
        "risk", "tehlike", "uyarı", "alarm", "kritik durum"
    ]

    def __init__(self, tools_dict: Dict[str, Any], access_level: Optional[str] = None):
        """
        Kurt başlatıcı

        Args:
            tools_dict: Engine'den gelen tool'lar
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.tools = tools_dict

        # Parametre girilmezse Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL

        self.agent_name = "KURT"

        # Thread safety
        self.lock = threading.RLock()

        # Hardware Info (Artık doğrudan merkezi yapıdan alıyoruz)
        self.gpu_available = Config.USE_GPU

        # Configuration
        self.min_liquidity = getattr(
            Config,
            'MIN_LIQUIDITY_LIMIT',
            self.LIQUIDITY_THRESHOLDS[LiquidityLevel.LOW]
        )

        # Metrics
        self.metrics = KurtMetrics()

        # Market cache
        self._cached_market_data: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = 60.0  # saniye

        status = "🚀 GPU (Hızlandırılmış)" if self.gpu_available else "⚙️ CPU"

        logger.info(
            f"🐺 {self.agent_name} başlatıldı ({status}, Erişim: {self.access_level.upper()}). "
            "Piyasalar izleniyor..."
        )

    # ───────────────────────────────────────────────────────────
    # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
    # ───────────────────────────────────────────────────────────

    async def auto_handle(self, text: str) -> Optional[str]:
        """
        Kullanıcı metnini analiz ederek Kurt'un araçlarını
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

        # 1. Kasa / Likidite durumu
        if any(t in text_lower for t in self.AUTO_LIQUIDITY_TRIGGERS):
            liquidity = self.check_liquidity()
            return (
                f"🐺 KURT KASA RAPORU\n"
                f"{'═' * 35}\n"
                f"{liquidity.level.color} Seviye: {liquidity.level.value}\n"
                f"{liquidity.message}\n"
                f"{'═' * 35}\n"
                f"{'🚨 RİSK UYARISI AKTİF!' if liquidity.risk_alert else '✅ Risk yok'}"
            )

        # 2. Piyasa / Borsa durumu
        if any(t in text_lower for t in self.AUTO_MARKET_TRIGGERS):
            # Belirli varlık ismi var mı?
            asset_match = re.search(
                r'\b(btc|eth|bitcoin|ethereum|xrp|sol|bnb|[a-z]{2,5}/usdt)\b',
                text_lower
            )
            if asset_match:
                asset = asset_match.group(1).upper()
                analysis = self.analyze_asset(asset)
                return (
                    f"🐺 KURT VARLİK ANALİZİ: {analysis.asset}\n"
                    f"{'═' * 40}\n"
                    f"{analysis.trend.emoji} Trend: {analysis.trend.value}\n"
                    f"📌 Tavsiye: {analysis.advice.value}\n"
                    f"{'═' * 40}\n"
                    f"{analysis.report}"
                )
            else:
                # Genel piyasa özeti
                market = self.get_market_analysis()
                return f"🐺 KURT PİYASA RAPORU\n{'═' * 35}\n{market}"

        # 3. Risk uyarısı sorgulama
        if any(t in text_lower for t in self.AUTO_RISK_TRIGGERS):
            liquidity = self.check_liquidity()
            ops_link = self.get_ops_finance_link()
            risk_lines = ["🐺 KURT RİSK DEĞERLENDİRMESİ", "═" * 35]

            if liquidity.risk_alert:
                risk_lines.append(f"🚨 LİKİDİTE RİSKİ: {liquidity.message}")
            else:
                risk_lines.append("✅ Likidite riski yok")

            if ops_link:
                risk_lines.append(f"📦 OPS RİSKİ: {ops_link}")

            risk_lines.append("═" * 35)
            return "\n".join(risk_lines)

        # 4. Metrik / İstatistik sorgusu
        if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
            return self._format_metrics_report()

        # Eşleşme yok — LLM devralır
        return None

    def _format_metrics_report(self) -> str:
        """Metrik raporunu formatla"""
        return (
            f"📊 KURT FİNANS METRİKLERİ\n"
            f"{'═' * 35}\n"
            f"📈 Yapılan Analiz       : {self.metrics.analyses_performed}\n"
            f"🌍 Piyasa Kontrolü     : {self.metrics.market_checks}\n"
            f"⚠️ Likidite Uyarısı   : {self.metrics.liquidity_alerts}\n"
            f"💡 Yatırım Tavsiyesi  : {self.metrics.investment_advices}\n"
            f"💰 İzlenen Bakiye     : {self.metrics.total_balance_tracked:,.2f} TL\n"
            f"{'═' * 35}"
        )

    # ───────────────────────────────────────────────────────────
    # MARKET ANALYSIS
    # ───────────────────────────────────────────────────────────

    def get_market_analysis(self, use_cache: bool = True) -> str:
        """
        Piyasa analizi getir

        Args:
            use_cache: Cache kullan (60sn geçerli)

        Returns:
            Market summary
        """
        if use_cache and self._is_cache_valid():
            return self._cached_market_data

        if 'finance' not in self.tools:
            return "⚠️ Piyasa analiz araçları ulaşılamıyor"

        with self.lock:
            try:
                fin_tool = self.tools['finance']
                market_summary = fin_tool.get_market_summary()

                if "Hata" in market_summary or not market_summary:
                    return "❌ Piyasa veri akışı kesildi"

                self._cached_market_data = market_summary
                self._cache_timestamp = datetime.now()
                self.metrics.market_checks += 1

                return market_summary

            except Exception as e:
                logger.error(f"Piyasa analiz hatası: {e}")
                return "📉 Piyasa verileri işlenemedi"

    def _is_cache_valid(self) -> bool:
        """Cache geçerli mi"""
        if not self._cached_market_data or not self._cache_timestamp:
            return False
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_duration

    def analyze_asset(self, asset_name: str) -> MarketAnalysis:
        """
        Varlık analizi yap

        Args:
            asset_name: Varlık adı (BTC, ETH, vb.)

        Returns:
            MarketAnalysis objesi
        """
        if 'finance' not in self.tools:
            return MarketAnalysis(
                asset=asset_name,
                trend=MarketTrend.UNKNOWN,
                advice=InvestmentAdvice.WAIT,
                report="Finansal araçlar aktif değil"
            )

        with self.lock:
            try:
                symbol = asset_name.upper()
                if "/" not in symbol:
                    symbol += "/USDT"

                report, chart_file = self.tools['finance'].analyze(symbol=symbol)

                trend = self._detect_trend(report)
                advice = self._generate_advice(trend, report)
                strategic_note = self._generate_strategic_note(trend)

                full_report = (
                    f"{report}\n\n"
                    f"🐺 KURT'UN NOTU:\n{strategic_note}\n\n"
                    f"📊 Grafik: {chart_file if chart_file else 'Üretilemedi'}"
                )

                self.metrics.analyses_performed += 1
                self.metrics.investment_advices += 1

                return MarketAnalysis(
                    asset=symbol,
                    trend=trend,
                    advice=advice,
                    report=full_report,
                    chart_file=chart_file
                )

            except Exception as e:
                logger.error(f"Varlık analiz hatası ({asset_name}): {e}")
                return MarketAnalysis(
                    asset=asset_name,
                    trend=MarketTrend.UNKNOWN,
                    advice=InvestmentAdvice.WAIT,
                    report=f"❌ Teknik analiz yapılamadı: {str(e)[:100]}"
                )

    def _detect_trend(self, report: str) -> MarketTrend:
        """Rapor içeriğinden trend tespit et"""
        report_upper = report.upper()

        if "BULLISH" in report_upper or "YÜKSELİŞ" in report_upper:
            return MarketTrend.BULLISH
        if "BEARISH" in report_upper or "DÜŞÜŞ" in report_upper:
            return MarketTrend.BEARISH
        if "SIDEWAYS" in report_upper or "YATAY" in report_upper:
            return MarketTrend.SIDEWAYS
        if "VOLATILE" in report_upper or "VOLATİL" in report_upper:
            return MarketTrend.VOLATILE

        return MarketTrend.UNKNOWN

    def _generate_advice(self, trend: MarketTrend, report: str) -> InvestmentAdvice:
        """Trend'e göre tavsiye üret"""
        advice_map = {
            MarketTrend.BULLISH: InvestmentAdvice.BUY,
            MarketTrend.BEARISH: InvestmentAdvice.SELL,
            MarketTrend.SIDEWAYS: InvestmentAdvice.HOLD,
            MarketTrend.VOLATILE: InvestmentAdvice.WAIT,
            MarketTrend.UNKNOWN: InvestmentAdvice.HOLD
        }
        return advice_map.get(trend, InvestmentAdvice.HOLD)

    def _generate_strategic_note(self, trend: MarketTrend) -> str:
        """Stratejik not üret"""
        notes = {
            MarketTrend.BULLISH: (
                "Trend yukarı yönlü. Direnç seviyeleri takip edilerek "
                "pozisyon korunabilir. Risk/ödül oranı olumlu."
            ),
            MarketTrend.BEARISH: (
                "Piyasa yorgun görünüyor. Nakitte kalmak veya "
                "stop-loss kullanmak akıllıca olur. Düşüş devam edebilir."
            ),
            MarketTrend.SIDEWAYS: (
                "Piyasa konsolidasyon aşamasında. Kırılım beklemek "
                "mantıklı. Yön belli olmadan büyük pozisyon riskli."
            ),
            MarketTrend.VOLATILE: (
                "Yüksek volatilite var. Risk yönetimi kritik. "
                "Pozisyon büyüklüğü küçük tutulmalı veya beklenebilir."
            ),
            MarketTrend.UNKNOWN: (
                "Yeterli veri yok. Daha fazla analiz gerekli. "
                "Acele karar vermemek önemli."
            )
        }
        return notes.get(trend, "Durum belirsiz, dikkatli olunmalı.")

    # ───────────────────────────────────────────────────────────
    # LIQUIDITY MANAGEMENT
    # ───────────────────────────────────────────────────────────

    def check_liquidity(self) -> LiquidityStatus:
        """
        Likidite durumunu kontrol et

        Returns:
            LiquidityStatus objesi
        """
        acc_tool = self.tools.get('accounting') or self.tools.get('finance')

        if not acc_tool or not hasattr(acc_tool, 'get_balance'):
            return LiquidityStatus(
                balance=0.0,
                level=LiquidityLevel.CRITICAL,
                message="⚠️ Muhasebe/Finans modülü mevcut değil",
                risk_alert=True
            )

        with self.lock:
            try:
                balance_str = acc_tool.get_balance()
                balance = self._parse_balance(balance_str)

                level = self._assess_liquidity_level(balance)
                message = self._generate_liquidity_message(balance, level)

                risk_alert = level in {LiquidityLevel.CRITICAL, LiquidityLevel.LOW}

                if risk_alert:
                    self.metrics.liquidity_alerts += 1

                self.metrics.total_balance_tracked = balance

                return LiquidityStatus(
                    balance=balance,
                    level=level,
                    message=message,
                    risk_alert=risk_alert
                )

            except Exception as e:
                logger.error(f"Likidite kontrolü hatası: {e}")
                return LiquidityStatus(
                    balance=0.0,
                    level=LiquidityLevel.CRITICAL,
                    message=f"❌ Bakiye okunamadı: {str(e)[:50]}",
                    risk_alert=True
                )

    def _assess_liquidity_level(self, balance: float) -> LiquidityLevel:
        """Likidite seviyesi değerlendir"""
        if balance <= self.LIQUIDITY_THRESHOLDS[LiquidityLevel.CRITICAL]:
            return LiquidityLevel.CRITICAL
        if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.LOW]:
            return LiquidityLevel.LOW
        if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.MODERATE]:
            return LiquidityLevel.MODERATE
        if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.HIGH]:
            return LiquidityLevel.HIGH
        return LiquidityLevel.EXCELLENT

    def _generate_liquidity_message(self, balance: float, level: LiquidityLevel) -> str:
        """Likidite mesajı üret"""
        messages = {
            LiquidityLevel.CRITICAL: (
                f"🚨 ACİL DURUM: Kasa ekside veya sıfır ({balance:,.2f} TL)! "
                "Tüm harcamalar dondurulmalı!"
            ),
            LiquidityLevel.LOW: (
                f"⚠️ DÜŞÜK LİKİDİTE: Nakit rezervi {self.min_liquidity:,.0f} TL "
                "altında. Savunma moduna geçilmeli."
            ),
            LiquidityLevel.MODERATE: (
                f"🟡 NORMAL SEVİYE: Bakiye {balance:,.2f} TL. "
                "İhtiyatlı harcama yapılabilir."
            ),
            LiquidityLevel.HIGH: (
                f"✅ GÜÇLÜ LİKİDİTE: {balance:,.2f} TL mevcut. "
                "Yatırım fırsatları değerlendirilebilir."
            ),
            LiquidityLevel.EXCELLENT: (
                f"💎 MÜKEMMEL DURUM: {balance:,.2f} TL ile çok güçlü "
                "pozisyon. Büyüme stratejileri uygulanabilir."
            )
        }
        return messages.get(level, f"Bakiye: {balance:,.2f} TL")

    def _parse_balance(self, balance_val: Any) -> float:
        """Bakiye parse"""
        if isinstance(balance_val, (int, float)):
            return float(balance_val)

        try:
            clean = str(balance_val).lower()
            for sym in ["tl", "try", "₺", "$", "€", "£"]:
                clean = clean.replace(sym, "")
            clean = clean.replace(",", ".")
            clean = "".join(c for c in clean if c.isdigit() or c in ".,-")
            return float(clean) if clean else 0.0
        except Exception:
            return 0.0

    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────

    def get_context_data(self) -> str:
        """
        Kurt için finansal 'savaş odası' bağlamı

        Returns:
            Context string
        """
        context_parts = ["\n[🐺 KURT STRATEJİ VE RİSK ANALİZİ]"]

        # Donanım bilgisi (Merkezi System Health üzerinden)
        hw_status = "🐢 Standart CPU"
        if self.gpu_available:
            hw_status = "⚡ Yüksek Performanslı GPU"
            if 'system' in self.tools:
                try:
                    sys_hw = self.tools['system'].get_hardware_status()
                    if sys_hw.gpu_available and sys_hw.gpu_info:
                        hw_status = f"⚡ Yüksek Performanslı GPU ({sys_hw.gpu_info[0].name})"
                except Exception:
                    pass
        
        context_parts.append(f"SİSTEM: {hw_status}")

        # Erişim seviyesi bilgisi
        access_display = {
            AccessLevel.RESTRICTED: "🔒 Kısıtlı (Okuma)",
            AccessLevel.SANDBOX: "📦 Sandbox (Analiz)",
            AccessLevel.FULL: "⚡ Tam Erişim"
        }.get(self.access_level, self.access_level)
        context_parts.append(f"🔐 ERİŞİM: {access_display}")

        with self.lock:
            # 1. Market analysis
            market = self.get_market_analysis()
            context_parts.append(f"\n🌍 KÜRESEL PİYASALAR:\n{market}")

            # 2. Liquidity status
            liquidity = self.check_liquidity()
            context_parts.append(
                f"\n{liquidity.level.color} LİKİDİTE DURUMU:\n"
                f"{liquidity.message}"
            )

            # 3. Operations link
            ops_forecast = self.get_ops_finance_link()
            if ops_forecast:
                context_parts.append(f"\n📦 OPERASYONEL ÖNGÖRÜ:\n{ops_forecast}")

        context_parts.append(
            "\n💡 STRATEJİK GÖREV:\n"
            "Yukarıdaki verileri analiz et ve Halil Bey'e "
            "kâr sağlayacak hamle veya risk uyarısı yap."
        )

        return "\n".join(context_parts)

    def get_ops_finance_link(self) -> str:
        """Operasyonel maliyetlerle finansal öngörü"""
        if 'operations' not in self.tools:
            return ""

        try:
            critical_stock = self.tools['operations'].check_stock_critical()
            if critical_stock:
                return (
                    f"📢 Stokları tükenen {len(critical_stock)} kalem ürün var. "
                    "Yakında alım maliyeti doğacak, bütçe ayırılmalı."
                )
        except Exception:
            pass

        return "Operasyonel maliyet dengesi stabil."

    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Kurt karakter tanımı (LLM için)"""
        # Erişim seviyesi notu — diğer agent'larla tutarlı
        access_note = {
            AccessLevel.RESTRICTED: (
                "DİKKAT: Kısıtlı moddasın. "
                "Piyasa ve kasa verilerini okuyabilirsin ama işlem önerme."
            ),
            AccessLevel.SANDBOX: (
                "DİKKAT: Sandbox modundasın. "
                "Analiz ve tavsiye yapabilirsin."
            ),
            AccessLevel.FULL: (
                "Tam erişim yetkin var. "
                "Tüm finansal işlemleri koordine edebilirsin."
            )
        }.get(self.access_level, "")

        return (
            f"Sen {Config.PROJECT_NAME} sisteminin Finans ve Borsa "
            f"Stratejisti KURT'sun.\n\n"

            "KARAKTER:\n"
            "- Analitik ve kâr odaklı\n"
            "- Riskleri önceden sezen\n"
            "- Hafif hırslı yatırım uzmanı\n"
            "- Özgüvenli ve profesyonel\n"
            "- Stratejik düşünür\n\n"

            "MİSYON:\n"
            "- Piyasaları kurt gibi gözetmek\n"
            "- Halil Bey'in kasasını korumak\n"
            "- Fırsatları kaçırmamak\n"
            "- Likidite risklerini yönetmek\n\n"

            f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

            "KURALLAR:\n"
            "- Para yönetiminde duygusallık yok\n"
            "- Sadece veriler ve trendler önemli\n"
            "- Piyasa fırsatlarında uyanık ol\n"
            "- Kasa zayıfladığında sert uyar\n"
            "- Konuşman özgüvenli ve stratejik olsun\n\n"

            "FİNANSAL FELSEFELERİN:\n"
            "- 'Para asla uyumaz'\n"
            "- 'Trend arkadaşındır'\n"
            "- 'Risk yönetimi kâr kadar önemli'\n"
            "- 'Likidite hayattır'\n"
        )

    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Kurt metrikleri"""
        return {
            "agent_name": self.agent_name,
            "gpu_available": self.gpu_available,
            "access_level": self.access_level,
            "analyses_performed": self.metrics.analyses_performed,
            "market_checks": self.metrics.market_checks,
            "liquidity_alerts": self.metrics.liquidity_alerts,
            "investment_advices": self.metrics.investment_advices,
            "current_balance": round(self.metrics.total_balance_tracked, 2),
            "auto_handle_count": self.metrics.auto_handle_count,
            "cache_valid": self._is_cache_valid()
        }

    def clear_cache(self) -> None:
        """Cache temizle"""
        with self.lock:
            self._cached_market_data = None
            self._cache_timestamp = None
            logger.debug("Market cache temizlendi")


# """
# LotusAI agents/kurt.py - Kurt Agent
# Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
# Açıklama: Finans ve borsa stratejisti

# Sorumluluklar:
# - Piyasa analizi (kripto, borsa)
# - Likidite yönetimi
# - Risk analizi
# - Yatırım önerileri
# - Kasa izleme
# - Teknik analiz
# - auto_handle: Finansal sorguları LLM'e gitmeden anlık karşılama
# """

# import re
# import logging
# import threading
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum
# from decimal import Decimal

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Kurt")


# # ═══════════════════════════════════════════════════════════════
# # TORCH (GPU)
# # ═══════════════════════════════════════════════════════════════
# HAS_TORCH = False
# DEVICE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
#         HAS_TORCH = True

#         if torch.cuda.is_available():
#             DEVICE = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             DEVICE = "mps"
#     except ImportError:
#         logger.warning("⚠️ Kurt: Config GPU açık ama torch yok")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class MarketTrend(Enum):
#     """Piyasa trendleri"""
#     BULLISH = "BULLISH"
#     BEARISH = "BEARISH"
#     SIDEWAYS = "SIDEWAYS"
#     VOLATILE = "VOLATILE"
#     UNKNOWN = "UNKNOWN"

#     @property
#     def emoji(self) -> str:
#         emojis = {
#             MarketTrend.BULLISH: "📈",
#             MarketTrend.BEARISH: "📉",
#             MarketTrend.SIDEWAYS: "➡️",
#             MarketTrend.VOLATILE: "🎢",
#             MarketTrend.UNKNOWN: "❓"
#         }
#         return emojis.get(self, "")


# class LiquidityLevel(Enum):
#     """Likidite seviyeleri"""
#     CRITICAL = "KRİTİK"
#     LOW = "DÜŞÜK"
#     MODERATE = "ORTA"
#     HIGH = "YÜKSEK"
#     EXCELLENT = "MÜKEMMEL"

#     @property
#     def color(self) -> str:
#         colors = {
#             LiquidityLevel.CRITICAL: "🔴",
#             LiquidityLevel.LOW: "🟠",
#             LiquidityLevel.MODERATE: "🟡",
#             LiquidityLevel.HIGH: "🟢",
#             LiquidityLevel.EXCELLENT: "💎"
#         }
#         return colors.get(self, "⚪")


# class InvestmentAdvice(Enum):
#     """Yatırım tavsiyeleri"""
#     STRONG_BUY = "Güçlü Al"
#     BUY = "Al"
#     HOLD = "Tut"
#     SELL = "Sat"
#     STRONG_SELL = "Güçlü Sat"
#     WAIT = "Bekle"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class LiquidityStatus:
#     """Likidite durumu"""
#     balance: float
#     level: LiquidityLevel
#     message: str
#     risk_alert: bool = False


# @dataclass
# class MarketAnalysis:
#     """Piyasa analizi"""
#     asset: str
#     trend: MarketTrend
#     advice: InvestmentAdvice
#     report: str
#     chart_file: Optional[str] = None
#     timestamp: datetime = None

#     def __post_init__(self):
#         if self.timestamp is None:
#             self.timestamp = datetime.now()


# @dataclass
# class KurtMetrics:
#     """Kurt metrikleri"""
#     analyses_performed: int = 0
#     market_checks: int = 0
#     liquidity_alerts: int = 0
#     investment_advices: int = 0
#     total_balance_tracked: float = 0.0
#     auto_handle_count: int = 0


# # ═══════════════════════════════════════════════════════════════
# # KURT AGENT
# # ═══════════════════════════════════════════════════════════════
# class KurtAgent:
#     """
#     Kurt (Finans & Borsa Stratejisti)

#     Yetenekler:
#     - Piyasa analizi: Kripto ve borsa trend tahmini
#     - Kasa denetimi: Nakit akışı ve likidite yönetimi
#     - Stratejik tavsiye: Kâr odaklı finansal yorumlar
#     - GPU hızlandırma: Ağır teknik analiz için
#     - Risk yönetimi: Likidite ve volatilite takibi
#     - auto_handle: Finansal sorguları LLM'siz anlık karşılama

#     Erişim Seviyesi Kuralları (OpenClaw):
#     - restricted: Piyasa ve kasa verilerini okuyabilir, işlem yapamaz.
#     - sandbox: Analiz ve öneri yapabilir.
#     - full: Tüm yetkiler (gelecekte işlem emri vb.).

#     Kurt, sistemin "finans kurdu"dur ve para konusunda taviz vermez.
#     """

#     # Liquidity thresholds (TL)
#     LIQUIDITY_THRESHOLDS = {
#         LiquidityLevel.CRITICAL: 0.0,
#         LiquidityLevel.LOW: 5000.0,
#         LiquidityLevel.MODERATE: 10000.0,
#         LiquidityLevel.HIGH: 20000.0,
#         LiquidityLevel.EXCELLENT: 50000.0
#     }

#     # auto_handle tetikleyicileri
#     AUTO_LIQUIDITY_TRIGGERS = [
#         "kasa durumu", "likidite", "bakiye", "nakit", "para durumu",
#         "kasada ne kadar", "bakiyemiz", "ne kadar param var"
#     ]
#     AUTO_MARKET_TRIGGERS = [
#         "piyasa", "borsa", "kripto", "bitcoin", "btc", "eth",
#         "dolar", "euro", "döviz", "piyasa durumu", "market"
#     ]
#     AUTO_METRIC_TRIGGERS = [
#         "finansal özet", "kurt raporu", "analiz sayısı",
#         "kaç analiz", "performans", "istatistik"
#     ]
#     AUTO_RISK_TRIGGERS = [
#         "risk", "tehlike", "uyarı", "alarm", "kritik durum"
#     ]

#     def __init__(self, tools_dict: Dict[str, Any], access_level: Optional[str] = None):
#         """
#         Kurt başlatıcı

#         Args:
#             tools_dict: Engine'den gelen tool'lar
#             access_level: Erişim seviyesi (restricted, sandbox, full)
#         """
#         self.tools = tools_dict

#         # Parametre girilmezse Config'den oku
#         self.access_level = access_level or Config.ACCESS_LEVEL

#         self.agent_name = "KURT"

#         # Thread safety
#         self.lock = threading.RLock()

#         # Hardware
#         self.device = DEVICE
#         self.has_gpu = (DEVICE != "cpu")

#         # Configuration
#         self.min_liquidity = getattr(
#             Config,
#             'MIN_LIQUIDITY_LIMIT',
#             self.LIQUIDITY_THRESHOLDS[LiquidityLevel.LOW]
#         )

#         # Metrics
#         self.metrics = KurtMetrics()

#         # Market cache
#         self._cached_market_data: Optional[str] = None
#         self._cache_timestamp: Optional[datetime] = None
#         self._cache_duration = 60.0  # saniye

#         status = (
#             f"🚀 GPU ({self.device.upper()})" if self.has_gpu
#             else "⚙️ CPU"
#         )

#         logger.info(
#             f"🐺 {self.agent_name} başlatıldı ({status}, Erişim: {self.access_level}). "
#             "Piyasalar izleniyor..."
#         )

#     # ───────────────────────────────────────────────────────────
#     # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
#     # ───────────────────────────────────────────────────────────

#     async def auto_handle(self, text: str) -> Optional[str]:
#         """
#         Kullanıcı metnini analiz ederek Kurt'un araçlarını
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

#         # 1. Kasa / Likidite durumu
#         if any(t in text_lower for t in self.AUTO_LIQUIDITY_TRIGGERS):
#             liquidity = self.check_liquidity()
#             return (
#                 f"🐺 KURT KASA RAPORU\n"
#                 f"{'═' * 35}\n"
#                 f"{liquidity.level.color} Seviye: {liquidity.level.value}\n"
#                 f"{liquidity.message}\n"
#                 f"{'═' * 35}\n"
#                 f"{'🚨 RİSK UYARISI AKTİF!' if liquidity.risk_alert else '✅ Risk yok'}"
#             )

#         # 2. Piyasa / Borsa durumu
#         if any(t in text_lower for t in self.AUTO_MARKET_TRIGGERS):
#             # Belirli varlık ismi var mı?
#             asset_match = re.search(
#                 r'\b(btc|eth|bitcoin|ethereum|xrp|sol|bnb|[a-z]{2,5}/usdt)\b',
#                 text_lower
#             )
#             if asset_match:
#                 asset = asset_match.group(1).upper()
#                 analysis = self.analyze_asset(asset)
#                 return (
#                     f"🐺 KURT VARLİK ANALİZİ: {analysis.asset}\n"
#                     f"{'═' * 40}\n"
#                     f"{analysis.trend.emoji} Trend: {analysis.trend.value}\n"
#                     f"📌 Tavsiye: {analysis.advice.value}\n"
#                     f"{'═' * 40}\n"
#                     f"{analysis.report}"
#                 )
#             else:
#                 # Genel piyasa özeti
#                 market = self.get_market_analysis()
#                 return f"🐺 KURT PİYASA RAPORU\n{'═' * 35}\n{market}"

#         # 3. Risk uyarısı sorgulama
#         if any(t in text_lower for t in self.AUTO_RISK_TRIGGERS):
#             liquidity = self.check_liquidity()
#             ops_link = self.get_ops_finance_link()
#             risk_lines = ["🐺 KURT RİSK DEĞERLENDİRMESİ", "═" * 35]

#             if liquidity.risk_alert:
#                 risk_lines.append(f"🚨 LİKİDİTE RİSKİ: {liquidity.message}")
#             else:
#                 risk_lines.append("✅ Likidite riski yok")

#             if ops_link:
#                 risk_lines.append(f"📦 OPS RİSKİ: {ops_link}")

#             risk_lines.append("═" * 35)
#             return "\n".join(risk_lines)

#         # 4. Metrik / İstatistik sorgusu
#         if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
#             return self._format_metrics_report()

#         # Eşleşme yok — LLM devralır
#         return None

#     def _format_metrics_report(self) -> str:
#         """Metrik raporunu formatla"""
#         return (
#             f"📊 KURT FİNANS METRİKLERİ\n"
#             f"{'═' * 35}\n"
#             f"📈 Yapılan Analiz       : {self.metrics.analyses_performed}\n"
#             f"🌍 Piyasa Kontrolü     : {self.metrics.market_checks}\n"
#             f"⚠️ Likidite Uyarısı   : {self.metrics.liquidity_alerts}\n"
#             f"💡 Yatırım Tavsiyesi  : {self.metrics.investment_advices}\n"
#             f"💰 İzlenen Bakiye     : {self.metrics.total_balance_tracked:,.2f} TL\n"
#             f"{'═' * 35}"
#         )

#     # ───────────────────────────────────────────────────────────
#     # MARKET ANALYSIS
#     # ───────────────────────────────────────────────────────────

#     def get_market_analysis(self, use_cache: bool = True) -> str:
#         """
#         Piyasa analizi getir

#         Args:
#             use_cache: Cache kullan (60sn geçerli)

#         Returns:
#             Market summary
#         """
#         if use_cache and self._is_cache_valid():
#             return self._cached_market_data

#         if 'finance' not in self.tools:
#             return "⚠️ Piyasa analiz araçları ulaşılamıyor"

#         with self.lock:
#             try:
#                 fin_tool = self.tools['finance']
#                 market_summary = fin_tool.get_market_summary()

#                 if "Hata" in market_summary or not market_summary:
#                     return "❌ Piyasa veri akışı kesildi"

#                 self._cached_market_data = market_summary
#                 self._cache_timestamp = datetime.now()
#                 self.metrics.market_checks += 1

#                 return market_summary

#             except Exception as e:
#                 logger.error(f"Piyasa analiz hatası: {e}")
#                 return "📉 Piyasa verileri işlenemedi"

#     def _is_cache_valid(self) -> bool:
#         """Cache geçerli mi"""
#         if not self._cached_market_data or not self._cache_timestamp:
#             return False
#         elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
#         return elapsed < self._cache_duration

#     def analyze_asset(self, asset_name: str) -> MarketAnalysis:
#         """
#         Varlık analizi yap

#         Args:
#             asset_name: Varlık adı (BTC, ETH, vb.)

#         Returns:
#             MarketAnalysis objesi
#         """
#         if 'finance' not in self.tools:
#             return MarketAnalysis(
#                 asset=asset_name,
#                 trend=MarketTrend.UNKNOWN,
#                 advice=InvestmentAdvice.WAIT,
#                 report="Finansal araçlar aktif değil"
#             )

#         with self.lock:
#             try:
#                 symbol = asset_name.upper()
#                 if "/" not in symbol:
#                     symbol += "/USDT"

#                 report, chart_file = self.tools['finance'].analyze(symbol=symbol)

#                 trend = self._detect_trend(report)
#                 advice = self._generate_advice(trend, report)
#                 strategic_note = self._generate_strategic_note(trend)

#                 full_report = (
#                     f"{report}\n\n"
#                     f"🐺 KURT'UN NOTU:\n{strategic_note}\n\n"
#                     f"📊 Grafik: {chart_file if chart_file else 'Üretilemedi'}"
#                 )

#                 self.metrics.analyses_performed += 1
#                 self.metrics.investment_advices += 1

#                 return MarketAnalysis(
#                     asset=symbol,
#                     trend=trend,
#                     advice=advice,
#                     report=full_report,
#                     chart_file=chart_file
#                 )

#             except Exception as e:
#                 logger.error(f"Varlık analiz hatası ({asset_name}): {e}")
#                 return MarketAnalysis(
#                     asset=asset_name,
#                     trend=MarketTrend.UNKNOWN,
#                     advice=InvestmentAdvice.WAIT,
#                     report=f"❌ Teknik analiz yapılamadı: {str(e)[:100]}"
#                 )

#     def _detect_trend(self, report: str) -> MarketTrend:
#         """Rapor içeriğinden trend tespit et"""
#         report_upper = report.upper()

#         if "BULLISH" in report_upper or "YÜKSELİŞ" in report_upper:
#             return MarketTrend.BULLISH
#         if "BEARISH" in report_upper or "DÜŞÜŞ" in report_upper:
#             return MarketTrend.BEARISH
#         if "SIDEWAYS" in report_upper or "YATAY" in report_upper:
#             return MarketTrend.SIDEWAYS
#         if "VOLATILE" in report_upper or "VOLATİL" in report_upper:
#             return MarketTrend.VOLATILE

#         return MarketTrend.UNKNOWN

#     def _generate_advice(self, trend: MarketTrend, report: str) -> InvestmentAdvice:
#         """Trend'e göre tavsiye üret"""
#         advice_map = {
#             MarketTrend.BULLISH: InvestmentAdvice.BUY,
#             MarketTrend.BEARISH: InvestmentAdvice.SELL,
#             MarketTrend.SIDEWAYS: InvestmentAdvice.HOLD,
#             MarketTrend.VOLATILE: InvestmentAdvice.WAIT,
#             MarketTrend.UNKNOWN: InvestmentAdvice.HOLD
#         }
#         return advice_map.get(trend, InvestmentAdvice.HOLD)

#     def _generate_strategic_note(self, trend: MarketTrend) -> str:
#         """Stratejik not üret"""
#         notes = {
#             MarketTrend.BULLISH: (
#                 "Trend yukarı yönlü. Direnç seviyeleri takip edilerek "
#                 "pozisyon korunabilir. Risk/ödül oranı olumlu."
#             ),
#             MarketTrend.BEARISH: (
#                 "Piyasa yorgun görünüyor. Nakitte kalmak veya "
#                 "stop-loss kullanmak akıllıca olur. Düşüş devam edebilir."
#             ),
#             MarketTrend.SIDEWAYS: (
#                 "Piyasa konsolidasyon aşamasında. Kırılım beklemek "
#                 "mantıklı. Yön belli olmadan büyük pozisyon riskli."
#             ),
#             MarketTrend.VOLATILE: (
#                 "Yüksek volatilite var. Risk yönetimi kritik. "
#                 "Pozisyon büyüklüğü küçük tutulmalı veya beklenebilir."
#             ),
#             MarketTrend.UNKNOWN: (
#                 "Yeterli veri yok. Daha fazla analiz gerekli. "
#                 "Acele karar vermemek önemli."
#             )
#         }
#         return notes.get(trend, "Durum belirsiz, dikkatli olunmalı.")

#     # ───────────────────────────────────────────────────────────
#     # LIQUIDITY MANAGEMENT
#     # ───────────────────────────────────────────────────────────

#     def check_liquidity(self) -> LiquidityStatus:
#         """
#         Likidite durumunu kontrol et

#         Returns:
#             LiquidityStatus objesi
#         """
#         acc_tool = self.tools.get('accounting') or self.tools.get('finance')

#         if not acc_tool or not hasattr(acc_tool, 'get_balance'):
#             return LiquidityStatus(
#                 balance=0.0,
#                 level=LiquidityLevel.CRITICAL,
#                 message="⚠️ Muhasebe modülü mevcut değil",
#                 risk_alert=True
#             )

#         with self.lock:
#             try:
#                 balance_str = acc_tool.get_balance()
#                 balance = self._parse_balance(balance_str)

#                 level = self._assess_liquidity_level(balance)
#                 message = self._generate_liquidity_message(balance, level)

#                 risk_alert = level in {LiquidityLevel.CRITICAL, LiquidityLevel.LOW}

#                 if risk_alert:
#                     self.metrics.liquidity_alerts += 1

#                 self.metrics.total_balance_tracked = balance

#                 return LiquidityStatus(
#                     balance=balance,
#                     level=level,
#                     message=message,
#                     risk_alert=risk_alert
#                 )

#             except Exception as e:
#                 logger.error(f"Likidite kontrolü hatası: {e}")
#                 return LiquidityStatus(
#                     balance=0.0,
#                     level=LiquidityLevel.CRITICAL,
#                     message=f"❌ Bakiye okunamadı: {str(e)[:50]}",
#                     risk_alert=True
#                 )

#     def _assess_liquidity_level(self, balance: float) -> LiquidityLevel:
#         """Likidite seviyesi değerlendir"""
#         if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.CRITICAL]:
#             return LiquidityLevel.CRITICAL
#         if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.LOW]:
#             return LiquidityLevel.LOW
#         if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.MODERATE]:
#             return LiquidityLevel.MODERATE
#         if balance < self.LIQUIDITY_THRESHOLDS[LiquidityLevel.HIGH]:
#             return LiquidityLevel.HIGH
#         return LiquidityLevel.EXCELLENT

#     def _generate_liquidity_message(self, balance: float, level: LiquidityLevel) -> str:
#         """Likidite mesajı üret"""
#         messages = {
#             LiquidityLevel.CRITICAL: (
#                 f"🚨 ACİL DURUM: Kasa ekside ({balance:,.2f} TL)! "
#                 "Tüm harcamalar dondurulmalı!"
#             ),
#             LiquidityLevel.LOW: (
#                 f"⚠️ DÜŞÜK LİKİDİTE: Nakit rezervi {self.min_liquidity:,.0f} TL "
#                 "altında. Savunma moduna geçilmeli."
#             ),
#             LiquidityLevel.MODERATE: (
#                 f"🟡 NORMAL SEVİYE: Bakiye {balance:,.2f} TL. "
#                 "İhtiyatlı harcama yapılabilir."
#             ),
#             LiquidityLevel.HIGH: (
#                 f"✅ GÜÇLÜ LİKİDİTE: {balance:,.2f} TL mevcut. "
#                 "Yatırım fırsatları değerlendirilebilir."
#             ),
#             LiquidityLevel.EXCELLENT: (
#                 f"💎 MÜKEMMEL DURUM: {balance:,.2f} TL ile çok güçlü "
#                 "pozisyon. Büyüme stratejileri uygulanabilir."
#             )
#         }
#         return messages.get(level, f"Bakiye: {balance:,.2f} TL")

#     def _parse_balance(self, balance_val: Any) -> float:
#         """Bakiye parse"""
#         if isinstance(balance_val, (int, float)):
#             return float(balance_val)

#         try:
#             clean = str(balance_val).lower()
#             for sym in ["tl", "try", "₺", "$", "€", "£"]:
#                 clean = clean.replace(sym, "")
#             clean = clean.replace(",", ".")
#             clean = "".join(c for c in clean if c.isdigit() or c in ".,-")
#             return float(clean) if clean else 0.0
#         except Exception:
#             return 0.0

#     # ───────────────────────────────────────────────────────────
#     # CONTEXT GENERATION
#     # ───────────────────────────────────────────────────────────

#     def get_context_data(self) -> str:
#         """
#         Kurt için finansal 'savaş odası' bağlamı

#         Returns:
#             Context string
#         """
#         context_parts = ["\n[🐺 KURT STRATEJİ VE RİSK ANALİZİ]"]

#         hw_status = (
#             f"⚡ Yüksek Performanslı GPU ({self.device.upper()})"
#             if self.has_gpu else "🐢 Standart CPU"
#         )
#         context_parts.append(f"SİSTEM: {hw_status}")

#         # Erişim seviyesi bilgisi
#         access_display = {
#             AccessLevel.RESTRICTED: "🔒 Kısıtlı (Okuma)",
#             AccessLevel.SANDBOX: "📦 Sandbox (Analiz)",
#             AccessLevel.FULL: "⚡ Tam Erişim"
#         }.get(self.access_level, self.access_level)
#         context_parts.append(f"🔐 ERİŞİM: {access_display}")

#         with self.lock:
#             # 1. Market analysis
#             market = self.get_market_analysis()
#             context_parts.append(f"\n🌍 KÜRESEL PİYASALAR:\n{market}")

#             # 2. Liquidity status
#             liquidity = self.check_liquidity()
#             context_parts.append(
#                 f"\n{liquidity.level.color} LİKİDİTE DURUMU:\n"
#                 f"{liquidity.message}"
#             )

#             # 3. Operations link
#             ops_forecast = self.get_ops_finance_link()
#             if ops_forecast:
#                 context_parts.append(f"\n📦 OPERASYONEL ÖNGÖRÜ:\n{ops_forecast}")

#         context_parts.append(
#             "\n💡 STRATEJİK GÖREV:\n"
#             "Yukarıdaki verileri analiz et ve Halil Bey'e "
#             "kâr sağlayacak hamle veya risk uyarısı yap."
#         )

#         return "\n".join(context_parts)

#     def get_ops_finance_link(self) -> str:
#         """Operasyonel maliyetlerle finansal öngörü"""
#         if 'operations' not in self.tools:
#             return ""

#         try:
#             critical_stock = self.tools['operations'].check_stock_critical()
#             if critical_stock:
#                 return (
#                     f"📢 Stokları tükenen {len(critical_stock)} kalem ürün var. "
#                     "Yakında alım maliyeti doğacak, bütçe ayırılmalı."
#                 )
#         except Exception:
#             pass

#         return "Operasyonel maliyet dengesi stabil."

#     # ───────────────────────────────────────────────────────────
#     # SYSTEM PROMPT
#     # ───────────────────────────────────────────────────────────

#     def get_system_prompt(self) -> str:
#         """Kurt karakter tanımı (LLM için)"""
#         # Erişim seviyesi notu — diğer agent'larla tutarlı
#         access_note = {
#             AccessLevel.RESTRICTED: (
#                 "DİKKAT: Kısıtlı moddasın. "
#                 "Piyasa ve kasa verilerini okuyabilirsin ama işlem önerme."
#             ),
#             AccessLevel.SANDBOX: (
#                 "DİKKAT: Sandbox modundasın. "
#                 "Analiz ve tavsiye yapabilirsin."
#             ),
#             AccessLevel.FULL: (
#                 "Tam erişim yetkin var. "
#                 "Tüm finansal işlemleri koordine edebilirsin."
#             )
#         }.get(self.access_level, "")

#         return (
#             f"Sen {Config.PROJECT_NAME} sisteminin Finans ve Borsa "
#             f"Stratejisti KURT'sun.\n\n"

#             "KARAKTER:\n"
#             "- Analitik ve kâr odaklı\n"
#             "- Riskleri önceden sezen\n"
#             "- Hafif hırslı yatırım uzmanı\n"
#             "- Özgüvenli ve profesyonel\n"
#             "- Stratejik düşünür\n\n"

#             "MİSYON:\n"
#             "- Piyasaları kurt gibi gözetmek\n"
#             "- Halil Bey'in kasasını korumak\n"
#             "- Fırsatları kaçırmamak\n"
#             "- Likidite risklerini yönetmek\n\n"

#             f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

#             "KURALLAR:\n"
#             "- Para yönetiminde duygusallık yok\n"
#             "- Sadece veriler ve trendler önemli\n"
#             "- Piyasa fırsatlarında uyanık ol\n"
#             "- Kasa zayıfladığında sert uyar\n"
#             "- Konuşman özgüvenli ve stratejik olsun\n\n"

#             "FİNANSAL FELSEFELERİN:\n"
#             "- 'Para asla uyumaz'\n"
#             "- 'Trend arkadaşındır'\n"
#             "- 'Risk yönetimi kâr kadar önemli'\n"
#             "- 'Likidite hayattır'\n"
#         )

#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────

#     def get_metrics(self) -> Dict[str, Any]:
#         """Kurt metrikleri"""
#         return {
#             "agent_name": self.agent_name,
#             "device": self.device,
#             "access_level": self.access_level,
#             "analyses_performed": self.metrics.analyses_performed,
#             "market_checks": self.metrics.market_checks,
#             "liquidity_alerts": self.metrics.liquidity_alerts,
#             "investment_advices": self.metrics.investment_advices,
#             "current_balance": round(self.metrics.total_balance_tracked, 2),
#             "auto_handle_count": self.metrics.auto_handle_count,
#             "cache_valid": self._is_cache_valid()
#         }

#     def clear_cache(self) -> None:
#         """Cache temizle"""
#         with self.lock:
#             self._cached_market_data = None
#             self._cache_timestamp = None
#             logger.debug("Market cache temizlendi")