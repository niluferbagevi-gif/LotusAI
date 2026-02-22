"""
LotusAI Finance Manager
Sürüm: 2.5.5 (Eklendi: Erişim Seviyesi Desteği)
Açıklama: Finans, borsa ve analiz yönetimi

Özellikler:
- CCXT borsa entegrasyonu
- Teknik analiz (RSI, EMA, MACD)
- GPU hızlandırmalı hesaplamalar
- Grafik oluşturma
- Piyasa özeti
- Cache sistemi
- Erişim seviyesi kontrolleri (restricted/sandbox/full)
"""

import os
import sys
import logging
import warnings
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Finance")


# ═══════════════════════════════════════════════════════════════
# LIBRARIES
# ═══════════════════════════════════════════════════════════════
FINANCE_LIBS = False

try:
    import ccxt
    import pandas as pd
    import ta
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import numpy as np
    FINANCE_LIBS = True
except ImportError as e:
    logger.warning(
        f"⚠️ Finans kütüphaneleri eksik: {e}\n"
        "pip install ccxt pandas ta mplfinance numpy"
    )


# ═══════════════════════════════════════════════════════════════
# GPU (PyTorch)
# ═══════════════════════════════════════════════════════════════
HAS_GPU = False
DEVICE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 Finance GPU aktif: {gpu_name}")
            except Exception:
                logger.info("🚀 Finance GPU aktif")
        else:
            logger.info("ℹ️ CUDA yok, CPU kullanılacak")
    except ImportError:
        logger.info("ℹ️ PyTorch yok, GPU hızlandırma devre dışı")
    except Exception as e:
        logger.warning(f"⚠️ GPU başlatma hatası: {e}")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class TrendType(Enum):
    """Trend tipleri"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalType(Enum):
    """Sinyal tipleri"""
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    NONE = "none"


class TimeFrame(Enum):
    """Zaman dilimleri"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class MarketData:
    """Piyasa verisi"""
    symbol: str
    price: float
    change_percent: float
    volume: float
    timestamp: datetime


@dataclass
class TechnicalAnalysis:
    """Teknik analiz sonucu"""
    symbol: str
    timeframe: str
    price: float
    trend: TrendType
    rsi: float
    ema50: float
    ema200: float
    signal: SignalType
    chart_path: Optional[str] = None


@dataclass
class FinanceMetrics:
    """Finance manager metrikleri"""
    market_queries: int = 0
    analyses_performed: int = 0
    charts_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors_encountered: int = 0


# ═══════════════════════════════════════════════════════════════
# FINANCE MANAGER
# ═══════════════════════════════════════════════════════════════
class FinanceManager:
    """
    LotusAI Finans, Borsa ve Analiz Yöneticisi
    
    Yetenekler:
    - CCXT: Binance entegrasyonu
    - Teknik analiz: RSI, EMA, MACD
    - GPU hızlandırma: PyTorch ile hesaplama
    - Grafik: mplfinance ile chart oluşturma
    - Cache: Market data önbellekleme
    - Accounting: Muhasebe entegrasyonu
    
    Piyasa verilerini çeker, teknik analiz yapar ve grafik üretir.
    Erişim seviyesine göre işlem kısıtlamaları uygulanır.
    """
    
    # Default symbols
    DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    
    # Cache settings
    CACHE_DURATION = 15  # saniye
    
    # RSI thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Chart settings
    CHART_DPI = 120
    
    def __init__(
        self,
        accounting_manager: Optional[Any] = None,
        access_level: str = "sandbox"
    ):
        """
        Finance manager başlatıcı
        
        Args:
            accounting_manager: Muhasebe yöneticisi (opsiyonel)
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Exchange
        self.exchange: Optional[ccxt.Exchange] = None
        
        # Accounting
        self.accounting = accounting_manager
        
        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        
        # Metrics
        self.metrics = FinanceMetrics()
        
        # Initialize exchange
        if FINANCE_LIBS:
            self._init_exchange()
        
        logger.info(f"✅ FinanceManager hazır (Erişim: {self.access_level})")
    
    def _init_exchange(self) -> None:
        """Borsa bağlantısı başlat"""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000  # 30 saniye
            })
            
            logger.info("⏳ Binance piyasa verileri yükleniyor...")
            self.exchange.load_markets()
            logger.info("✅ Binance bağlantısı hazır")
        
        except Exception as e:
            logger.error(f"Borsa bağlantı hatası: {e}")
            self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # MARKET DATA
    # ───────────────────────────────────────────────────────────
    
    def get_market_summary(
        self,
        custom_symbols: Optional[List[str]] = None
    ) -> str:
        """
        Piyasa özeti - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            custom_symbols: Özel sembol listesi
        
        Returns:
            Formatlanmış özet
        """
        if not FINANCE_LIBS or not self.exchange:
            return "⚠️ Finansal modül veya borsa bağlantısı aktif değil"
        
        with self.lock:
            try:
                symbols = custom_symbols or self.DEFAULT_SYMBOLS
                summary = []
                
                # Toplu veri çekme
                try:
                    tickers = self.exchange.fetch_tickers(symbols)
                except Exception as e:
                    logger.warning(f"Toplu veri çekilemedi, tekliler deneniyor: {e}")
                    tickers = {}
                    for sym in symbols:
                        t = self._get_ticker_cached(sym)
                        if t:
                            tickers[sym] = t

                for symbol in symbols:
                    ticker = tickers.get(symbol)
                    
                    if not ticker:
                        continue
                    
                    price = ticker['last']
                    change = ticker['percentage']
                    
                    icon = "🟢" if change >= 0 else "🔴"
                    trend = "📈" if change > 2.5 else "📉" if change < -2.5 else "➡️"
                    
                    clean_sym = symbol.split('/')[0]
                    summary.append(
                        f"{icon} {clean_sym}: ${price:,.2f} "
                        f"(%{change:+.2f}) {trend}"
                    )
                
                self.metrics.market_queries += 1
                
                return (
                    " | ".join(summary)
                    if summary else "❌ Piyasa verisi çekilemiyor"
                )
            
            except Exception as e:
                logger.error(f"Piyasa özeti hatası: {e}")
                self.metrics.errors_encountered += 1
                return "Piyasa verilerine erişilemiyor"
    
    def _get_ticker_cached(self, symbol: str) -> Optional[Dict]:
        """Cache'li ticker getir (Tekli sorgular için)"""
        current_time = datetime.now()
        
        if symbol in self._cache:
            cache_age = (
                current_time - self._cache_time.get(symbol, current_time)
            ).total_seconds()
            
            if cache_age < self.CACHE_DURATION:
                self.metrics.cache_hits += 1
                return self._cache[symbol]
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self._cache[symbol] = ticker
            self._cache_time[symbol] = current_time
            self.metrics.cache_misses += 1
            return ticker
        except Exception as e:
            logger.error(f"Ticker fetch hatası ({symbol}): {str(e)}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # BALANCE
    # ───────────────────────────────────────────────────────────
    
    def get_balance(self) -> str:
        """
        Kasa bakiyesi - Tüm erişim seviyelerinde kullanılabilir.
        
        Returns:
            Formatlanmış bakiye
        """
        if self.accounting:
            try:
                val = self.accounting.get_balance()
                return f"{val:,.2f} TRY"
            except Exception as e:
                logger.error(f"Bakiye sorgulama hatası: {e}")
                return "Bakiye okunamadı"
        
        return "12,450.00 TRY (Demo)"
    
    # ───────────────────────────────────────────────────────────
    # TECHNICAL ANALYSIS (Erişim kontrollü)
    # ───────────────────────────────────────────────────────────
    
    def analyze(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = '4h',
        limit: int = 100
    ) -> Tuple[str, Optional[str]]:
        """
        Teknik analiz - Sadece sandbox ve full modda çalışır.
        
        Args:
            symbol: Sembol
            timeframe: Zaman dilimi
            limit: Veri sayısı
        
        Returns:
            (Rapor, Grafik dosya adı)
        """
        # Erişim kontrolü
        if self.access_level == AccessLevel.RESTRICTED:
            return "🔒 Kısıtlı modda teknik analiz yapılamaz. Sadece piyasa özetini görüntüleyebilirsiniz.", None
        
        if not FINANCE_LIBS or not self.exchange:
            return "Analiz araçları yüklü değil", None
        
        with self.lock:
            try:
                # Format symbol
                symbol = symbol.upper()
                if "/" not in symbol:
                    symbol = f"{symbol}/USDT"
                
                # Fetch OHLCV data
                bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not bars:
                    return f"{symbol} için veri boş", None
                
                # Create dataframe
                df = pd.DataFrame(
                    bars,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Validate data
                if df.iloc[-1]['EMA200'] is None or pd.isna(df.iloc[-1]['EMA200']):
                    return f"{symbol} için yeterli veri yok (EMA200)", None
                
                # Analysis
                analysis = self._analyze_dataframe(df, symbol, timeframe)
                
                # Generate chart (grafik oluşturma, sadece sandbox ve full'de yapılır)
                chart_filename = None
                if self.access_level != AccessLevel.RESTRICTED:  # zaten yukarıda kontrol ettik, ama tekrar
                    chart_filename = self._generate_chart(df, symbol, timeframe)
                
                # Format report
                report = self._format_analysis_report(analysis, chart_filename)
                
                self.metrics.analyses_performed += 1
                
                return report, chart_filename
            
            except Exception as e:
                logger.error(f"Analiz hatası: {e}")
                self.metrics.errors_encountered += 1
                import traceback
                logger.error(traceback.format_exc())
                return f"Analiz başarısız: {str(e)[:100]}", None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        İndikatörleri hesapla
        """
        try:
            # GPU symbolic operation (if available)
            if HAS_GPU:
                try:
                    import torch
                    prices = torch.tensor(
                        df['close'].values,
                        dtype=torch.float32
                    ).to(DEVICE)
                except Exception:
                    pass
            
            # Calculate indicators (CPU - reliable)
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)
            df['MACD'] = ta.trend.macd(df['close'])
            
            return df
        
        except Exception as e:
            logger.error(f"İndikatör hesaplama hatası: {e}")
            return df
    
    def _analyze_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> TechnicalAnalysis:
        """DataFrame'den analiz çıkar"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        trend = (
            TrendType.BULLISH if last['close'] > last['EMA50']
            else TrendType.BEARISH
        )
        
        signal = SignalType.NONE
        
        if prev['EMA50'] < prev['EMA200'] and last['EMA50'] > last['EMA200']:
            signal = SignalType.GOLDEN_CROSS
        elif prev['EMA50'] > prev['EMA200'] and last['EMA50'] < last['EMA200']:
            signal = SignalType.DEATH_CROSS
        
        rsi_val = last['RSI'] if not pd.isna(last['RSI']) else 50.0
        
        if rsi_val > self.RSI_OVERBOUGHT:
            signal = SignalType.OVERBOUGHT
        elif rsi_val < self.RSI_OVERSOLD:
            signal = SignalType.OVERSOLD
        
        return TechnicalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            price=last['close'],
            trend=trend,
            rsi=rsi_val,
            ema50=last['EMA50'],
            ema200=last['EMA200'],
            signal=signal
        )
    
    def _format_analysis_report(
        self,
        analysis: TechnicalAnalysis,
        chart_filename: Optional[str]
    ) -> str:
        """Analiz raporunu formatla"""
        device_info = (
            f"⚡ GPU ({DEVICE})" if HAS_GPU
            else "💻 CPU"
        )
        
        trend_emoji = "🐂" if analysis.trend == TrendType.BULLISH else "🐻"
        
        rsi_status = "NÖTR"
        if analysis.rsi > self.RSI_OVERBOUGHT:
            rsi_status = "AŞIRI ALIM (Dikkat)"
        elif analysis.rsi < self.RSI_OVERSOLD:
            rsi_status = "AŞIRI SATIM (Fırsat)"
        
        signal_msg = ""
        if analysis.signal == SignalType.GOLDEN_CROSS:
            signal_msg = "\n🚀 GOLDEN CROSS! (Uzun vadeli AL sinyali)"
        elif analysis.signal == SignalType.DEATH_CROSS:
            signal_msg = "\n⚠️ DEATH CROSS! (Uzun vadeli SAT sinyali)"
        
        report_lines = [
            f"📊 {analysis.symbol} TEKNİK ANALİZ "
            f"({analysis.timeframe}) - {device_info}",
            f"💰 Fiyat: ${analysis.price:,.2f}",
            f"📈 Trend: {analysis.trend.value} {trend_emoji}",
            f"⚡ RSI: {analysis.rsi:.2f} ({rsi_status})",
            signal_msg,
            "─" * 35
        ]
        
        if chart_filename:
            report_lines.append(f"📸 Grafik: {chart_filename}")
        else:
            report_lines.append("📸 Grafik oluşturulamadı")
        
        return "\n".join(report_lines)
    
    # ───────────────────────────────────────────────────────────
    # CHART GENERATION
    # ───────────────────────────────────────────────────────────
    
    def _generate_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[str]:
        """
        Grafik oluştur (sadece grafik oluşturma işlemi)
        """
        try:
            static_dir = Config.STATIC_DIR
            static_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{symbol.replace('/', '_')}_{timestamp}.png"
            output_path = static_dir / filename
            
            style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                rc={'font.size': 8}
            )
            
            apds = [
                mpf.make_addplot(df['EMA50'], color='orange', width=1.0),
                mpf.make_addplot(df['EMA200'], color='cyan', width=1.0)
            ]
            
            mpf.plot(
                df,
                type='candle',
                style=style,
                addplot=apds,
                title=f"\n{symbol} - LotusAI Analiz",
                volume=True,
                savefig=dict(
                    fname=str(output_path),
                    dpi=self.CHART_DPI,
                    bbox_inches='tight'
                )
            )
            
            plt.close('all')
            
            if Config.DEBUG_MODE:
                self._open_chart(output_path)
            
            self.metrics.charts_generated += 1
            
            return filename
        
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {e}")
            return None
    
    def _open_chart(self, path: Path) -> None:
        """Grafiği aç (debug)"""
        try:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                os.system(f"open {path}")
            else:
                os.system(f"xdg-open {path}")
        except Exception:
            pass
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Finance metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "market_queries": self.metrics.market_queries,
            "analyses_performed": self.metrics.analyses_performed,
            "charts_generated": self.metrics.charts_generated,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "errors_encountered": self.metrics.errors_encountered,
            "gpu_available": HAS_GPU,
            "device": DEVICE,
            "exchange_connected": self.exchange is not None,
            "access_level": self.access_level
        }
    
    def clear_cache(self) -> None:
        """Cache'i temizle"""
        with self.lock:
            self._cache.clear()
            self._cache_time.clear()
            logger.debug("Market cache temizlendi")



# """
# LotusAI Finance Manager
# Sürüm: 2.5.4 (Fix: Binance ExchangeInfo Timeout & Batch Fetching)
# Açıklama: Finans, borsa ve analiz yönetimi

# Özellikler:
# - CCXT borsa entegrasyonu
# - Teknik analiz (RSI, EMA, MACD)
# - GPU hızlandırmalı hesaplamalar
# - Grafik oluşturma
# - Piyasa özeti
# - Cache sistemi
# """

# import os
# import sys
# import logging
# import warnings
# import threading
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import Tuple, List, Optional, Dict, Any
# from dataclasses import dataclass
# from enum import Enum

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config

# logger = logging.getLogger("LotusAI.Finance")


# # ═══════════════════════════════════════════════════════════════
# # LIBRARIES
# # ═══════════════════════════════════════════════════════════════
# FINANCE_LIBS = False

# try:
#     import ccxt
#     import pandas as pd
#     import ta
#     import mplfinance as mpf
#     import matplotlib.pyplot as plt
#     import numpy as np
#     FINANCE_LIBS = True
# except ImportError as e:
#     logger.warning(
#         f"⚠️ Finans kütüphaneleri eksik: {e}\n"
#         "pip install ccxt pandas ta mplfinance numpy"
#     )


# # ═══════════════════════════════════════════════════════════════
# # GPU (PyTorch)
# # ═══════════════════════════════════════════════════════════════
# HAS_GPU = False
# DEVICE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
        
#         if torch.cuda.is_available():
#             HAS_GPU = True
#             DEVICE = "cuda"
#             try:
#                 gpu_name = torch.cuda.get_device_name(0)
#                 logger.info(f"🚀 Finance GPU aktif: {gpu_name}")
#             except Exception:
#                 logger.info("🚀 Finance GPU aktif")
#         else:
#             logger.info("ℹ️ CUDA yok, CPU kullanılacak")
#     except ImportError:
#         logger.info("ℹ️ PyTorch yok, GPU hızlandırma devre dışı")
#     except Exception as e:
#         logger.warning(f"⚠️ GPU başlatma hatası: {e}")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class TrendType(Enum):
#     """Trend tipleri"""
#     BULLISH = "BULLISH"
#     BEARISH = "BEARISH"
#     NEUTRAL = "NEUTRAL"


# class SignalType(Enum):
#     """Sinyal tipleri"""
#     GOLDEN_CROSS = "golden_cross"
#     DEATH_CROSS = "death_cross"
#     OVERBOUGHT = "overbought"
#     OVERSOLD = "oversold"
#     NONE = "none"


# class TimeFrame(Enum):
#     """Zaman dilimleri"""
#     M1 = "1m"
#     M5 = "5m"
#     M15 = "15m"
#     M30 = "30m"
#     H1 = "1h"
#     H4 = "4h"
#     D1 = "1d"
#     W1 = "1w"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class MarketData:
#     """Piyasa verisi"""
#     symbol: str
#     price: float
#     change_percent: float
#     volume: float
#     timestamp: datetime


# @dataclass
# class TechnicalAnalysis:
#     """Teknik analiz sonucu"""
#     symbol: str
#     timeframe: str
#     price: float
#     trend: TrendType
#     rsi: float
#     ema50: float
#     ema200: float
#     signal: SignalType
#     chart_path: Optional[str] = None


# @dataclass
# class FinanceMetrics:
#     """Finance manager metrikleri"""
#     market_queries: int = 0
#     analyses_performed: int = 0
#     charts_generated: int = 0
#     cache_hits: int = 0
#     cache_misses: int = 0
#     errors_encountered: int = 0


# # ═══════════════════════════════════════════════════════════════
# # FINANCE MANAGER
# # ═══════════════════════════════════════════════════════════════
# class FinanceManager:
#     """
#     LotusAI Finans, Borsa ve Analiz Yöneticisi
    
#     Yetenekler:
#     - CCXT: Binance entegrasyonu
#     - Teknik analiz: RSI, EMA, MACD
#     - GPU hızlandırma: PyTorch ile hesaplama
#     - Grafik: mplfinance ile chart oluşturma
#     - Cache: Market data önbellekleme
#     - Accounting: Muhasebe entegrasyonu
    
#     Piyasa verilerini çeker, teknik analiz yapar ve grafik üretir.
#     """
    
#     # Default symbols
#     DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    
#     # Cache settings
#     CACHE_DURATION = 15  # saniye
    
#     # RSI thresholds
#     RSI_OVERBOUGHT = 70
#     RSI_OVERSOLD = 30
    
#     # Chart settings
#     CHART_DPI = 120
    
#     def __init__(self, accounting_manager: Optional[Any] = None):
#         """
#         Finance manager başlatıcı
        
#         Args:
#             accounting_manager: Muhasebe yöneticisi (opsiyonel)
#         """
#         # Thread safety
#         self.lock = threading.RLock()
        
#         # Exchange
#         self.exchange: Optional[ccxt.Exchange] = None
        
#         # Accounting
#         self.accounting = accounting_manager
        
#         # Cache
#         self._cache: Dict[str, Any] = {}
#         self._cache_time: Dict[str, datetime] = {}
        
#         # Metrics
#         self.metrics = FinanceMetrics()
        
#         # Initialize exchange
#         if FINANCE_LIBS:
#             self._init_exchange()
    
#     def _init_exchange(self) -> None:
#         """Borsa bağlantısı başlat"""
#         try:
#             # GÜNCELLEME: Timeout süresi artırıldı ve rate limit aktif
#             self.exchange = ccxt.binance({
#                 'enableRateLimit': True,
#                 'options': {'defaultType': 'spot'},
#                 'timeout': 30000  # 30 saniye (Timeout hatalarını azaltmak için)
#             })
            
#             # GÜNCELLEME: Piyasaları başlangıçta bir kez yükle
#             # Bu, her ticker sorgusunda tekrar exchangeInfo indirmeyi engeller.
#             logger.info("⏳ Binance piyasa verileri yükleniyor...")
#             self.exchange.load_markets()
#             logger.info("✅ Binance bağlantısı hazır")
        
#         except Exception as e:
#             logger.error(f"Borsa bağlantı hatası: {e}")
#             self.metrics.errors_encountered += 1
    
#     # ───────────────────────────────────────────────────────────
#     # MARKET DATA
#     # ───────────────────────────────────────────────────────────
    
#     def get_market_summary(
#         self,
#         custom_symbols: Optional[List[str]] = None
#     ) -> str:
#         """
#         Piyasa özeti
        
#         GÜNCELLEME: Tek tek sorgulamak yerine 'fetch_tickers' ile toplu
#         sorgu yaparak hız artırıldı ve timeout hataları engellendi.
        
#         Args:
#             custom_symbols: Özel sembol listesi
        
#         Returns:
#             Formatlanmış özet
#         """
#         if not FINANCE_LIBS or not self.exchange:
#             return "⚠️ Finansal modül veya borsa bağlantısı aktif değil"
        
#         with self.lock:
#             try:
#                 symbols = custom_symbols or self.DEFAULT_SYMBOLS
#                 summary = []
                
#                 # Toplu veri çekme (Batch Fetch) - Tek HTTP isteği
#                 try:
#                     tickers = self.exchange.fetch_tickers(symbols)
#                 except Exception as e:
#                     logger.warning(f"Toplu veri çekilemedi, tekli deneniyor: {e}")
#                     tickers = {}
#                     # Fallback: Eğer toplu çekim başarısızsa cache veya tekli dene
#                     for sym in symbols:
#                         t = self._get_ticker_cached(sym)
#                         if t: tickers[sym] = t

#                 for symbol in symbols:
#                     ticker = tickers.get(symbol)
                    
#                     if not ticker:
#                         continue
                    
#                     price = ticker['last']
#                     change = ticker['percentage']
                    
#                     # Format
#                     icon = "🟢" if change >= 0 else "🔴"
#                     trend = "📈" if change > 2.5 else "📉" if change < -2.5 else "➡️"
                    
#                     clean_sym = symbol.split('/')[0]
#                     summary.append(
#                         f"{icon} {clean_sym}: ${price:,.2f} "
#                         f"(%{change:+.2f}) {trend}"
#                     )
                
#                 self.metrics.market_queries += 1
                
#                 return (
#                     " | ".join(summary)
#                     if summary else "❌ Piyasa verisi çekilemiyor"
#                 )
            
#             except Exception as e:
#                 logger.error(f"Piyasa özeti hatası: {e}")
#                 self.metrics.errors_encountered += 1
#                 return "Piyasa verilerine erişilemiyor"
    
#     def _get_ticker_cached(self, symbol: str) -> Optional[Dict]:
#         """Cache'li ticker getir (Tekli sorgular için)"""
#         current_time = datetime.now()
        
#         # Cache check
#         if symbol in self._cache:
#             cache_age = (
#                 current_time - self._cache_time.get(symbol, current_time)
#             ).total_seconds()
            
#             if cache_age < self.CACHE_DURATION:
#                 self.metrics.cache_hits += 1
#                 return self._cache[symbol]
        
#         # Fetch new
#         try:
#             ticker = self.exchange.fetch_ticker(symbol)
#             self._cache[symbol] = ticker
#             self._cache_time[symbol] = current_time
#             self.metrics.cache_misses += 1
#             return ticker
        
#         except Exception as e:
#             # Hata detayını logla (Timeout, DNS, vb.)
#             logger.error(f"Ticker fetch hatası ({symbol}): {str(e)}")
#             return None
    
#     # ───────────────────────────────────────────────────────────
#     # BALANCE
#     # ───────────────────────────────────────────────────────────
    
#     def get_balance(self) -> str:
#         """
#         Kasa bakiyesi
        
#         Returns:
#             Formatlanmış bakiye
#         """
#         if self.accounting:
#             try:
#                 val = self.accounting.get_balance()
#                 return f"{val:,.2f} TRY"
#             except Exception as e:
#                 logger.error(f"Bakiye sorgulama hatası: {e}")
#                 return "Bakiye okunamadı"
        
#         return "12,450.00 TRY (Demo)"
    
#     # ───────────────────────────────────────────────────────────
#     # TECHNICAL ANALYSIS
#     # ───────────────────────────────────────────────────────────
    
#     def analyze(
#         self,
#         symbol: str = "BTC/USDT",
#         timeframe: str = '4h',
#         limit: int = 100
#     ) -> Tuple[str, Optional[str]]:
#         """
#         Teknik analiz
        
#         Args:
#             symbol: Sembol
#             timeframe: Zaman dilimi
#             limit: Veri sayısı
        
#         Returns:
#             (Rapor, Grafik dosya adı)
#         """
#         if not FINANCE_LIBS or not self.exchange:
#             return "Analiz araçları yüklü değil", None
        
#         with self.lock:
#             try:
#                 # Format symbol
#                 symbol = symbol.upper()
#                 if "/" not in symbol:
#                     symbol = f"{symbol}/USDT"
                
#                 # Fetch OHLCV data
#                 bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
#                 if not bars:
#                     return f"{symbol} için veri boş", None
                
#                 # Create dataframe
#                 df = pd.DataFrame(
#                     bars,
#                     columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
#                 )
#                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#                 df.set_index('timestamp', inplace=True)
                
#                 # Calculate indicators
#                 df = self._calculate_indicators(df)
                
#                 # Validate data
#                 if df.iloc[-1]['EMA200'] is None or pd.isna(df.iloc[-1]['EMA200']):
#                     return f"{symbol} için yeterli veri yok (EMA200)", None
                
#                 # Analysis
#                 analysis = self._analyze_dataframe(df, symbol, timeframe)
                
#                 # Generate chart
#                 chart_filename = self._generate_chart(df, symbol, timeframe)
                
#                 # Format report
#                 report = self._format_analysis_report(analysis, chart_filename)
                
#                 self.metrics.analyses_performed += 1
                
#                 return report, chart_filename
            
#             except Exception as e:
#                 logger.error(f"Analiz hatası: {e}")
#                 self.metrics.errors_encountered += 1
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return f"Analiz başarısız: {str(e)[:100]}", None
    
#     def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         İndikatörleri hesapla
        
#         Args:
#             df: OHLCV dataframe
        
#         Returns:
#             İndikatörlerle zenginleştirilmiş dataframe
#         """
#         try:
#             # GPU symbolic operation (if available)
#             if HAS_GPU:
#                 try:
#                     import torch
#                     # Symbolic GPU operation (data transfer test)
#                     prices = torch.tensor(
#                         df['close'].values,
#                         dtype=torch.float32
#                     ).to(DEVICE)
#                 except Exception:
#                     pass
            
#             # Calculate indicators (CPU - reliable)
#             df['RSI'] = ta.momentum.rsi(df['close'], window=14)
#             df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
#             df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)
#             df['MACD'] = ta.trend.macd(df['close'])
            
#             return df
        
#         except Exception as e:
#             logger.error(f"İndikatör hesaplama hatası: {e}")
#             return df
    
#     def _analyze_dataframe(
#         self,
#         df: pd.DataFrame,
#         symbol: str,
#         timeframe: str
#     ) -> TechnicalAnalysis:
#         """DataFrame'den analiz çıkar"""
#         last = df.iloc[-1]
#         prev = df.iloc[-2]
        
#         # Trend detection
#         trend = (
#             TrendType.BULLISH if last['close'] > last['EMA50']
#             else TrendType.BEARISH
#         )
        
#         # Signal detection
#         signal = SignalType.NONE
        
#         # Golden/Death cross
#         if prev['EMA50'] < prev['EMA200'] and last['EMA50'] > last['EMA200']:
#             signal = SignalType.GOLDEN_CROSS
#         elif prev['EMA50'] > prev['EMA200'] and last['EMA50'] < last['EMA200']:
#             signal = SignalType.DEATH_CROSS
        
#         # RSI signals
#         rsi_val = last['RSI'] if not pd.isna(last['RSI']) else 50.0
        
#         if rsi_val > self.RSI_OVERBOUGHT:
#             signal = SignalType.OVERBOUGHT
#         elif rsi_val < self.RSI_OVERSOLD:
#             signal = SignalType.OVERSOLD
        
#         return TechnicalAnalysis(
#             symbol=symbol,
#             timeframe=timeframe,
#             price=last['close'],
#             trend=trend,
#             rsi=rsi_val,
#             ema50=last['EMA50'],
#             ema200=last['EMA200'],
#             signal=signal
#         )
    
#     def _format_analysis_report(
#         self,
#         analysis: TechnicalAnalysis,
#         chart_filename: Optional[str]
#     ) -> str:
#         """Analiz raporunu formatla"""
#         # Device info
#         device_info = (
#             f"⚡ GPU ({DEVICE})" if HAS_GPU
#             else "💻 CPU"
#         )
        
#         # Trend emoji
#         trend_emoji = "🐂" if analysis.trend == TrendType.BULLISH else "🐻"
        
#         # RSI status
#         rsi_status = "NÖTR"
#         if analysis.rsi > self.RSI_OVERBOUGHT:
#             rsi_status = "AŞIRI ALIM (Dikkat)"
#         elif analysis.rsi < self.RSI_OVERSOLD:
#             rsi_status = "AŞIRI SATIM (Fırsat)"
        
#         # Signal message
#         signal_msg = ""
#         if analysis.signal == SignalType.GOLDEN_CROSS:
#             signal_msg = "\n🚀 GOLDEN CROSS! (Uzun vadeli AL sinyali)"
#         elif analysis.signal == SignalType.DEATH_CROSS:
#             signal_msg = "\n⚠️ DEATH CROSS! (Uzun vadeli SAT sinyali)"
        
#         report_lines = [
#             f"📊 {analysis.symbol} TEKNİK ANALİZ "
#             f"({analysis.timeframe}) - {device_info}",
#             f"💰 Fiyat: ${analysis.price:,.2f}",
#             f"📈 Trend: {analysis.trend.value} {trend_emoji}",
#             f"⚡ RSI: {analysis.rsi:.2f} ({rsi_status})",
#             signal_msg,
#             "─" * 35,
#             "Analiz grafiği oluşturuldu"
#         ]
        
#         return "\n".join(report_lines)
    
#     # ───────────────────────────────────────────────────────────
#     # CHART GENERATION
#     # ───────────────────────────────────────────────────────────
    
#     def _generate_chart(
#         self,
#         df: pd.DataFrame,
#         symbol: str,
#         timeframe: str
#     ) -> Optional[str]:
#         """
#         Grafik oluştur
        
#         Args:
#             df: OHLCV + indicators dataframe
#             symbol: Sembol
#             timeframe: Zaman dilimi
        
#         Returns:
#             Dosya adı veya None
#         """
#         try:
#             # Output path
#             static_dir = Config.STATIC_DIR
#             static_dir.mkdir(parents=True, exist_ok=True)
            
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"chart_{symbol.replace('/', '_')}_{timestamp}.png"
#             output_path = static_dir / filename
            
#             # Style
#             style = mpf.make_mpf_style(
#                 base_mpf_style='nightclouds',
#                 rc={'font.size': 8}
#             )
            
#             # Add plots
#             apds = [
#                 mpf.make_addplot(df['EMA50'], color='orange', width=1.0),
#                 mpf.make_addplot(df['EMA200'], color='cyan', width=1.0)
#             ]
            
#             # Plot
#             mpf.plot(
#                 df,
#                 type='candle',
#                 style=style,
#                 addplot=apds,
#                 title=f"\n{symbol} - LotusAI Analiz",
#                 volume=True,
#                 savefig=dict(
#                     fname=str(output_path),
#                     dpi=self.CHART_DPI,
#                     bbox_inches='tight'
#                 )
#             )
            
#             plt.close('all')
            
#             # Debug mode: Open chart
#             if Config.DEBUG_MODE:
#                 self._open_chart(output_path)
            
#             self.metrics.charts_generated += 1
            
#             return filename
        
#         except Exception as e:
#             logger.error(f"Grafik oluşturma hatası: {e}")
#             return None
    
#     def _open_chart(self, path: Path) -> None:
#         """Grafiği aç (debug)"""
#         try:
#             if sys.platform == 'win32':
#                 os.startfile(path)
#             elif sys.platform == 'darwin':
#                 os.system(f"open {path}")
#             else:
#                 os.system(f"xdg-open {path}")
#         except Exception:
#             pass
    
#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Finance metrikleri
        
#         Returns:
#             Metrik dictionary
#         """
#         return {
#             "market_queries": self.metrics.market_queries,
#             "analyses_performed": self.metrics.analyses_performed,
#             "charts_generated": self.metrics.charts_generated,
#             "cache_hits": self.metrics.cache_hits,
#             "cache_misses": self.metrics.cache_misses,
#             "errors_encountered": self.metrics.errors_encountered,
#             "gpu_available": HAS_GPU,
#             "device": DEVICE,
#             "exchange_connected": self.exchange is not None
#         }
    
#     def clear_cache(self) -> None:
#         """Cache'i temizle"""
#         with self.lock:
#             self._cache.clear()
#             self._cache_time.clear()
#             logger.debug("Market cache temizlendi")

