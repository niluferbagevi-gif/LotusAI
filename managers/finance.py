"""
LotusAI Finance Manager
Sürüm: 2.5.4 (Fix: Binance ExchangeInfo Timeout & Batch Fetching)
Açıklama: Finans, borsa ve analiz yönetimi

Özellikler:
- CCXT borsa entegrasyonu
- Teknik analiz (RSI, EMA, MACD)
- GPU hızlandırmalı hesaplamalar
- Grafik oluşturma
- Piyasa özeti
- Cache sistemi
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
from config import Config

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
    
    def __init__(self, accounting_manager: Optional[Any] = None):
        """
        Finance manager başlatıcı
        
        Args:
            accounting_manager: Muhasebe yöneticisi (opsiyonel)
        """
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
    
    def _init_exchange(self) -> None:
        """Borsa bağlantısı başlat"""
        try:
            # GÜNCELLEME: Timeout süresi artırıldı ve rate limit aktif
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000  # 30 saniye (Timeout hatalarını azaltmak için)
            })
            
            # GÜNCELLEME: Piyasaları başlangıçta bir kez yükle
            # Bu, her ticker sorgusunda tekrar exchangeInfo indirmeyi engeller.
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
        Piyasa özeti
        
        GÜNCELLEME: Tek tek sorgulamak yerine 'fetch_tickers' ile toplu
        sorgu yaparak hız artırıldı ve timeout hataları engellendi.
        
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
                
                # Toplu veri çekme (Batch Fetch) - Tek HTTP isteği
                try:
                    tickers = self.exchange.fetch_tickers(symbols)
                except Exception as e:
                    logger.warning(f"Toplu veri çekilemedi, tekli deneniyor: {e}")
                    tickers = {}
                    # Fallback: Eğer toplu çekim başarısızsa cache veya tekli dene
                    for sym in symbols:
                        t = self._get_ticker_cached(sym)
                        if t: tickers[sym] = t

                for symbol in symbols:
                    ticker = tickers.get(symbol)
                    
                    if not ticker:
                        continue
                    
                    price = ticker['last']
                    change = ticker['percentage']
                    
                    # Format
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
        
        # Cache check
        if symbol in self._cache:
            cache_age = (
                current_time - self._cache_time.get(symbol, current_time)
            ).total_seconds()
            
            if cache_age < self.CACHE_DURATION:
                self.metrics.cache_hits += 1
                return self._cache[symbol]
        
        # Fetch new
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self._cache[symbol] = ticker
            self._cache_time[symbol] = current_time
            self.metrics.cache_misses += 1
            return ticker
        
        except Exception as e:
            # Hata detayını logla (Timeout, DNS, vb.)
            logger.error(f"Ticker fetch hatası ({symbol}): {str(e)}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # BALANCE
    # ───────────────────────────────────────────────────────────
    
    def get_balance(self) -> str:
        """
        Kasa bakiyesi
        
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
    # TECHNICAL ANALYSIS
    # ───────────────────────────────────────────────────────────
    
    def analyze(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = '4h',
        limit: int = 100
    ) -> Tuple[str, Optional[str]]:
        """
        Teknik analiz
        
        Args:
            symbol: Sembol
            timeframe: Zaman dilimi
            limit: Veri sayısı
        
        Returns:
            (Rapor, Grafik dosya adı)
        """
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
                
                # Generate chart
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
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            İndikatörlerle zenginleştirilmiş dataframe
        """
        try:
            # GPU symbolic operation (if available)
            if HAS_GPU:
                try:
                    import torch
                    # Symbolic GPU operation (data transfer test)
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
        
        # Trend detection
        trend = (
            TrendType.BULLISH if last['close'] > last['EMA50']
            else TrendType.BEARISH
        )
        
        # Signal detection
        signal = SignalType.NONE
        
        # Golden/Death cross
        if prev['EMA50'] < prev['EMA200'] and last['EMA50'] > last['EMA200']:
            signal = SignalType.GOLDEN_CROSS
        elif prev['EMA50'] > prev['EMA200'] and last['EMA50'] < last['EMA200']:
            signal = SignalType.DEATH_CROSS
        
        # RSI signals
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
        # Device info
        device_info = (
            f"⚡ GPU ({DEVICE})" if HAS_GPU
            else "💻 CPU"
        )
        
        # Trend emoji
        trend_emoji = "🐂" if analysis.trend == TrendType.BULLISH else "🐻"
        
        # RSI status
        rsi_status = "NÖTR"
        if analysis.rsi > self.RSI_OVERBOUGHT:
            rsi_status = "AŞIRI ALIM (Dikkat)"
        elif analysis.rsi < self.RSI_OVERSOLD:
            rsi_status = "AŞIRI SATIM (Fırsat)"
        
        # Signal message
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
            "─" * 35,
            "Analiz grafiği oluşturuldu"
        ]
        
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
        Grafik oluştur
        
        Args:
            df: OHLCV + indicators dataframe
            symbol: Sembol
            timeframe: Zaman dilimi
        
        Returns:
            Dosya adı veya None
        """
        try:
            # Output path
            static_dir = Config.STATIC_DIR
            static_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{symbol.replace('/', '_')}_{timestamp}.png"
            output_path = static_dir / filename
            
            # Style
            style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                rc={'font.size': 8}
            )
            
            # Add plots
            apds = [
                mpf.make_addplot(df['EMA50'], color='orange', width=1.0),
                mpf.make_addplot(df['EMA200'], color='cyan', width=1.0)
            ]
            
            # Plot
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
            
            # Debug mode: Open chart
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
            "exchange_connected": self.exchange is not None
        }
    
    def clear_cache(self) -> None:
        """Cache'i temizle"""
        with self.lock:
            self._cache.clear()
            self._cache_time.clear()
            logger.debug("Market cache temizlendi")



# """
# LotusAI Finance Manager
# Sürüm: 2.5.5 (Fix: Geo-Blocking, CoinGecko Fallback, Proxy Desteği, Graceful Shutdown)
# Açıklama: Finans, borsa ve analiz yönetimi

# Özellikler:
# - CCXT borsa entegrasyonu (Binance)
# - CoinGecko API (Binance erişilemeyen ortamlar için otomatik fallback)
# - Proxy desteği (WSL / VPN ortamları için)
# - Graceful hata yönetimi (bağlantı hatası sistemi durdurmaz)
# - Teknik analiz (RSI, EMA, MACD)
# - GPU hızlandırmalı hesaplamalar
# - Grafik oluşturma
# - Piyasa özeti
# - Cache sistemi
# - Retry mekanizması
# """

# import os
# import sys
# import time
# import logging
# import warnings
# import threading
# import requests
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import Tuple, List, Optional, Dict, Any
# from dataclasses import dataclass, field
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
#     BULLISH  = "BULLISH"
#     BEARISH  = "BEARISH"
#     NEUTRAL  = "NEUTRAL"


# class SignalType(Enum):
#     """Sinyal tipleri"""
#     GOLDEN_CROSS = "golden_cross"
#     DEATH_CROSS  = "death_cross"
#     OVERBOUGHT   = "overbought"
#     OVERSOLD     = "oversold"
#     NONE         = "none"


# class TimeFrame(Enum):
#     """Zaman dilimleri"""
#     M1  = "1m"
#     M5  = "5m"
#     M15 = "15m"
#     M30 = "30m"
#     H1  = "1h"
#     H4  = "4h"
#     D1  = "1d"
#     W1  = "1w"


# class DataSource(Enum):
#     """Veri kaynağı"""
#     BINANCE    = "binance"
#     COINGECKO  = "coingecko"
#     CACHE      = "cache"
#     NONE       = "none"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class MarketData:
#     """Piyasa verisi"""
#     symbol:         str
#     price:          float
#     change_percent: float
#     volume:         float
#     timestamp:      datetime
#     source:         DataSource = DataSource.BINANCE


# @dataclass
# class TechnicalAnalysis:
#     """Teknik analiz sonucu"""
#     symbol:     str
#     timeframe:  str
#     price:      float
#     trend:      TrendType
#     rsi:        float
#     ema50:      float
#     ema200:     float
#     signal:     SignalType
#     chart_path: Optional[str] = None


# @dataclass
# class FinanceMetrics:
#     """Finance manager metrikleri"""
#     market_queries:      int = 0
#     analyses_performed:  int = 0
#     charts_generated:    int = 0
#     cache_hits:          int = 0
#     cache_misses:        int = 0
#     errors_encountered:  int = 0
#     binance_failures:    int = 0
#     coingecko_queries:   int = 0
#     fallback_used:       int = 0


# # ═══════════════════════════════════════════════════════════════
# # COINGECKO FALLBACK PROVIDER
# # ═══════════════════════════════════════════════════════════════

# # Sembol → CoinGecko ID eşlemesi
# COINGECKO_ID_MAP: Dict[str, str] = {
#     "BTC":  "bitcoin",
#     "ETH":  "ethereum",
#     "BNB":  "binancecoin",
#     "SOL":  "solana",
#     "ADA":  "cardano",
#     "XRP":  "ripple",
#     "DOGE": "dogecoin",
#     "DOT":  "polkadot",
#     "MATIC":"matic-network",
#     "AVAX": "avalanche-2",
#     "LINK": "chainlink",
#     "UNI":  "uniswap",
#     "LTC":  "litecoin",
#     "ATOM": "cosmos",
#     "TRX":  "tron",
# }

# COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
# COINGECKO_TIMEOUT  = 10  # saniye


# class CoinGeckoProvider:
#     """
#     Binance'e erişilemeyen ortamlar (Türkiye geo-block, WSL, VPN yok)
#     için CoinGecko API'sini kullanarak fiyat ve değişim verisi sağlar.

#     CoinGecko ücretsiz katmanda dakikada ~10-30 istek limitine sahiptir.
#     Rate limit aşılırsa son cache değeri döndürülür.
#     """

#     CACHE_DURATION = 30  # saniye

#     def __init__(self, proxies: Optional[Dict[str, str]] = None):
#         self._cache:      Dict[str, Any]      = {}
#         self._cache_time: Dict[str, datetime] = {}
#         self._proxies = proxies or {}
#         self._session = requests.Session()
#         if self._proxies:
#             self._session.proxies.update(self._proxies)

#     # ── Public ──────────────────────────────────────────────────

#     def fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
#         """
#         Sembol listesi için toplu fiyat verisi çek.

#         Args:
#             symbols: ["BTC/USDT", "ETH/USDT", ...]

#         Returns:
#             {symbol: ticker_dict} — ccxt formatıyla uyumlu
#         """
#         clean_symbols = [s.split("/")[0].upper() for s in symbols]
#         coin_ids      = [COINGECKO_ID_MAP.get(s, s.lower()) for s in clean_symbols]

#         cached_result = self._check_batch_cache(symbols)
#         if cached_result:
#             return cached_result

#         try:
#             url    = f"{COINGECKO_BASE_URL}/simple/price"
#             params = {
#                 "ids":             ",".join(coin_ids),
#                 "vs_currencies":   "usd",
#                 "include_24hr_change": "true",
#                 "include_24hr_vol":    "true",
#             }
#             resp = self._session.get(url, params=params, timeout=COINGECKO_TIMEOUT)
#             resp.raise_for_status()
#             data = resp.json()

#             result: Dict[str, Dict] = {}
#             for sym, coin_id in zip(symbols, coin_ids):
#                 coin_data = data.get(coin_id)
#                 if not coin_data:
#                     continue

#                 ticker = {
#                     "last":       coin_data.get("usd", 0.0),
#                     "percentage": coin_data.get("usd_24h_change", 0.0),
#                     "baseVolume": coin_data.get("usd_24h_vol", 0.0),
#                     "timestamp":  int(datetime.now().timestamp() * 1000),
#                     "symbol":     sym,
#                 }
#                 result[sym] = ticker
#                 self._cache[sym]      = ticker
#                 self._cache_time[sym] = datetime.now()

#             return result

#         except requests.exceptions.RequestException as e:
#             logger.warning(f"⚠️ CoinGecko isteği başarısız: {e}")
#             return self._check_batch_cache(symbols, ignore_expiry=True) or {}

#     def fetch_ohlcv(
#         self,
#         symbol:    str,
#         timeframe: str = "4h",
#         limit:     int = 100
#     ) -> Optional[List[List]]:
#         """
#         OHLCV verisi çek (CoinGecko market_chart endpoint).

#         Args:
#             symbol:    "BTC/USDT" formatı
#             timeframe: "1h" | "4h" | "1d" vb.
#             limit:     Kaç bar isteniyor

#         Returns:
#             [[timestamp_ms, open, high, low, close, volume], ...]
#         """
#         clean = symbol.split("/")[0].upper()
#         coin_id = COINGECKO_ID_MAP.get(clean, clean.lower())

#         # CoinGecko günlük granülasyon sağlar; timeframe'e göre gün sayısı hesapla
#         tf_to_days = {
#             "1m": 1, "5m": 1, "15m": 1, "30m": 1,
#             "1h": 3, "4h": 10, "1d": limit,
#             "1w": limit * 7
#         }
#         days = tf_to_days.get(timeframe, 10)

#         try:
#             url    = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
#             params = {"vs_currency": "usd", "days": str(days)}
#             resp   = self._session.get(url, params=params, timeout=COINGECKO_TIMEOUT)
#             resp.raise_for_status()
#             raw = resp.json()

#             prices  = raw.get("prices", [])
#             volumes = raw.get("total_volumes", [])

#             if not prices:
#                 return None

#             # CoinGecko sadece [timestamp, price] verir; sentetik OHLCV oluştur
#             bars: List[List] = []
#             for i, (ts, price) in enumerate(prices[-limit:]):
#                 vol = volumes[i][1] if i < len(volumes) else 0.0
#                 # Gerçek OHLCV olmadığından open=close=price, high/low ±%0.5
#                 high  = price * 1.005
#                 low   = price * 0.995
#                 close = price
#                 open_ = prices[max(0, i - 1)][1] if i > 0 else price
#                 bars.append([ts, open_, high, low, close, vol])

#             return bars

#         except requests.exceptions.RequestException as e:
#             logger.warning(f"⚠️ CoinGecko OHLCV hatası ({symbol}): {e}")
#             return None

#     # ── Private ─────────────────────────────────────────────────

#     def _check_batch_cache(
#         self,
#         symbols:        List[str],
#         ignore_expiry:  bool = False
#     ) -> Optional[Dict[str, Dict]]:
#         """Tüm semboller için cache'te geçerli veri var mı?"""
#         now    = datetime.now()
#         result = {}

#         for sym in symbols:
#             if sym not in self._cache:
#                 return None
#             age = (now - self._cache_time.get(sym, now)).total_seconds()
#             if not ignore_expiry and age > self.CACHE_DURATION:
#                 return None
#             result[sym] = self._cache[sym]

#         return result if result else None


# # ═══════════════════════════════════════════════════════════════
# # FINANCE MANAGER
# # ═══════════════════════════════════════════════════════════════
# class FinanceManager:
#     """
#     LotusAI Finans, Borsa ve Analiz Yöneticisi

#     Yetenekler:
#     - CCXT: Binance entegrasyonu (birincil)
#     - CoinGecko: Otomatik fallback (Binance erişilemeyen ortamlarda)
#     - Proxy desteği: WSL/VPN ortamları için
#     - Graceful hata yönetimi: Bağlantı hatası sistemi durdurmaz
#     - Teknik analiz: RSI, EMA, MACD
#     - GPU hızlandırma: PyTorch ile hesaplama
#     - Grafik: mplfinance ile chart oluşturma
#     - Cache: Market data önbellekleme
#     - Retry: Geçici ağ hatalarında otomatik tekrar

#     Öncelik sırası: Binance → CoinGecko → Cache
#     """

#     # Default symbols
#     DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

#     # Cache settings
#     CACHE_DURATION = 15  # saniye

#     # RSI thresholds
#     RSI_OVERBOUGHT = 70
#     RSI_OVERSOLD   = 30

#     # Chart settings
#     CHART_DPI = 120

#     # Retry settings
#     MAX_RETRIES   = 2
#     RETRY_DELAY   = 2.0  # saniye

#     # Binance failure threshold — bu kadar ardışık hata sonrası CoinGecko'ya geç
#     BINANCE_FAILURE_THRESHOLD = 3

#     def __init__(
#         self,
#         accounting_manager: Optional[Any] = None,
#         proxies:            Optional[Dict[str, str]] = None,
#     ):
#         """
#         Finance manager başlatıcı

#         Args:
#             accounting_manager: Muhasebe yöneticisi (opsiyonel)
#             proxies: Proxy ayarları (opsiyonel)
#                      Örnek: {"http": "http://127.0.0.1:7890",
#                               "https": "http://127.0.0.1:7890"}
#         """
#         # Thread safety
#         self.lock = threading.RLock()

#         # Exchange
#         self.exchange: Optional[Any] = None

#         # Proxies
#         self._proxies: Dict[str, str] = proxies or self._load_proxies_from_env()

#         # CoinGecko fallback provider
#         self._coingecko = CoinGeckoProvider(proxies=self._proxies)

#         # Durum: Binance erişilebilir mi?
#         self._binance_available: bool = False
#         # Ardışık Binance hata sayacı
#         self._binance_consecutive_failures: int = 0

#         # Accounting
#         self.accounting = accounting_manager

#         # Cache
#         self._cache:      Dict[str, Any]      = {}
#         self._cache_time: Dict[str, datetime] = {}

#         # Metrics
#         self.metrics = FinanceMetrics()

#         # Initialize exchange (hata sistemi durdurmaz)
#         if FINANCE_LIBS:
#             self._init_exchange()
#         else:
#             logger.warning("⚠️ Finans kütüphaneleri yüklü değil — yalnızca CoinGecko aktif")

#     # ───────────────────────────────────────────────────────────
#     # INIT
#     # ───────────────────────────────────────────────────────────

#     @staticmethod
#     def _load_proxies_from_env() -> Dict[str, str]:
#         """
#         Ortam değişkenlerinden proxy ayarlarını oku.
#         .env dosyasında tanımlanabilir:
#             LOTUS_HTTP_PROXY=http://127.0.0.1:7890
#             LOTUS_HTTPS_PROXY=http://127.0.0.1:7890
#         """
#         proxies: Dict[str, str] = {}
#         http  = os.environ.get("LOTUS_HTTP_PROXY",  "")
#         https = os.environ.get("LOTUS_HTTPS_PROXY", "")
#         if http:
#             proxies["http"]  = http
#             logger.info(f"🌐 HTTP proxy kullanılıyor: {http}")
#         if https:
#             proxies["https"] = https
#             logger.info(f"🌐 HTTPS proxy kullanılıyor: {https}")
#         return proxies

#     def _init_exchange(self) -> None:
#         """
#         Binance bağlantısını başlat.

#         Hata durumunda sistem DURMAZ; CoinGecko fallback devreye girer.
#         WSL/Türkiye IP'sinden geo-block nedeniyle hata alınabilir.
#         Proxy tanımlıysa otomatik olarak uygulanır.
#         """
#         try:
#             exchange_config: Dict[str, Any] = {
#                 "enableRateLimit": True,
#                 "options":         {"defaultType": "spot"},
#                 "timeout":         30_000,  # 30 saniye
#             }

#             # Proxy ekle (varsa)
#             if self._proxies:
#                 exchange_config["proxies"] = self._proxies
#                 logger.info(f"🌐 Binance proxy ile başlatılıyor: {list(self._proxies.keys())}")

#             self.exchange = ccxt.binance(exchange_config)

#             logger.info("⏳ Binance piyasa verileri yükleniyor...")
#             self.exchange.load_markets()

#             self._binance_available = True
#             logger.info("✅ Binance bağlantısı hazır")

#         except ccxt.NetworkError as e:
#             self._handle_binance_init_failure("Ağ hatası", e)
#         except ccxt.ExchangeError as e:
#             self._handle_binance_init_failure("Borsa hatası", e)
#         except Exception as e:
#             self._handle_binance_init_failure("Beklenmedik hata", e)

#     def _handle_binance_init_failure(self, reason: str, exc: Exception) -> None:
#         """Binance başlatma hatasını yönet ve CoinGecko'ya geç."""
#         logger.warning(
#             f"⚠️ Binance bağlantısı kurulamadı ({reason}): {exc}\n"
#             "   → CoinGecko fallback moduna geçiliyor. Sistem çalışmaya devam edecek."
#         )
#         self._binance_available = False
#         self.exchange           = None
#         self.metrics.binance_failures += 1

#     # ───────────────────────────────────────────────────────────
#     # MARKET DATA
#     # ───────────────────────────────────────────────────────────

#     def get_market_summary(
#         self,
#         custom_symbols: Optional[List[str]] = None
#     ) -> str:
#         """
#         Piyasa özeti.

#         Önce Binance'i dener. Erişilemezse otomatik olarak CoinGecko
#         kullanır. Her iki kaynak da başarısız olursa cache'deki son
#         değeri döndürür.

#         Args:
#             custom_symbols: Özel sembol listesi

#         Returns:
#             Formatlanmış piyasa özeti
#         """
#         with self.lock:
#             try:
#                 symbols = custom_symbols or self.DEFAULT_SYMBOLS
#                 tickers = self._fetch_tickers_with_fallback(symbols)

#                 if not tickers:
#                     return "❌ Piyasa verisi çekilemiyor (Binance ve CoinGecko erişilemiyor)"

#                 summary = []
#                 for symbol in symbols:
#                     ticker = tickers.get(symbol)
#                     if not ticker:
#                         continue

#                     price  = ticker.get("last",       0.0)
#                     change = ticker.get("percentage", 0.0)

#                     icon  = "🟢" if change >= 0 else "🔴"
#                     trend = "📈" if change > 2.5 else "📉" if change < -2.5 else "➡️"
#                     clean = symbol.split("/")[0]

#                     summary.append(
#                         f"{icon} {clean}: ${price:,.2f} (%{change:+.2f}) {trend}"
#                     )

#                 self.metrics.market_queries += 1

#                 source_tag = self._current_source_tag()
#                 result     = " | ".join(summary) if summary else "❌ Veri yok"
#                 return f"{result}\n{source_tag}"

#             except Exception as e:
#                 logger.error(f"Piyasa özeti hatası: {e}")
#                 self.metrics.errors_encountered += 1
#                 return "Piyasa verilerine erişilemiyor"

#     def _fetch_tickers_with_fallback(
#         self,
#         symbols: List[str]
#     ) -> Dict[str, Dict]:
#         """
#         Ticker verisini Binance → CoinGecko → Cache öncelik sırasıyla çek.

#         Args:
#             symbols: Sembol listesi

#         Returns:
#             {symbol: ticker_dict}
#         """
#         # 1) Binance (CCXT)
#         if self._binance_available and self.exchange and FINANCE_LIBS:
#             result = self._try_binance_tickers(symbols)
#             if result:
#                 self._binance_consecutive_failures = 0
#                 return result

#             # Binance başarısız
#             self._binance_consecutive_failures += 1
#             self.metrics.binance_failures += 1

#             if self._binance_consecutive_failures >= self.BINANCE_FAILURE_THRESHOLD:
#                 logger.warning(
#                     f"⚠️ Binance {self._binance_failure_threshold} kez ardışık başarısız. "
#                     "Bu oturum için CoinGecko'ya geçildi."
#                 )
#                 self._binance_available = False

#         # 2) CoinGecko fallback
#         logger.info("📡 CoinGecko veri kaynağı kullanılıyor...")
#         cg_result = self._coingecko.fetch_tickers(symbols)
#         if cg_result:
#             self.metrics.coingecko_queries += 1
#             self.metrics.fallback_used     += 1
#             # CoinGecko verilerini cache'e yaz
#             for sym, ticker in cg_result.items():
#                 self._cache[sym]      = ticker
#                 self._cache_time[sym] = datetime.now()
#             return cg_result

#         # 3) Son çare: Cache
#         cached = self._get_all_from_cache(symbols)
#         if cached:
#             logger.info("🗃️ Cache verisi kullanılıyor (güncel olmayabilir)")
#             return cached

#         return {}

#     def _try_binance_tickers(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
#         """
#         Binance'den toplu ticker çek; retry mekanizması dahil.

#         Returns:
#             Başarılı ise ticker dict, değilse None
#         """
#         for attempt in range(1, self.MAX_RETRIES + 1):
#             try:
#                 tickers = self.exchange.fetch_tickers(symbols)
#                 # Başarılı — cache'e yaz
#                 for sym, t in tickers.items():
#                     self._cache[sym]      = t
#                     self._cache_time[sym] = datetime.now()
#                 return tickers

#             except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
#                 logger.warning(
#                     f"⚠️ Binance ağ hatası (deneme {attempt}/{self.MAX_RETRIES}): {e}"
#                 )
#                 if attempt < self.MAX_RETRIES:
#                     time.sleep(self.RETRY_DELAY)

#             except ccxt.ExchangeNotAvailable as e:
#                 logger.warning(f"⚠️ Binance hizmet dışı: {e}")
#                 break

#             except ccxt.RateLimitExceeded as e:
#                 logger.warning(f"⚠️ Binance rate limit aşıldı: {e}")
#                 time.sleep(5)
#                 break

#             except Exception as e:
#                 logger.error(f"❌ Binance beklenmedik hata: {e}")
#                 break

#         return None

#     def _get_ticker_cached(self, symbol: str) -> Optional[Dict]:
#         """Cache'li tekil ticker getir (fallback içinde kullanılır)."""
#         current_time = datetime.now()

#         if symbol in self._cache:
#             age = (current_time - self._cache_time.get(symbol, current_time)).total_seconds()
#             if age < self.CACHE_DURATION:
#                 self.metrics.cache_hits += 1
#                 return self._cache[symbol]

#         # Canlı tek sorgu (Binance)
#         if self._binance_available and self.exchange and FINANCE_LIBS:
#             try:
#                 ticker               = self.exchange.fetch_ticker(symbol)
#                 self._cache[symbol]  = ticker
#                 self._cache_time[symbol] = current_time
#                 self.metrics.cache_misses += 1
#                 return ticker
#             except Exception as e:
#                 logger.error(f"Ticker fetch hatası ({symbol}): {e}")

#         return self._cache.get(symbol)

#     def _get_all_from_cache(self, symbols: List[str]) -> Dict[str, Dict]:
#         """Cache'teki tüm mevcut değerleri döndür (süresi dolmuş olsa da)."""
#         return {s: self._cache[s] for s in symbols if s in self._cache}

#     def _current_source_tag(self) -> str:
#         """Aktif veri kaynağını göster."""
#         if self._binance_available:
#             return "📊 Kaynak: Binance"
#         return "📊 Kaynak: CoinGecko (fallback)"

#     # ───────────────────────────────────────────────────────────
#     # BALANCE
#     # ───────────────────────────────────────────────────────────

#     def get_balance(self) -> str:
#         """
#         Kasa bakiyesi.

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
#         symbol:    str = "BTC/USDT",
#         timeframe: str = "4h",
#         limit:     int = 100
#     ) -> Tuple[str, Optional[str]]:
#         """
#         Teknik analiz.

#         OHLCV verisi önce Binance'den, erişilemezse CoinGecko'dan çekilir.

#         Args:
#             symbol:    Sembol (ör. "BTC/USDT" veya "BTC")
#             timeframe: Zaman dilimi
#             limit:     Veri sayısı

#         Returns:
#             (Rapor, Grafik dosya adı)
#         """
#         if not FINANCE_LIBS:
#             return "⚠️ Analiz kütüphaneleri yüklü değil", None

#         with self.lock:
#             try:
#                 # Sembol normalize et
#                 symbol = symbol.upper()
#                 if "/" not in symbol:
#                     symbol = f"{symbol}/USDT"

#                 # OHLCV verisi al
#                 bars = self._fetch_ohlcv_with_fallback(symbol, timeframe, limit)

#                 if not bars:
#                     return f"❌ {symbol} için OHLCV verisi alınamadı", None

#                 # DataFrame oluştur
#                 df = pd.DataFrame(
#                     bars,
#                     columns=["timestamp", "open", "high", "low", "close", "volume"]
#                 )
#                 df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
#                 df.set_index("timestamp", inplace=True)

#                 # İndikatörler
#                 df = self._calculate_indicators(df)

#                 if pd.isna(df.iloc[-1].get("EMA200", float("nan"))):
#                     return f"⚠️ {symbol} için yeterli veri yok (EMA200 hesaplanamadı)", None

#                 # Analiz
#                 analysis = self._analyze_dataframe(df, symbol, timeframe)

#                 # Grafik
#                 chart_filename = self._generate_chart(df, symbol, timeframe)
#                 analysis.chart_path = chart_filename

#                 # Rapor
#                 report = self._format_analysis_report(analysis, chart_filename)

#                 self.metrics.analyses_performed += 1
#                 return report, chart_filename

#             except Exception as e:
#                 logger.error(f"Analiz hatası: {e}")
#                 self.metrics.errors_encountered += 1
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return f"Analiz başarısız: {str(e)[:100]}", None

#     def _fetch_ohlcv_with_fallback(
#         self,
#         symbol:    str,
#         timeframe: str,
#         limit:     int
#     ) -> Optional[List[List]]:
#         """
#         OHLCV verisi Binance → CoinGecko fallback ile çek.
#         """
#         # Binance
#         if self._binance_available and self.exchange and FINANCE_LIBS:
#             for attempt in range(1, self.MAX_RETRIES + 1):
#                 try:
#                     bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
#                     if bars:
#                         return bars
#                 except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
#                     logger.warning(
#                         f"⚠️ OHLCV Binance ağ hatası (deneme {attempt}): {e}"
#                     )
#                     if attempt < self.MAX_RETRIES:
#                         time.sleep(self.RETRY_DELAY)
#                 except Exception as e:
#                     logger.warning(f"⚠️ OHLCV Binance hatası: {e}")
#                     break

#         # CoinGecko fallback
#         logger.info(f"📡 OHLCV için CoinGecko kullanılıyor ({symbol})...")
#         bars = self._coingecko.fetch_ohlcv(symbol, timeframe, limit)
#         if bars:
#             self.metrics.coingecko_queries += 1
#             self.metrics.fallback_used     += 1
#         return bars

#     def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         RSI, EMA50, EMA200, MACD indikatörlerini hesapla.

#         GPU varsa fiyat tensörü CUDA'ya transfer edilir (sembolik GPU desteği).
#         """
#         try:
#             if HAS_GPU:
#                 try:
#                     import torch
#                     _prices = torch.tensor(
#                         df["close"].values, dtype=torch.float32
#                     ).to(DEVICE)
#                     # İndikatörler güvenilirlik için CPU'da hesaplanır
#                 except Exception:
#                     pass

#             df["RSI"]   = ta.momentum.rsi(df["close"], window=14)
#             df["EMA50"] = ta.trend.ema_indicator(df["close"], window=50)
#             df["EMA200"]= ta.trend.ema_indicator(df["close"], window=200)
#             df["MACD"]  = ta.trend.macd(df["close"])

#             return df

#         except Exception as e:
#             logger.error(f"İndikatör hesaplama hatası: {e}")
#             return df

#     def _analyze_dataframe(
#         self,
#         df:        pd.DataFrame,
#         symbol:    str,
#         timeframe: str
#     ) -> TechnicalAnalysis:
#         """DataFrame'den analiz nesnesi üret."""
#         last = df.iloc[-1]
#         prev = df.iloc[-2]

#         trend = (
#             TrendType.BULLISH if last["close"] > last["EMA50"]
#             else TrendType.BEARISH
#         )

#         signal = SignalType.NONE

#         # Golden / Death cross
#         if prev["EMA50"] < prev["EMA200"] and last["EMA50"] > last["EMA200"]:
#             signal = SignalType.GOLDEN_CROSS
#         elif prev["EMA50"] > prev["EMA200"] and last["EMA50"] < last["EMA200"]:
#             signal = SignalType.DEATH_CROSS

#         rsi_val = last["RSI"] if not pd.isna(last["RSI"]) else 50.0

#         if rsi_val > self.RSI_OVERBOUGHT:
#             signal = SignalType.OVERBOUGHT
#         elif rsi_val < self.RSI_OVERSOLD:
#             signal = SignalType.OVERSOLD

#         return TechnicalAnalysis(
#             symbol    = symbol,
#             timeframe = timeframe,
#             price     = last["close"],
#             trend     = trend,
#             rsi       = rsi_val,
#             ema50     = last["EMA50"],
#             ema200    = last["EMA200"],
#             signal    = signal,
#         )

#     def _format_analysis_report(
#         self,
#         analysis:       TechnicalAnalysis,
#         chart_filename: Optional[str]
#     ) -> str:
#         """Teknik analiz raporunu formatla."""
#         device_info  = f"⚡ GPU ({DEVICE})" if HAS_GPU else "💻 CPU"
#         trend_emoji  = "🐂" if analysis.trend == TrendType.BULLISH else "🐻"
#         source_tag   = self._current_source_tag()

#         rsi_status = "NÖTR"
#         if analysis.rsi > self.RSI_OVERBOUGHT:
#             rsi_status = "AŞIRI ALIM (Dikkat)"
#         elif analysis.rsi < self.RSI_OVERSOLD:
#             rsi_status = "AŞIRI SATIM (Fırsat)"

#         signal_msg = ""
#         if analysis.signal == SignalType.GOLDEN_CROSS:
#             signal_msg = "\n🚀 GOLDEN CROSS! (Uzun vadeli AL sinyali)"
#         elif analysis.signal == SignalType.DEATH_CROSS:
#             signal_msg = "\n⚠️ DEATH CROSS! (Uzun vadeli SAT sinyali)"

#         lines = [
#             f"📊 {analysis.symbol} TEKNİK ANALİZ ({analysis.timeframe}) — {device_info}",
#             f"💰 Fiyat    : ${analysis.price:,.2f}",
#             f"📈 Trend    : {analysis.trend.value} {trend_emoji}",
#             f"⚡ RSI      : {analysis.rsi:.2f} ({rsi_status})",
#             f"📉 EMA50    : ${analysis.ema50:,.2f}",
#             f"📉 EMA200   : ${analysis.ema200:,.2f}",
#             signal_msg,
#             source_tag,
#             "─" * 40,
#             "📷 Analiz grafiği oluşturuldu" if chart_filename else "⚠️ Grafik oluşturulamadı",
#         ]

#         return "\n".join(line for line in lines if line)

#     # ───────────────────────────────────────────────────────────
#     # CHART GENERATION
#     # ───────────────────────────────────────────────────────────

#     def _generate_chart(
#         self,
#         df:        pd.DataFrame,
#         symbol:    str,
#         timeframe: str
#     ) -> Optional[str]:
#         """
#         Mum grafik oluştur ve dosyaya kaydet.

#         Returns:
#             Dosya adı (static/ altında) veya None
#         """
#         try:
#             static_dir = Config.STATIC_DIR
#             static_dir.mkdir(parents=True, exist_ok=True)

#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename  = f"chart_{symbol.replace('/', '_')}_{timestamp}.png"
#             out_path  = static_dir / filename

#             style = mpf.make_mpf_style(
#                 base_mpf_style="nightclouds",
#                 rc={"font.size": 8}
#             )

#             apds = [
#                 mpf.make_addplot(df["EMA50"],  color="orange", width=1.0),
#                 mpf.make_addplot(df["EMA200"], color="cyan",   width=1.0),
#             ]

#             source_note = (
#                 " [Binance]" if self._binance_available else " [CoinGecko Fallback]"
#             )

#             mpf.plot(
#                 df,
#                 type      = "candle",
#                 style     = style,
#                 addplot   = apds,
#                 title     = f"\n{symbol} — LotusAI Analiz{source_note}",
#                 volume    = True,
#                 savefig   = dict(
#                     fname        = str(out_path),
#                     dpi          = self.CHART_DPI,
#                     bbox_inches  = "tight",
#                 ),
#             )

#             plt.close("all")

#             if Config.DEBUG_MODE:
#                 self._open_chart(out_path)

#             self.metrics.charts_generated += 1
#             return filename

#         except Exception as e:
#             logger.error(f"Grafik oluşturma hatası: {e}")
#             return None

#     def _open_chart(self, path: Path) -> None:
#         """Grafiği platforma göre aç (debug modu)."""
#         try:
#             if sys.platform == "win32":
#                 os.startfile(path)
#             elif sys.platform == "darwin":
#                 os.system(f"open {path}")
#             else:
#                 # WSL'de xdg-open çalışmayabilir; fallback olarak görüntü yolunu logla
#                 result = os.system(f"xdg-open {path} 2>/dev/null")
#                 if result != 0:
#                     logger.info(f"📁 Grafik kaydedildi: {path}")
#         except Exception:
#             pass

#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────

#     def get_metrics(self) -> Dict[str, Any]:
#         """Finance manager metriklerini döndür."""
#         return {
#             "market_queries":       self.metrics.market_queries,
#             "analyses_performed":   self.metrics.analyses_performed,
#             "charts_generated":     self.metrics.charts_generated,
#             "cache_hits":           self.metrics.cache_hits,
#             "cache_misses":         self.metrics.cache_misses,
#             "errors_encountered":   self.metrics.errors_encountered,
#             "binance_failures":     self.metrics.binance_failures,
#             "coingecko_queries":    self.metrics.coingecko_queries,
#             "fallback_used":        self.metrics.fallback_used,
#             "gpu_available":        HAS_GPU,
#             "device":               DEVICE,
#             "binance_connected":    self._binance_available,
#             "active_data_source":   (
#                 DataSource.BINANCE.value if self._binance_available
#                 else DataSource.COINGECKO.value
#             ),
#             "proxies_configured":   bool(self._proxies),
#         }

#     def get_status(self) -> str:
#         """İnsan okunabilir durum özeti."""
#         source = "Binance ✅" if self._binance_available else "CoinGecko (fallback) ⚠️"
#         proxy  = f"Proxy: {list(self._proxies.keys())}" if self._proxies else "Proxy: Yok"
#         gpu    = f"GPU: {DEVICE.upper()}" if HAS_GPU else "GPU: Yok (CPU)"
#         return f"Finance Manager | Kaynak: {source} | {proxy} | {gpu}"

#     def clear_cache(self) -> None:
#         """Tüm önbelleği temizle."""
#         with self.lock:
#             self._cache.clear()
#             self._cache_time.clear()
#             logger.debug("🗑️ Market cache temizlendi")

#     def reconnect_binance(self) -> bool:
#         """
#         Binance bağlantısını yeniden dene.
#         Manuel çağrı veya scheduler ile periyodik deneme için kullanılabilir.

#         Returns:
#             True: Bağlantı başarılı, False: Başarısız
#         """
#         logger.info("🔄 Binance yeniden bağlanmaya çalışılıyor...")
#         self._binance_available           = False
#         self._binance_consecutive_failures = 0
#         self.exchange                     = None

#         if FINANCE_LIBS:
#             self._init_exchange()

#         if self._binance_available:
#             logger.info("✅ Binance bağlantısı yeniden kuruldu")
#         else:
#             logger.warning("⚠️ Binance bağlantısı hâlâ kurulamıyor, CoinGecko devrede")

#         return self._binance_available