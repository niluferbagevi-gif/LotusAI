"""
LotusAI NLP Manager
Sürüm: 2.5.4 (Eklendi: Erişim Seviyesi Desteği)
Açıklama: NLP ve duygu analizi yönetimi

Özellikler:
- Transformers/BERT entegrasyonu
- GPU hızlandırmalı duygu analizi
- Batch processing
- Keyword extraction
- Rezervasyon ayıklama
- Türkçe stop words
- Intent detection
- Erişim seviyesi kontrolleri
"""

import re
import logging
import threading
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.NLP")


# ═══════════════════════════════════════════════════════════════
# TRANSFORMERS & GPU
# ═══════════════════════════════════════════════════════════════
NLP_AVAILABLE = False
HAS_GPU = False
DEVICE_ID = -1  # -1: CPU, 0: GPU

try:
    import torch
    from transformers import pipeline
    NLP_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ NLP kütüphaneleri eksik\n"
        "pip install torch transformers"
    )

# GPU detection
if NLP_AVAILABLE and Config.USE_GPU:
    try:
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE_ID = 0
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 NLP GPU aktif: {gpu_name}")
            except Exception:
                logger.info("🚀 NLP GPU aktif")
    except Exception as e:
        logger.warning(f"⚠️ GPU kontrol hatası: {e}")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class Sentiment(Enum):
    """Duygu tipleri"""
    POSITIVE = "POZITIF"
    NEGATIVE = "NEGATIF"
    NEUTRAL = "NOTR"


class Intent(Enum):
    """Niyet tipleri"""
    RESERVATION = "rezervasyon"
    COMPLAINT = "şikayet"
    QUESTION = "soru"
    PRAISE = "övgü"
    ORDER = "sipariş"
    CANCEL = "iptal"
    UNKNOWN = "bilinmiyor"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class SentimentResult:
    """Duygu analizi sonucu"""
    sentiment: Sentiment
    confidence: float
    text: str


@dataclass
class ReservationData:
    """Rezervasyon verisi"""
    person_count: str
    time: str
    phone: Optional[str]
    summary: str


@dataclass
class BatchAnalysis:
    """Toplu analiz sonucu"""
    total_count: int
    positive: int
    negative: int
    neutral: int
    satisfaction_rate: float
    top_complaints: List[str]
    popular_topics: List[str]
    device_used: str


@dataclass
class NLPMetrics:
    """NLP metrikleri"""
    sentiment_analyses: int = 0
    batch_analyses: int = 0
    reservations_extracted: int = 0
    keywords_extracted: int = 0
    errors_encountered: int = 0


# ═══════════════════════════════════════════════════════════════
# NLP MANAGER
# ═══════════════════════════════════════════════════════════════
class NLPManager:
    """
    LotusAI Doğal Dil İşleme ve Duygu Analizi Yöneticisi
    
    Yetenekler:
    - BERT: Türkçe duygu analizi (GPU hızlandırmalı)
    - Batch processing: Toplu metin işleme
    - Keyword extraction: Anahtar kelime çıkarma
    - Reservation parsing: Rezervasyon bilgisi ayıklama
    - Intent detection: Niyet tespiti
    - Stop words: Türkçe dolgu kelime filtreleme
    
    Transformers kütüphanesi ile Türkçe BERT modelini kullanarak
    yüksek doğrulukta duygu analizi yapar. Erişim seviyesine göre
    bazı işlemler kısıtlanabilir.
    """
    
    # Turkish stop words
    STOP_WORDS = {
        "ve", "ile", "ama", "fakat", "lakin", "ancak", "de", "da",
        "ki", "bu", "şu", "o", "bir", "daha", "en", "çok",
        "mi", "mı", "mu", "mü", "ben", "sen", "biz", "siz",
        "için", "gibi", "kadar", "diye", "yok", "var",
        "ne", "neden", "nasıl", "mıdır"
    }
    
    # Number words
    WORD_TO_NUM = {
        "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5,
        "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10
    }
    
    # Intent keywords
    INTENT_KEYWORDS = {
        Intent.RESERVATION: ["rezervasyon", "masa", "ayır", "yerim", "kişi"],
        Intent.COMPLAINT: ["şikayet", "kötü", "berbat", "memnun değil", "soğuk"],
        Intent.QUESTION: ["ne zaman", "nasıl", "nerede", "kaça", "var mı"],
        Intent.PRAISE: ["harika", "mükemmel", "çok güzel", "teşekkür", "muhteşem"],
        Intent.ORDER: ["sipariş", "istiyorum", "getir", "yemek"],
        Intent.CANCEL: ["iptal", "vazgeç", "istemiyorum"]
    }
    
    # Model settings
    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
    MAX_LENGTH = 512
    CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self, access_level: str = "sandbox"):
        """
        NLP manager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Model
        self.sentiment_pipeline = None
        
        # Metrics
        self.metrics = NLPMetrics()
        
        # Initialize model
        if NLP_AVAILABLE:
            self._init_model()
        else:
            logger.warning("⚠️ NLP modülleri eksik, duygu analizi pasif")
        
        logger.info(f"✅ NLPManager hazır (Erişim: {self.access_level})")
    
    def _init_model(self) -> None:
        """BERT modelini yükle"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=DEVICE_ID
            )
            
            device_name = "GPU (CUDA)" if HAS_GPU else "CPU"
            logger.info(f"✅ NLP modeli {device_name} üzerinde yüklendi")
        
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # TEXT PROCESSING
    # ───────────────────────────────────────────────────────────
    
    def turkish_lower(self, text: str) -> str:
        """
        Türkçe karaktere duyarlı lowercase
        
        Args:
            text: Ham metin
        
        Returns:
            Küçük harfli metin
        """
        if not text:
            return ""
        
        # Turkish character mapping
        map_chars = str.maketrans("IİĞÜŞÖÇ", "ıiğüşöç")
        return text.translate(map_chars).lower()
    
    def clean_text(
        self,
        text: str,
        for_analysis: bool = False
    ) -> str:
        """
        Metni temizle
        
        Args:
            text: Ham metin
            for_analysis: Analiz için daha agresif temizleme
        
        Returns:
            Temiz metin
        """
        if not text:
            return ""
        
        with self.lock:
            # Lowercase
            clean = self.turkish_lower(text)
            
            # Remove filler sounds
            clean = re.sub(
                r'\b(eee|mmm|hmm|şey|yani|ııı|aa|ee|bi)\b',
                '',
                clean
            )
            
            # For analysis: keep only alphanumeric
            if for_analysis:
                clean = re.sub(r'[^\w\s]', '', clean)
            
            # Normalize whitespace
            return " ".join(clean.split())
    
    # ───────────────────────────────────────────────────────────
    # SENTIMENT ANALYSIS (Erişim kontrollü)
    # ───────────────────────────────────────────────────────────
    
    def detect_emotion(self, text: str) -> SentimentResult:
        """
        Duygu analizi - Kısıtlı modda sadece nötr döner.
        
        Args:
            text: Analiz edilecek metin
        
        Returns:
            SentimentResult objesi
        """
        # Kısıtlı modda model kullanma
        if self.access_level == AccessLevel.RESTRICTED:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                text=text
            )
        
        if not self.sentiment_pipeline or not text:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                text=text
            )
        
        with self.lock:
            try:
                # Truncate to max length
                truncated = text[:self.MAX_LENGTH]
                
                # Run model
                result = self.sentiment_pipeline(truncated)[0]
                label = result['label'].upper()
                score = result['score']
                
                # Low confidence -> neutral
                if score < self.CONFIDENCE_THRESHOLD:
                    sentiment = Sentiment.NEUTRAL
                elif "POSITIVE" in label:
                    sentiment = Sentiment.POSITIVE
                elif "NEGATIVE" in label:
                    sentiment = Sentiment.NEGATIVE
                else:
                    sentiment = Sentiment.NEUTRAL
                
                self.metrics.sentiment_analyses += 1
                
                return SentimentResult(
                    sentiment=sentiment,
                    confidence=score,
                    text=text
                )
            
            except Exception as e:
                logger.error(f"Duygu analizi hatası: {e}")
                self.metrics.errors_encountered += 1
                
                return SentimentResult(
                    sentiment=Sentiment.NEUTRAL,
                    confidence=0.0,
                    text=text
                )
    
    def analyze_batch(self, text_list: List[str]) -> BatchAnalysis:
        """
        Toplu duygu analizi - Kısıtlı modda boş sonuç döner.
        
        Args:
            text_list: Metin listesi
        
        Returns:
            BatchAnalysis objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return BatchAnalysis(
                total_count=0,
                positive=0,
                negative=0,
                neutral=0,
                satisfaction_rate=0.0,
                top_complaints=[],
                popular_topics=[],
                device_used="Disabled (Restricted Mode)"
            )
        
        if not text_list or not self.sentiment_pipeline:
            return BatchAnalysis(
                total_count=0,
                positive=0,
                negative=0,
                neutral=0,
                satisfaction_rate=0.0,
                top_complaints=[],
                popular_topics=[],
                device_used="N/A"
            )
        
        with self.lock:
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            all_text = ""
            negative_text = ""
            
            try:
                # Filter empty texts
                valid_texts = [t for t in text_list if t.strip()]
                
                # Batch inference
                model_results = self.sentiment_pipeline(
                    valid_texts,
                    truncation=True
                )
                
                # Process results
                for i, res in enumerate(model_results):
                    label = res['label'].upper()
                    score = res['score']
                    text = valid_texts[i]
                    
                    cleaned = self.clean_text(text, for_analysis=True)
                    all_text += cleaned + " "
                    
                    # Categorize
                    if score < self.CONFIDENCE_THRESHOLD:
                        neutral_count += 1
                    elif "POSITIVE" in label:
                        positive_count += 1
                    elif "NEGATIVE" in label:
                        negative_count += 1
                        negative_text += cleaned + " "
                    else:
                        neutral_count += 1
                
                total = len(valid_texts)
                satisfaction_rate = (
                    (positive_count / total * 100)
                    if total > 0 else 0.0
                )
                
                # Extract keywords
                top_complaints = self.extract_keywords(negative_text, 3)
                popular_topics = self.extract_keywords(all_text, 5)
                
                device_used = "GPU (CUDA)" if HAS_GPU else "CPU"
                
                self.metrics.batch_analyses += 1
                
                return BatchAnalysis(
                    total_count=total,
                    positive=positive_count,
                    negative=negative_count,
                    neutral=neutral_count,
                    satisfaction_rate=satisfaction_rate,
                    top_complaints=top_complaints,
                    popular_topics=popular_topics,
                    device_used=device_used
                )
            
            except Exception as e:
                logger.error(f"Batch analiz hatası: {e}")
                self.metrics.errors_encountered += 1
                
                return BatchAnalysis(
                    total_count=0,
                    positive=0,
                    negative=0,
                    neutral=0,
                    satisfaction_rate=0.0,
                    top_complaints=[],
                    popular_topics=[],
                    device_used="Error"
                )
    
    # ───────────────────────────────────────────────────────────
    # KEYWORD EXTRACTION
    # ───────────────────────────────────────────────────────────
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Anahtar kelime çıkarma - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            text: Metin
            top_n: Maksimum kelime sayısı
        
        Returns:
            Kelime listesi
        """
        if not text:
            return []
        
        # Split and filter
        words = text.split()
        filtered = [
            w for w in words
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        
        # Count and return top N
        keywords = [
            item[0]
            for item in Counter(filtered).most_common(top_n)
        ]
        
        self.metrics.keywords_extracted += len(keywords)
        
        return keywords
    
    # ───────────────────────────────────────────────────────────
    # INTENT DETECTION
    # ───────────────────────────────────────────────────────────
    
    def detect_intent(self, text: str) -> Intent:
        """
        Niyet tespiti - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            text: Kullanıcı metni
        
        Returns:
            Intent enum
        """
        text_lower = self.turkish_lower(text)
        
        # Check each intent's keywords
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        
        return Intent.UNKNOWN
    
    # ───────────────────────────────────────────────────────────
    # BEHAVIOR PROMPT
    # ───────────────────────────────────────────────────────────
    
    def get_behavior_prompt(self, text: str) -> str:
        """
        LLM davranış talimatı - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            text: Kullanıcı metni
        
        Returns:
            Davranış talimatı
        """
        result = self.detect_emotion(text)
        
        if result.sentiment == Sentiment.POSITIVE:
            return (
                "[DAVRANIŞ: Enerjik, minnettar ve samimi ol. "
                "Gülümseyen bir ton kullan.]"
            )
        elif result.sentiment == Sentiment.NEGATIVE:
            return (
                "[DAVRANIŞ: Alttan al, son derece nazik ve profesyonel ol. "
                "Çözüm odaklı ve özür dileyen bir ton kullan.]"
            )
        else:
            return (
                "[DAVRANIŞ: Net, kısa ve ciddi bir profesyonellikle "
                "yanıt ver.]"
            )
    
    # ───────────────────────────────────────────────────────────
    # RESERVATION EXTRACTION (Erişim kontrollü)
    # ───────────────────────────────────────────────────────────
    
    def extract_reservation_details(self, text: str) -> ReservationData:
        """
        Rezervasyon bilgisi ayıklama - Kısıtlı modda bilgi verilmez.
        
        Args:
            text: Rezervasyon metni
        
        Returns:
            ReservationData objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return ReservationData(
                person_count="🔒 Gizli",
                time="🔒 Gizli",
                phone=None,
                summary="Kısıtlı modda rezervasyon bilgisi gösterilmez."
            )
        
        text_lower = self.turkish_lower(text)
        
        with self.lock:
            # Person count
            person_count = self._extract_person_count(text_lower)
            
            # Time
            time = self._extract_time(text_lower)
            
            # Phone
            phone = self._extract_phone(text)
            
            self.metrics.reservations_extracted += 1
            
            return ReservationData(
                person_count=person_count,
                time=time,
                phone=phone,
                summary=f"{person_count} kişi / {time}"
            )
    
    def _extract_person_count(self, text: str) -> str:
        """Kişi sayısı çıkar"""
        # Numeric pattern
        count_match = re.search(
            r'(\d+)\s*(?:kişi|kisilik|tane|masa|pax)',
            text
        )
        
        if count_match:
            return count_match.group(1)
        
        # Word pattern
        for word, val in self.WORD_TO_NUM.items():
            if f"{word} kişi" in text or f"{word} kişilik" in text:
                return str(val)
        
        return "Bilinmiyor"
    
    def _extract_time(self, text: str) -> str:
        """Saat çıkar"""
        # HH:MM pattern
        time_match = re.search(r'(\d{1,2})[:. ](\d{2})', text)
        
        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2))
            
            if h < 24 and m < 60:
                return f"{str(h).zfill(2)}:{str(m).zfill(2)}"
        
        # General time descriptions
        if "akşam" in text:
            for word, val in self.WORD_TO_NUM.items():
                if f"akşam {word}" in text:
                    hour = val + 12 if val < 12 else val
                    return f"{hour}:00"
            return "Akşam (Saat Belirsiz)"
        
        if "öğle" in text:
            return "Öğle (Saat Belirsiz)"
        
        if "yarın" in text:
            return "Yarın"
        
        return "Belirtilmedi"
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Telefon numarası çıkar"""
        # Turkish phone pattern
        phone_pattern = r'(\+?9?0?\s?5\d{2}\s?-?\d{3}\s?-?\d{2}\s?-?\d{2})'
        phone_matches = re.findall(phone_pattern, text)
        
        if not phone_matches:
            return None
        
        raw_num = phone_matches[0]
        clean_num = re.sub(r'\D', '', raw_num)
        
        # Format variations
        if clean_num.startswith("05") and len(clean_num) == 11:
            return f"+90{clean_num[1:]}"
        elif clean_num.startswith("5") and len(clean_num) == 10:
            return f"+90{clean_num}"
        elif clean_num.startswith("905") and len(clean_num) == 12:
            return f"+{clean_num}"
        else:
            return raw_num
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        NLP metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "sentiment_analyses": self.metrics.sentiment_analyses,
            "batch_analyses": self.metrics.batch_analyses,
            "reservations_extracted": self.metrics.reservations_extracted,
            "keywords_extracted": self.metrics.keywords_extracted,
            "errors_encountered": self.metrics.errors_encountered,
            "model_loaded": self.sentiment_pipeline is not None,
            "gpu_available": HAS_GPU,
            "device": "GPU (CUDA)" if HAS_GPU else "CPU",
            "access_level": self.access_level
        }