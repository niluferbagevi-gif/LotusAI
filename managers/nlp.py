import re
import logging
import threading
from collections import Counter
from typing import Dict, List, Any, Optional

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    class Config:
        USE_GPU = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.NLP")

# --- KÜTÜPHANE VE GPU KONTROLÜ ---
NLP_AVAILABLE = False
HAS_GPU = False
DEVICE_ID = -1  # -1: CPU, 0: GPU (CUDA)
DEVICE_NAME = "CPU"

try:
    import torch
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    NLP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ NLP kütüphaneleri eksik: {e}. (pip install torch transformers)")

# Config üzerinden GPU kontrolü
USE_GPU_CONFIG = getattr(Config, "USE_GPU", False)

if NLP_AVAILABLE and USE_GPU_CONFIG:
    try:
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE_ID = 0
            DEVICE_NAME = "GPU (CUDA)"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 NLP Modülü GPU Aktif: {gpu_name}")
            except:
                logger.info("🚀 NLP Modülü GPU Aktif")
        else:
            logger.info("ℹ️ NLP: Config GPU açık ancak donanım bulunamadı. CPU kullanılacak.")
    except Exception as e:
        logger.warning(f"⚠️ NLP GPU kontrol hatası: {e}")
else:
    if NLP_AVAILABLE:
        logger.info("ℹ️ NLP işlemleri CPU modunda (Config ayarı).")


class NLPManager:
    """
    LotusAI Doğal Dil İşleme (NLP) ve Duygu Analizi Yöneticisi.
    
    Yetenekler:
    - GPU Hızlandırma: Transformer modelleri CUDA üzerinden çalıştırılır (Config kontrollü).
    - Derin Öğrenme Tabanlı Duygu Analizi: Türkçe BERT modeli ile yüksek doğruluk.
    - Akıllı Temizleme: Metni gürültüden ve dolgu kelimelerinden arındırır.
    - Veri Ayıklama: Rezervasyon ve iletişim bilgilerini Regex ile çeker.
    """
    
    def __init__(self):
        # Çoklu thread erişimi için kilit
        self.lock = threading.RLock()
        
        self.sentiment_pipeline = None
        
        if NLP_AVAILABLE:
            # Model ve Tokenizer Yükleme (Türkçe Duygu Analizi için BERT tabanlı model)
            try:
                model_name = "savasy/bert-base-turkish-sentiment-cased"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=model_name, 
                    tokenizer=model_name, 
                    device=DEVICE_ID
                )
                logger.info(f"✅ NLP Modeli {DEVICE_NAME} üzerinde yüklendi.")
            except Exception as e:
                logger.error(f"❌ Model yükleme hatası: {e}")
        else:
            logger.warning("⚠️ NLP modülleri eksik olduğu için duygu analizi pasif.")

        # Stop Words (Analiz dışı bırakılacak etkisiz kelimeler)
        self.stop_words = {
            "ve", "ile", "ama", "fakat", "lakin", "ancak", "de", "da", "ki", "bu", "şu", "o",
            "bir", "daha", "en", "çok", "mi", "mı", "mu", "mü", "ben", "sen", "biz", "siz",
            "için", "gibi", "kadar", "diye", "yok", "var", "ne", "neden", "nasıl", "mıdır"
        }
        
        # Sayısal karşılıklar (Rezervasyon ayıklama için)
        self.word_to_num = {
            "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, 
            "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10
        }

    def turkish_lower(self, text: str) -> str:
        """Türkçe karakterlere duyarlı küçük harfe çevirme."""
        if not text: return ""
        map_chars = str.maketrans("IİĞÜŞÖÇ", "ıiğüşöç")
        return text.translate(map_chars).lower()

    def clean_text(self, text: str, for_analysis: bool = False) -> str:
        """Metni gereksiz boşluklardan ve dolgu kelimelerinden temizler."""
        if not text: return ""
        
        with self.lock:
            clean = self.turkish_lower(text)
            # Dolgu seslerini temizle
            clean = re.sub(r'\b(eee|mmm|hmm|şey|yani|ııı|aa|ee|bi)\b', '', clean)
            
            if for_analysis:
                # Sadece harf ve rakamları bırak
                clean = re.sub(r'[^\w\s]', '', clean)
                
            return " ".join(clean.split())

    def detect_emotion(self, text: str) -> str:
        """
        Derin öğrenme modeli kullanarak duygu analizi yapar.
        GPU desteği sayesinde model tahminleri çok hızlı gerçekleşir.
        """
        if not self.sentiment_pipeline or not text:
            return "NOTR"

        with self.lock:
            try:
                # Modeli çalıştır
                result = self.sentiment_pipeline(text[:512])[0] # BERT 512 token sınırı
                label = result['label'].upper() # 'POSITIVE' veya 'NEGATIVE'
                score = result['score']
                
                # Skor eşiği kontrolü (Düşük güvenilirlikte nötr kabul et)
                if score < 0.6:
                    return "NOTR"
                
                # Model etiketlerini LotusAI formatına çevir
                if "POSITIVE" in label: return "POZITIF"
                if "NEGATIVE" in label: return "NEGATIF"
                return "NOTR"
            except Exception as e:
                logger.error(f"Duygu analizi hatası: {e}")
                return "NOTR"

    def analyze_batch(self, text_list: List[str]) -> Dict[str, Any]:
        """
        Yorum listesi üzerinde TOPLU (Batch) analiz yapar.
        GPU üzerinde toplu işleme performansı maksimize eder.
        """
        if not text_list or not self.sentiment_pipeline:
            return {"status": "Veri Yok"}

        with self.lock:
            results = {
                "positive": 0, "negative": 0, "neutral": 0,
                "all_text": "", "neg_text": ""
            }

            # GPU avantajını kullanmak için listeyi toplu olarak modele gönderiyoruz
            try:
                # Boş metinleri filtrele
                valid_texts = [t for t in text_list if t.strip()]
                model_results = self.sentiment_pipeline(valid_texts, truncation=True)
                
                for i, res in enumerate(model_results):
                    label = res['label'].upper()
                    score = res['score']
                    text = valid_texts[i]
                    cleaned = self.clean_text(text, for_analysis=True)
                    results["all_text"] += cleaned + " "
                    
                    if score < 0.6:
                        results["neutral"] += 1
                    elif "POSITIVE" in label:
                        results["positive"] += 1
                    elif "NEGATIVE" in label:
                        results["negative"] += 1
                        results["neg_text"] += cleaned + " "
                    else:
                        results["neutral"] += 1

                total = len(valid_texts)
                sentiment_score = int((results["positive"] / total) * 100) if total > 0 else 0
                
                return {
                    "total_count": total,
                    "device_used": DEVICE_NAME,
                    "sentiment_distribution": {
                        "positive": results["positive"],
                        "negative": results["negative"],
                        "neutral": results["neutral"]
                    },
                    "satisfaction_rate": f"%{sentiment_score}",
                    "top_complaints": self.extract_keywords(results["neg_text"], 3),
                    "popular_topics": self.extract_keywords(results["all_text"], 5)
                }
            except Exception as e:
                logger.error(f"Batch analiz hatası: {e}")
                return {"status": "Hata", "error": str(e)}

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """En sık geçen ve anlam taşıyan anahtar kelimeleri bulur."""
        if not text: return []
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        return [item[0] for item in Counter(filtered).most_common(top_n)]

    def get_behavior_prompt(self, text: str) -> str:
        """Kullanıcının moduna göre Gemini/LLM için davranış talimatı üretir."""
        emotion = self.detect_emotion(text)
        if emotion == "POZITIF": 
            return "[DAVRANIŞ: Enerjik, minnettar ve samimi ol. Gülümseyen bir ton kullan.]"
        elif emotion == "NEGATIF": 
            return "[DAVRANIŞ: Alttan al, son derece nazik ve profesyonel ol. Çözüm odaklı ve özür dileyen bir ton kullan.]"
        return "[DAVRANIŞ: Net, kısa ve ciddi bir profesyonellikle yanıt ver.]"

    def extract_reservation_details(self, text: str) -> Dict[str, Any]:
        """Metinden rezervasyon verilerini (Kişi, Saat, Telefon) ayıklar (Regex)."""
        text = self.turkish_lower(text)
        
        with self.lock:
            # 1. Kişi Sayısı Ayıklama
            count = "Bilinmiyor"
            count_match = re.search(r'(\d+)\s*(?:kişi|kisilik|tane|masa|pax)', text)
            if count_match:
                count = count_match.group(1)
            else:
                for word, val in self.word_to_num.items():
                    if f"{word} kişi" in text or f"{word} kişilik" in text:
                        count = str(val)
                        break
            
            # 2. Saat Ayıklama
            time_val = "Belirtilmedi"
            time_match = re.search(r'(\d{1,2})[:. ](\d{2})', text)
            if time_match:
                h, m = int(time_match.group(1)), int(time_match.group(2))
                if h < 24 and m < 60:
                    time_val = f"{str(h).zfill(2)}:{str(m).zfill(2)}"
            
            if time_val == "Belirtilmedi":
                if "akşam" in text: 
                    time_val = "Akşam (Saat Belirsiz)"
                    for word, val in self.word_to_num.items():
                        if f"akşam {word}" in text:
                            time_val = f"{val + 12}:00" if val < 12 else f"{val}:00"
                elif "öğle" in text: time_val = "Öğle (Saat Belirsiz)"
                elif "yarın" in text: time_val = "Yarın"
            
            # 3. Telefon Numarası Ayıklama
            phone = None
            phone_pattern = r'(\+?9?0?\s?5\d{2}\s?-?\d{3}\s?-?\d{2}\s?-?\d{2})'
            phone_matches = re.findall(phone_pattern, text)
            
            if phone_matches:
                raw_num = phone_matches[0]
                clean_num = re.sub(r'\D', '', raw_num)
                if clean_num.startswith("05") and len(clean_num) == 11:
                    phone = f"+90{clean_num[1:]}"
                elif clean_num.startswith("5") and len(clean_num) == 10:
                    phone = f"+90{clean_num}"
                elif clean_num.startswith("905") and len(clean_num) == 12:
                    phone = f"+{clean_num}"
                else:
                    phone = raw_num 

            return {
                "kisi_sayisi": count,
                "saat": time_val,
                "iletisim": phone,
                "ham_metin_ozeti": f"{count} kişi / {time_val}"
            }