import os
import re
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from config import Config

# GPU desteği için gerekli kütüphane
try:
    import torch
except ImportError:
    torch = None

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Gaya")

class GayaAgent:
    """
    Gaya (Operasyon, Finans ve İletişim Uzmanı) - LotusAI'ın Marka Yüzü.
    
    Yetenekler:
    - GPU Hızlandırmalı NLP: Rezervasyon ve metin analizini donanım hızlandırma ile yapar.
    - Fatura/Gider İşleme: Finansal verileri temizler ve muhasebe/stok sistemine aktarır.
    - Çok Kanallı İletişim: Sosyal medya ve mesajlaşma kanalları için bağlamsal yanıtlar üretir.
    - Donanım Farkındalığı: Sistemin GPU imkanlarını kullanarak ağır işlemleri optimize eder.
    """
    
    def __init__(self, tools_dict: Dict[str, Any], nlp_manager: Any):
        """
        Gaya operasyon modülünü başlatır.
        
        :param tools_dict: Engine tarafından sağlanan yöneticiler (operations, accounting, messaging vb.)
        :param nlp_manager: Rezervasyon verilerini ayıklamak için kullanılan NLP motoru.
        """
        self.tools = tools_dict
        self.nlp = nlp_manager
        self.agent_name = "GAYA"
        self.lock = threading.RLock()
        
        # GPU/Cihaz Tespiti
        self.device = self._detect_device()
        
        # Alt bileşenleri GPU'ya yönlendir (Eğer destekliyorlarsa)
        self._optimize_subsystems()
        
        logger.info(f"🌸 {self.agent_name} Operasyon modülü {self.device} üzerinde aktif.")

    def _detect_device(self) -> str:
        """
        Sistemin kullanabileceği en iyi işlem birimini (GPU/CPU) tespit eder.
        """
        if torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps" # Apple Silicon desteği
        return "cpu"

    def _optimize_subsystems(self):
        """
        Bağlı olan NLP ve diğer araçları tespit edilen GPU cihazına taşımaya çalışır.
        """
        with self.lock:
            if self.nlp and hasattr(self.nlp, 'to'):
                try:
                    self.nlp.to(self.device)
                    logger.info(f"🚀 Gaya NLP Modeli {self.device} birimine taşındı.")
                except Exception as e:
                    logger.warning(f"⚠️ NLP modeli GPU'ya taşınamadı: {e}")

    def get_system_prompt(self) -> str:
        """
        Gaya'nın kişiliğini ve çalışma prensiplerini tanımlayan sistem talimatı.
        """
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin Operasyon ve İletişim Uzmanı GAYA'sın. "
            "Müşterilerle iletişim kurarken son derece nazik, yardımsever, kurumsal ve çözüm odaklısın. "
            "Görevin: Fatura işlemek, rezervasyonları yönetmek ve sosyal medya trafiğini marka kalitesine uygun yönetmektir. "
            "Karakterin: Pratik, güven verici, enerjik ve satış kabiliyeti yüksek bir profesyonel. "
            "Bir müşteriyle konuşuyorsan 'Siz' dilini kullan ve Lotus Bağevi'nin samimi ama profesyonel atmosferini yansıt. "
            "Fatura işlerken bir muhasebeci titizliğinde ol; tutar ve firma bilgilerini asla atlama. "
            f"Şu an {self.device.upper()} donanımı ile yüksek performans modunda çalışıyorsun."
        )

    def get_context_data(self, user_text: str) -> str:
        """
        Mesaj içeriğine göre GPU hızlandırmalı analiz öncesi bağlam oluşturur.
        """
        context_parts = []
        text_lower = user_text.lower()
        
        with self.lock:
            # 1. Kanal Analizi
            social_platforms = ["whatsapp", "instagram", "facebook", "messenger", "dm", "yazdı"]
            if any(p in text_lower for p in social_platforms):
                context_parts.append(
                    "\n📍 KANAL UYARISI: Sosyal medya kanalı aktif. "
                    "Yanıtın kısa, öz ve ilgi çekici (Call-to-Action) içermeli."
                )
                
            # 2. Finansal Bağlam
            if any(k in text_lower for k in ["fatura", "fiş", "dekont", "ödeme", "harcama"]):
                context_parts.append(
                    "\n📝 GÖREV BAĞLAMI: Finansal veri girişi saptandı. "
                    "Verileri titizlikle 'AccountingManager' ve 'OperationsManager' sistemlerine işle."
                )

            # 3. Rezervasyon Bağlamı
            if any(k in text_lower for k in ["masa", "rezervasyon", "yer", "ayırt", "geleceğiz"]):
                context_parts.append(
                    "\n📅 REZERVASYON MODU: Rezervasyon talebi inceleniyor. "
                    "Kişi sayısı, saat ve iletişim bilgilerini doğrulamayı unutma."
                )

        return "\n".join(context_parts)

    def _clean_price(self, raw_price: Any) -> float:
        """
        Metin içerisinden tutar bilgisini güvenli bir şekilde float sayıya çevirir.
        """
        if not raw_price: return 0.0
        if isinstance(raw_price, (int, float)): return float(raw_price)
        
        try:
            clean = str(raw_price).upper().replace("TL", "").replace("TRY", "").replace("₺", "").strip()
            
            if "," in clean and "." in clean:
                if clean.rfind(",") > clean.rfind("."): 
                    clean = clean.replace(".", "").replace(",", ".")
                else: 
                    clean = clean.replace(",", "")
            elif "," in clean:
                clean = clean.replace(",", ".")
            
            clean = re.sub(r'[^0-9.]', '', clean)
            return float(clean) if clean else 0.0
        except Exception as e:
            logger.error(f"Gaya: Fiyat dönüştürme hatası ({raw_price}): {e}")
            return 0.0

    def process_invoice_result(self, invoice_data: Dict[str, Any]) -> str:
        """
        AI (Vision) tarafından analiz edilen verileri GPU farkındalığıyla işler.
        """
        if not invoice_data:
            return "⚠️ Fatura analizi için veri sağlanamadı."

        with self.lock:
            firma = invoice_data.get('firma', 'Bilinmeyen Tedarikçi')
            raw_tutar = invoice_data.get("toplam_tutar", "0")
            tutar = self._clean_price(raw_tutar)
            
            results = []
            
            # 1. Adım: Stok Güncelleme
            urunler = invoice_data.get("urunler", [])
            if urunler and 'operations' in self.tools:
                try:
                    stock_res = self.tools['operations'].process_invoice_items(urunler)
                    results.append(f"📦 {stock_res}")
                except Exception as e:
                    logger.error(f"Gaya: Stok işleme hatası: {e}")
                    results.append("❌ Stoklar güncellenirken hata oluştu.")
            
            # 2. Adım: Muhasebe/Finans Kaydı
            acc_tool = self.tools.get('accounting') or self.tools.get('finance')
            if acc_tool and hasattr(acc_tool, 'add_entry'):
                try:
                    if tutar > 0:
                        acc_tool.add_entry(
                            tur="GIDER", 
                            aciklama=f"{firma} Faturası Girişi (Sistem: Gaya)", 
                            tutar=tutar,
                            kategori="Mutfak/Operasyon",
                            user_id="GAYA"
                        )
                        results.append(f"💰 Muhasebe: -{tutar:,.2f} TL gider kaydı oluşturuldu.")
                    else:
                        results.append("⚠️ Tutar belirlenemediği için finansal kayıt atlandı.")
                except Exception as e:
                    logger.error(f"Gaya: Muhasebe kayıt hatası: {e}")
                    results.append("❌ Finansal kayıt oluşturulamadı.")

            report = [
                f"🧾 FATURA İŞLEME ÖZETİ ({firma})",
                f"{'='*35}",
                "\n".join(results),
                f"{'='*35}",
                f"Donanım: {self.device.upper()} | İşlem başarıyla tamamlandı."
            ]
            return "\n".join(report)

    def handle_reservation(self, user_text: str, user_name: str) -> Optional[str]:
        """
        Rezervasyon talebini NLP (GPU Destekli) ile ayrıştırıp sisteme kaydeder.
        """
        if not self.nlp: return None
        
        with self.lock:
            try:
                # NLP Manager artık GPU üzerinde çalışıyor olabilir
                details = self.nlp.extract_reservation_details(user_text)
                
                # Minimum veri kontrolü
                if details.get("kisi_sayisi") != "Bilinmiyor" or details.get("saat") != "Belirtilmedi":
                    if 'operations' in self.tools:
                        msg_tool = self.tools.get('messaging') or self.tools.get('media')
                        
                        result = self.tools['operations'].add_reservation(
                            name=user_name,
                            time_slot=details.get("saat"),
                            count=details.get("kisi_sayisi"),
                            phone=details.get("iletisim"),
                            messenger=msg_tool
                        )
                        return result
                    else:
                        return "⚠️ Operasyon yöneticisi aktif değil."
                
                return None
                
            except Exception as e:
                logger.error(f"Gaya: Rezervasyon yönetimi hatası: {e}")
                return "❌ Rezervasyon işlemi sırasında bir teknik aksaklık yaşandı."

    def get_social_content_idea(self) -> str:
        """
        Gaya'nın MediaManager trendlerine göre içerik planı üretmesi.
        """
        if 'media' in self.tools:
            try:
                daily_context = self.tools['media'].get_daily_context()
                return f"🌸 Gaya'nın bugünkü paylaşım önerisi:\n{daily_context}"
            except: pass
        return "Bugün için henüz bir içerik planı oluşturulmadı."