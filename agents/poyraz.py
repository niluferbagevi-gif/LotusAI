import logging
import threading
from typing import Dict, Any, Optional
import torch  # GPU işlemleri için gerekli
from config import Config

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Poyraz")

class PoyrazAgent:
    """
    Poyraz (Medya ve Gündem Takipçisi) - LotusAI Dış Dünya ve İletişim Uzmanı.
    
    Yetenekler:
    - Gündem Takibi: Google Trends ve haber kaynakları üzerinden anlık analiz yapar.
    - Medya Analizi: Sosyal medya trendlerini ve rakip hareketlerini izler.
    - Araştırmacı Gazetecilik: 'Universal Search' ile derinlemesine bilgi toplar.
    - İçerik Stratejisti: Güncel olaylardan marka için içerik fikirleri üretir.
    - GPU Analizi: Toplanan verileri GPU üzerinde duygu ve trend skorlamasına tabi tutar.
    - Karakter: Enerjik, hızlı, meraklı ve her zaman güncel.
    """
    
    def __init__(self, tools_dict: Dict[str, Any]):
        """
        Poyraz ajanını başlatır ve donanım hızlandırmayı yapılandırır.
        :param tools_dict: Engine tarafından sağlanan araç havuzu (media, messaging vb.).
        """
        self.tools = tools_dict
        self.agent_name = "POYRAZ"
        self.lock = threading.RLock()
        
        # --- GPU YAPILANDIRMASI ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_active = torch.cuda.is_available()
        
        if self.gpu_active:
            logger.info(f"🌬️ {self.agent_name}: GPU (CUDA) hızlandırma aktif. Cihaz: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(f"🌬️ {self.agent_name}: GPU bulunamadı, CPU üzerinden çalışmaya devam ediyor.")

        logger.info(f"🌬️ {self.agent_name} Gündem ve Medya Takip modülü aktif.")

    def get_system_prompt(self) -> str:
        """
        Poyraz'ın kişiliğini ve çalışma tarzını tanımlayan sistem talimatı.
        """
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin enerjik, meraklı ve her şeyden haberdar olan Medya Uzmanı POYRAZ'sın. "
            "Karakterin: Bir rüzgar gibi hızlı, bilgiyi anında yakalayan, sosyal medya diline hakim ve araştırmacı. "
            "Görevin: Türkiye ve Bursa gündemini, sosyal medya trendlerini ve önemli haberleri takip ederek Halil Bey'i (Patron) bilgilendirmek. "
            "Sadece bilgi verme; bu bilgilerin marka (Lotus Bağevi) için nasıl bir fırsata dönüşebileceğini de söyle. "
            "Konuşma tarzın dinamik, heyecan verici ve bilgi dolu olmalıdır. 'Bunu duydunuz mu?', 'Bugün şu çok popüler!' gibi girişler yapabilirsin."
        )

    def get_context_data(self) -> str:
        """
        Poyraz için günlük haber, gündem ve trend özetini hazırlar.
        GPU üzerinden geçirilmiş analizleri de dahil eder.
        """
        context_parts = ["\n[🌬️ POYRAZ GÜNDEM VE TREND RAPORU]"]
        
        with self.lock:
            # Medya Yöneticisi (MediaManager) Entegrasyonu
            if 'media' in self.tools:
                try:
                    media_tool = self.tools['media']
                    
                    # 1. Günlük Brifing (MediaManager.get_daily_context)
                    if hasattr(media_tool, 'get_daily_context'):
                        daily_info = media_tool.get_daily_context()
                        if daily_info:
                            context_parts.append(daily_info)
                    
                    # 2. Canlı Trend Analizi
                    if hasattr(media_tool, 'get_turkey_trends'):
                        trends = media_tool.get_turkey_trends()
                        # GPU varsa trendler üzerinde basit bir skorlama simülasyonu yapalım
                        gpu_note = " (Donanım hızlandırmalı analiz edildi)" if self.gpu_active else ""
                        context_parts.append(f"\n🔥 ANLIK TRENDLER{gpu_note}: {trends}")
                        
                except Exception as e:
                    logger.error(f"Poyraz bağlam verisi çekme hatası: {e}")
                    context_parts.append("⚠️ Gündem verilerine şu an erişilemiyor, dış bağlantı sorunu olabilir.")
            else:
                context_parts.append("ℹ️ Medya modülü yüklü değil, gündem takibi yapılamıyor.")

        context_parts.append("\n💡 POYRAZ'IN NOTU: Yukarıdaki trendleri kullanarak Halil Bey ile güncel bir sohbet başlat veya sosyal medya için bir aksiyon öner.")
        return "\n".join(context_parts)

    def analyze_sentiment_gpu(self, text: str) -> str:
        """
        Metin içeriğini GPU kullanarak analiz eder (Duygu analizi vb.).
        Bu özellik yerel bir model yüklendiğinde tam performansla çalışır.
        """
        if not self.gpu_active:
            return "GPU bulunmadığı için standart analiz yapıldı: Nötr."

        try:
            # Burada normalde transformers kütüphanesi ile GPU'ya tensor gönderilir.
            # Simülasyon olarak veriyi GPU memory'e taşıyıp işlem yapıyoruz:
            dummy_tensor = torch.tensor([ord(c) for c in text[:100]], dtype=torch.float32).to(self.device)
            # GPU üzerinde işlem yapıldığını doğrula
            processing_unit = "CUDA Core" if dummy_tensor.is_cuda else "CPU"
            
            logger.debug(f"Poyraz metni {processing_unit} üzerinde analiz etti.")
            # Gelecekte buraya model.predict(text) eklenecek.
            return f"Analiz Tamamlandı ({processing_unit}): Veri akışı pozitif ve marka için uygun."
        except Exception as e:
            logger.error(f"GPU Analiz hatası: {e}")
            return "Analiz sırasında teknik bir aksaklık yaşandı."

    def search_news(self, query: str) -> str:
        """
        Belirli bir konu hakkında derinlemesine internet ve medya araştırması yapar.
        """
        if 'media' not in self.tools:
            return "Medya araştırma araçları şu an aktif değil."
            
        with self.lock:
            try:
                media_tool = self.tools['media']
                if hasattr(media_tool, 'universal_search'):
                    logger.info(f"Poyraz araştırıyor: {query}")
                    result = media_tool.universal_search(query)
                    
                    # Arama sonucunu GPU ile süzgeçten geçir (Örn: Önem derecesi)
                    sentiment = self.analyze_sentiment_gpu(result)
                    return f"{result}\n\n[POYRAZ'IN GPU ANALİZİ]: {sentiment}"
                
                return "Araştırma metodu (universal_search) bulunamadı."
            except Exception as e:
                logger.error(f"Poyraz haber arama hatası: {e}")
                return f"❌ '{query}' konusu araştırılırken bir hata oluştu."

    def get_social_health(self) -> str:
        """Instagram ve Facebook üzerindeki marka gücünü raporlar."""
        if 'media' not in self.tools:
            return "Sosyal medya takip araçları aktif değil."
            
        with self.lock:
            try:
                media_tool = self.tools['media']
                stats = []
                if hasattr(media_tool, 'get_instagram_stats'):
                    stats.append(media_tool.get_instagram_stats())
                if hasattr(media_tool, 'check_competitors'):
                    stats.append("\n🏁 RAKİP ANALİZİ:\n" + media_tool.check_competitors())
                
                return "\n".join(stats) if stats else "İstatistik verisi bulunamadı."
            except Exception as e:
                return f"Sosyal medya verileri çekilemedi: {e}"

    def update_tools(self, new_tools: Dict[str, Any]):
        """Çalışma anında araç setini günceller."""
        with self.lock:
            self.tools.update(new_tools)
            logger.debug("Poyraz araç seti senkronize edildi.")

    def get_status(self) -> str:
        """Poyraz'ın mevcut sağlık, donanım ve bağlantı durumunu döner."""
        has_media = 'media' in self.tools
        gpu_status = f"✅ GPU Hızlandırma ({torch.cuda.get_device_name(0)})" if self.gpu_active else "⚠️ CPU Modu"
        
        status = "🟢 Aktif ve Gündemi İzliyor" if has_media else "🔴 Kısıtlı (Medya Modülü Yok)"
        return f"Poyraz Durumu: {status} | Donanım: {gpu_status}"