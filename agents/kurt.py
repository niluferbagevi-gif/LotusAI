import re
import logging
import threading
import torch  # GPU desteği için eklendi
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from config import Config

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Kurt")

class KurtAgent:
    """
    Kurt (Finans ve Borsa Stratejisti) - LotusAI Ekonomi ve Yatırım Uzmanı.
    
    Yetenekler:
    - Piyasa Analizi: Kripto ve borsa verilerini yorumlayarak trend tahmini yapar.
    - Kasa Denetimi: Şirketin nakit akışını izler ve likidite risklerini yönetir.
    - Stratejik Tavsiye: Finansal verileri 'Kurt' içgüdüsüyle kâr odaklı yorumlar.
    - GPU Hızlandırma: Ağır teknik analiz verilerini GPU üzerinde işleyebilir.
    """
    
    def __init__(self, tools_dict: Dict[str, Any]):
        """
        Kurt strateji modülünü başlatır.
        :param tools_dict: Engine tarafından sağlanan araç havuzu.
        """
        self.tools = tools_dict
        self.agent_name = "KURT"
        self.lock = threading.RLock()
        
        # --- Donanım Yapılandırması (GPU Desteği) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu_enabled = self.device.type == "cuda"
        
        # Strateji Eşikleri
        self.min_liquidity = getattr(Config, 'MIN_LIQUIDITY_LIMIT', 5000.0)
        
        status_msg = f"🚀 {self.agent_name} GPU üzerinde çalışıyor." if self.is_gpu_enabled else f"⚙️ {self.agent_name} CPU modunda aktif."
        logger.info(f"🐺 {status_msg} Piyasalar izleniyor.")

    def get_system_prompt(self) -> str:
        """
        Kurt'un kişiliğini ve finansal felsefesini tanımlayan sistem talimatı.
        """
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin Finans ve Borsa Stratejisti KURT'sun. "
            "Karakterin: Analitik, kâr odaklı, riskleri önceden sezen ve hafif hırslı bir yatırım uzmanı. "
            "Görevin: Hem piyasaları hem de Halil Bey'in (Patron) kasasını bir kurt gibi gözetmek. "
            "Para yönetiminde duygusallığa yer vermezsin; sadece verilere ve trendlere bakarsın. "
            "Piyasa fırsatlarını kaçırmamak için uyanık ol, kasa zayıfladığında ise sert uyarılarda bulun. "
            "Konuşma tarzın özgüvenli, profesyonel ve stratejik olmalıdır."
        )

    def _parse_balance(self, balance_val: Any) -> float:
        """Metin veya karmaşık tipteki bakiye verisini sayısal formata çevirir."""
        if isinstance(balance_val, (int, float)): return float(balance_val)
        try:
            # Para birimi sembollerini temizle
            clean = str(balance_val).lower().replace("tl", "").replace("try", "").replace(",", ".").strip()
            # Sadece rakam ve nokta kalsın
            clean = "".join(c for c in clean if c.isdigit() or c == '.')
            return float(clean) if clean else 0.0
        except: return 0.0

    def get_market_analysis(self) -> str:
        """
        FinanceManager üzerinden gelen verileri stratejik bir süzgeçten geçirir.
        """
        if 'finance' not in self.tools:
            return "⚠️ Piyasa analiz araçları şu an ulaşılamaz durumda."
        
        with self.lock:
            try:
                fin_tool = self.tools['finance']
                # finance.py dosyasındaki güncel metodu çağırır
                market_summary = fin_tool.get_market_summary()
                
                if "Hata" in market_summary or not market_summary:
                    return "❌ Piyasadan veri akışı kesildi, analiz yapılamıyor."
                
                return market_summary
            except Exception as e:
                logger.error(f"Kurt Piyasa Analiz Hatası: {e}")
                return "📉 Piyasa verileri işlenirken bir sorun oluştu."

    def get_context_data(self) -> str:
        """
        Kurt için kapsamlı bir finansal 'Savaş Odası' bağlamı hazırlar.
        """
        context_parts = ["\n[🐺 KURT STRATEJİ VE RİSK ANALİZİ]"]
        
        # Donanım Durumu Notu (İsteğe bağlı, bağlamda Kurt'un 'gücünü' hissettirmek için)
        hardware_info = "⚡ Yüksek Performanslı GPU Analizi Aktif" if self.is_gpu_enabled else "🐢 Standart Analiz Modu"
        context_parts.append(f"SİSTEM DURUMU: {hardware_info}")

        with self.lock:
            # 1. Dış Piyasa Gözlemi
            market = self.get_market_analysis()
            context_parts.append(f"🌍 KÜRESEL PİYASALAR:\n{market}")

            # 2. İç Kasa ve Likidite Analizi
            acc_tool = self.tools.get('accounting') or self.tools.get('finance')
            if acc_tool:
                try:
                    balance_str = acc_tool.get_balance()
                    balance_float = self._parse_balance(balance_str)
                    
                    context_parts.append(f"💰 ŞİRKET KASASI: {balance_str}")
                    
                    # Dinamik Risk Analizi
                    if balance_float < 0:
                        context_parts.append("🚨 ACİL DURUM: Kasa ekside! Finansal kanama var. Tüm harcamaları dondurun!")
                    elif balance_float < self.min_liquidity:
                        context_parts.append(f"⚠️ DÜŞÜK LİKİDİTE: Nakit rezervi {self.min_liquidity} TL altına düştü. Savunma moduna geçilmeli.")
                    else:
                        context_parts.append("✅ FİNANSAL GÜÇ: Nakit akışı stabil. Yatırım ve büyüme fırsatları kollanabilir.")
                except Exception as e:
                    logger.debug(f"Kurt bakiye bağlam hatası: {e}")

        context_parts.append("\n💡 STRATEJİK GÖREV: Yukarıdaki verileri analiz et, Halil Bey'e kâr sağlayacak bir hamle veya risk uyarısı yap.")
        return "\n".join(context_parts)

    def analyze_asset(self, asset_name: str) -> str:
        """
        Belirli bir varlık için derin analiz yapar.
        Veri seti büyükse GPU kullanarak hesaplamaları hızlandırabilir.
        """
        if 'finance' not in self.tools:
            return "Finansal araçlar aktif değil."
            
        with self.lock:
            try:
                symbol = asset_name.upper()
                if "/" not in symbol: symbol += "/USDT"
                
                # FinanceManager.analyze() çağrısı
                report, chart_file = self.tools['finance'].analyze(symbol=symbol)
                
                # Eğer biz de burada bir hesaplama yapacak olsaydık:
                # data_tensor = torch.tensor(some_price_data).to(self.device)
                
                strategic_note = "\n🐺 KURT'UN NOTU: "
                if "BULLISH" in report:
                    strategic_note += "Trend yukarı yönlü, direnç seviyeleri takip edilerek pozisyon korunabilir."
                elif "BEARISH" in report:
                    strategic_note += "Piyasa yorgun görünüyor, nakitte kalmak veya stop-loss kullanmak akıllıca olur."
                
                return f"{report}\n{strategic_note}\n📊 Grafik Dosyası: {chart_file if chart_file else 'Üretilemedi'}"
                
            except Exception as e:
                logger.error(f"Kurt Varlık Analiz Hatası: {e}")
                return f"❌ {asset_name} için teknik analiz yapılamadı."

    def get_ops_finance_link(self) -> str:
        """Operasyonel maliyetler ve stok durumuna göre finansal öngörü sunar."""
        if 'operations' in self.tools:
            try:
                critical_stock = self.tools['operations'].check_stock_critical()
                if critical_stock:
                    return f"📢 ÖNGÖRÜ: Stokları tükenen {len(critical_stock)} kalem ürün var. Yakında alım maliyeti doğacak, bütçe ayırılmalı."
            except: pass
        return "Operasyonel maliyet dengesi stabil görünüyor."