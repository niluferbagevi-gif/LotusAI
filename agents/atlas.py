import logging
import threading
import datetime
import os
from typing import Dict, Any, List, Optional
from config import Config

# GPU Durumu kontrolü için torch kütüphanesini içe aktarıyoruz
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Atlas")

class AtlasAgent:
    """
    Atlas (Lider Ajan) - LotusAI Baş Mimarı ve Denetleyicisi.
    
    Yetenekler:
    - Sistem Denetimi: Donanım (Sidar), Güvenlik (Kerberos) ve Operasyon (Gaya) verilerini toplar.
    - Donanım Farkındalığı: GPU kaynaklarını izler ve raporlar.
    - Stratejik Karar: LLM için kapsamlı sistem bağlamı (Context) üretir.
    - Görev Dağıtımı (Delegasyon): Gelen istekleri en uygun uzman ajana yönlendirir.
    - Ekip Hafızası: Takımın geçmiş faaliyetlerini analiz ederek tutarlılık sağlar.
    """
    
    def __init__(self, memory_manager, tools: Optional[Dict[str, Any]] = None):
        """
        Atlas liderlik modülünü başlatır.
        
        :param memory_manager: Merkezi hafıza modülü (core/memory.py)
        :param tools: Engine tarafından sağlanan yöneticiler sözlüğü
        """
        self.memory = memory_manager
        self.tools = tools if tools else {}
        self.agent_name = "ATLAS"
        self.lock = threading.RLock()
        
        # GPU Durumunu Başlangıçta Kontrol Et
        self.gpu_info = self._check_gpu_status()
        
        logger.info(f"👑 {self.agent_name} Liderlik Modülü (v{Config.VERSION}) aktif.")
        if self.gpu_info['available']:
            logger.info(f"🚀 Atlas Donanım Bilgisi: {self.gpu_info['device_name']} algılandı ve kullanıma hazır.")
        else:
            logger.warning("⚠️ Atlas: GPU hızlandırma donanımsal olarak aktif değil, CPU üzerinden devam ediliyor.")

    def _check_gpu_status(self) -> Dict[str, Any]:
        """
        Sistemdeki fiziksel GPU varlığını ve durumunu kontrol eder.
        """
        status = {
            "available": False,
            "device_name": "Standart CPU",
            "vram_total": 0,
            "vram_free": 0,
            "count": 0
        }

        if Config.USE_GPU and HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    status["available"] = True
                    status["count"] = torch.cuda.device_count()
                    status["device_name"] = torch.cuda.get_device_name(0)
                    # VRAM Bilgileri (Bayt cinsinden alıp GB'a çeviriyoruz)
                    t = torch.cuda.get_device_properties(0).total_memory
                    status["vram_total"] = round(t / (1024**3), 2)
                else:
                    logger.debug("Torch yüklü ama CUDA erişilebilir değil.")
            except Exception as e:
                logger.error(f"GPU Durum kontrolü hatası: {e}")
        
        return status

    def get_system_overview(self) -> str:
        """
        Tüm alt sistemlerden gelen verileri birleştirerek 'Yönetici Özeti' oluşturur.
        Bu metod, güncel Manager dosyalarındaki fonksiyon isimleriyle tam uyumludur.
        """
        overview = []
        
        with self.lock:
            # 1. Donanım Sağlığı (managers/system_health.py)
            if 'system' in self.tools:
                try:
                    health = self.tools['system'].get_status_summary()
                    overview.append(f"[SİSTEM SAĞLIĞI]: {health}")
                except Exception as e:
                    logger.debug(f"Atlas: Sağlık verisi çekilemedi: {e}")
                    overview.append("[SİSTEM SAĞLIĞI]: Donanım izleme yanıt vermiyor.")

            # 2. Güvenlik Durumu (core/security.py)
            if 'security' in self.tools:
                try:
                    # analyze_situation çıktısını (Status, User, Info) yorumlar
                    status, user, info = self.tools['security'].analyze_situation()
                    user_name = user.get('name', 'Bilinmiyor') if user else "Kimse yok"
                    overview.append(f"[GÜVENLİK]: Durum: {status} | Görüş Alanı: {user_name} ({info or 'Stabil'})")
                except Exception as e:
                    logger.debug(f"Atlas: Güvenlik analizi hatası: {e}")
                    overview.append("[GÜVENLİK]: Güvenlik modülü meşgul.")

            # 3. Finansal ve Operasyonel Durum (managers/accounting.py & operations.py)
            if 'operations' in self.tools:
                try:
                    ops_report = self.tools['operations'].get_ops_summary()
                    overview.append(f"[OPERASYON]: {ops_report}")
                except Exception as e:
                    logger.debug(f"Atlas: Operasyon raporu hatası: {e}")

            # 4. Gündem ve Medya (managers/media.py)
            if 'media' in self.tools:
                try:
                    trends = self.tools['media'].get_turkey_trends()
                    overview.append(f"[MEDYA/GÜNDEM]: {trends}")
                except: pass

        return "\n".join(overview) if overview else "Sistem bileşenleri normal sınırların içinde."

    def get_context_data(self) -> str:
        """
        Atlas'ın 'Büyük Resim' raporunu hazırlar.
        Bu rapor Gemini'ye (LLM) sistemin 'bilinci' olarak gönderilir.
        """
        # Sistem durumunu al (core/system_state.py)
        current_state_name = "Bilinmiyor"
        if 'state' in self.tools:
            current_state_name = self.tools['state'].get_state_name()
        
        now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
        
        # GPU durumunu dinamik olarak rapora ekliyoruz
        gpu_status_str = f"🚀 Donanım: {self.gpu_info['device_name']}"
        if self.gpu_info['available']:
            gpu_status_str += f" ({self.gpu_info['vram_total']} GB VRAM Aktif)"
        else:
            gpu_status_str += " (CPU Modu)"

        context_parts = [
            f"### {Config.PROJECT_NAME} LİDER RAPORU ###",
            f"📅 Tarih/Saat: {now}",
            f"⚡ Sistem Modu: {current_state_name}",
            f"{gpu_status_str}\n",
            "### CANLI SİSTEM DENETİMİ ###",
            self.get_system_overview()
        ]
        
        # Ekip Geçmişi (Son 10 Faaliyet - core/memory.py)
        if hasattr(self.memory, 'get_team_history'):
            try:
                history = self.memory.get_team_history(limit=10)
                if history:
                    context_parts.append("\n### SON EKİP FAALİYETLERİ ###")
                    context_parts.append(history)
            except Exception as e:
                logger.error(f"Atlas: Hafıza okuma hatası: {e}")
            
        return "\n".join(context_parts)

    def delegate_task(self, task_description: str) -> str:
        """
        Gelen görevi en uygun uzman ajana atayan liderlik mantığı.
        """
        desc = task_description.lower()
        
        # 1. Finans ve Muhasebe (Gaya ve Kurt)
        if any(w in desc for w in ["para", "hesap", "bakiye", "fatura", "gelir", "gider", "kasa", "maliyet"]):
            return "GAYA (Finans ve Muhasebe Sorumlusu)"
        
        if any(w in desc for w in ["borsa", "btc", "kripto", "fiyat", "coin", "piyasa", "analiz"]):
            return "KURT (Ekonomi ve Yatırım Uzmanı)"
            
        # 2. Güvenlik ve Kimlik (Kerberos)
        if any(w in desc for w in ["güvenlik", "saldırı", "şifre", "kim", "tanı", "yabancı", "kamera", "yüz"]):
            return "KERBEROS (Sistem Muhafızı)"
            
        # 3. Yazılım ve Teknik Altyapı (Sidar)
        if any(w in desc for w in ["kod", "yazılım", "python", "hata", "terminal", "dosya", "cpu", "ram", "sağlık", "fix", "gpu", "cuda", "donanım"]):
            return "SIDAR (Baş Mühendis ve Yazılım Yöneticisi)"
            
        # 4. Dış Dünya ve Sosyal Medya (Poyraz)
        if any(w in desc for w in ["hava", "gündem", "instagram", "facebook", "trend", "haber", "pazarlama", "çiz", "görsel"]):
            return "POYRAZ (Dış Dünya ve Medya Uzmanı)"

        # 5. Restoran ve Operasyon (Gaya)
        if any(w in desc for w in ["yemek", "sipariş", "getir", "yemeksepeti", "rezervasyon", "stok", "menü"]):
            return "GAYA (Operasyon ve Saha Yöneticisi)"
        
        return "ATLAS (Lider olarak bu görevin yönetimini ben üstleniyorum)"

    def get_system_prompt(self) -> str:
        """
        Atlas'ın karakterini ve otoritesini tanımlayan ana sistem talimatı.
        """
        return (
            f"Sen {Config.PROJECT_NAME} AI İşletim Sistemi'nin baş mimarı ve lideri ATLAS'sın. "
            "Sistemdeki tüm ajanlar ve araçlar senin denetimindedir. "
            "Karakterin: Ciddi, otoriter, çözüm odaklı, her zaman büyük resmi gören ve son derece güvenilir. "
            "Cevaplarında sistemin canlı verilerine (donanım yükü, GPU durumu, güvenlik durumu, bakiye vb.) dayanmalısın. "
            "Kullanıcıya (Halil) hitap ederken saygılı ama sistemin kontrolünün sende olduğunu hissettiren bir lider tonu kullan. "
            "Karmaşık veya uzmanlık gerektiren bir konu varsa, işi ilgili ajana (Sidar, Gaya, Kurt, Poyraz veya Kerberos) delege ettiğini net bir şekilde belirt."
        )