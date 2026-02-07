import aiohttp
import asyncio
import io
import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from PIL import Image
import google.generativeai as genai

# GPU Desteği için PyTorch Kontrolü
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- LOGLAMA YAPILANDIRMASI ---
logger = logging.getLogger("LotusAI.Engine")

# OpenCV Kontrolü (Görsel işlemler ve kamera için)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ 'opencv-python' yüklü değil, kamera fonksiyonları kısıtlı çalışacaktır.")

# --- KONFİGÜRASYON VE MODÜLLER ---
from agents.definitions import AGENTS_CONFIG
from config import Config
from managers.nlp import NLPManager 

# --- AJANLARIN DİNAMİK VE GÜVENLİ YÜKLENMESİ ---
AGENTS_LOADED = {
    "ATLAS": False, "GAYA": False, "POYRAZ": False, 
    "KURT": False, "SIDAR": False, "KERBEROS": False
}

def _import_agent(name: str, class_name: str):
    """Ajan modüllerini güvenli bir şekilde içe aktarır."""
    try:
        module = __import__(f"agents.{name.lower()}", fromlist=[class_name])
        agent_class = getattr(module, class_name)
        AGENTS_LOADED[name] = True
        return agent_class
    except (ImportError, AttributeError) as e:
        logger.error(f"❌ {name} Ajanı yüklenemedi: {e}")
        return None

# Ajan sınıflarını yükle
AtlasAgent = _import_agent("ATLAS", "AtlasAgent")
GayaAgent = _import_agent("GAYA", "GayaAgent")
PoyrazAgent = _import_agent("POYRAZ", "PoyrazAgent")
KurtAgent = _import_agent("KURT", "KurtAgent")
SidarAgent = _import_agent("SIDAR", "SidarAgent")
KerberosAgent = _import_agent("KERBEROS", "KerberosAgent")


class AgentEngine:
    """
    LotusAI Karar ve Cevap Üretim Motoru.
    GPU Desteği ile ajanlar arası koordinasyonu, görsel analizi ve LLM iletişimini yönetir.
    """
    def __init__(self, memory_manager, tools_dict: Dict[str, Any]):
        self.memory = memory_manager
        self.tools = tools_dict
        self.app_id = getattr(Config, "APP_ID", "lotus-ai-core")
        
        # --- DONANIM (GPU) TESPİTİ ---
        self.device = "cpu"
        if HAS_TORCH:
            if torch.cuda.is_available():
                self.device = "cuda"
                # GPU Önbelleğini temizle
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 GPU Algılandı: {gpu_name}. LotusAI Engine GPU üzerinde hızlandırılıyor.")
            else:
                logger.info("ℹ️ GPU bulunamadı, işlemler CPU üzerinde yürütülecek.")
        
        # NLP Yöneticisi (GPU cihaz bilgisi ile başlatılıyor)
        self.nlp = tools_dict.get('nlp') or NLPManager(device=self.device)
        
        # --- AJANLARI BAŞLAT (Cihaz bilgisi aktarılıyor) ---
        self.atlas = AtlasAgent(memory_manager, tools_dict) if AGENTS_LOADED["ATLAS"] else None
        self.gaya = GayaAgent(tools_dict, self.nlp) if AGENTS_LOADED["GAYA"] else None
        
        # Poyraz (Müzik/Medya) Özel Durumu
        if "poyraz_special" in tools_dict:
            self.poyraz = tools_dict["poyraz_special"]
        else:
            self.poyraz = PoyrazAgent(self.nlp, tools_dict) if AGENTS_LOADED["POYRAZ"] else None
            
        self.kurt = KurtAgent(tools_dict) if AGENTS_LOADED["KURT"] else None
        
        # Sidar (Sistem/Kod) Özel Durumu
        if "sidar_special" in tools_dict:
            self.sidar = tools_dict["sidar_special"]
        else:
            sidar_tools = {k: tools_dict.get(k) for k in ['code', 'system', 'security']}
            self.sidar = SidarAgent(sidar_tools) if AGENTS_LOADED["SIDAR"] else None

        self.kerberos = KerberosAgent(tools_dict) if AGENTS_LOADED["KERBEROS"] else None

    def determine_agent(self, text: str) -> Optional[str]:
        """Kullanıcı girdisindeki anahtar kelimelere göre en uygun ajanı seçer."""
        if not text: return "ATLAS"
        clean_text = self.nlp.clean_text(text.lower())
        
        # Öncelikli kontrol: Güvenlik ve Sistem
        if any(k in clean_text for k in ["sistem", "kod", "hata", "terminal"]):
            return "SIDAR"
        if any(k in clean_text for k in ["kimsin", "yetki", "güvenlik", "kilit"]):
            return "KERBEROS"
        
        # Genel kontrol
        for name, data in AGENTS_CONFIG.items():
            triggers = data.get("wake_words", []) + data.get("keys", [])
            if any(k.lower() in clean_text for k in triggers):
                return name
        
        return "ATLAS"

    def _read_user_bio(self, user_obj: Optional[Dict] = None) -> str:
        """Kullanıcı biyografisini okur (Kişiselleştirilmiş yanıtlar için)."""
        bio_file = user_obj.get("bio_file") if user_obj else "halil_bio.txt"
        work_dir = getattr(Config, "WORK_DIR", os.getcwd())
        bio_path = Path(work_dir) / bio_file
        
        # Fallback to default
        if not bio_path.exists():
            bio_path = Path(work_dir) / "halil_bio.txt"

        if bio_path.exists():
            try:
                return bio_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Biyografi okuma hatası: {e}")
        return ""

    def _load_file_for_gemini(self, file_path: Union[str, Path]):
        """Dosyayı Gemini API'nin anlayacağı multimodal formatına hazırlar."""
        path = Path(file_path)
        if not path.exists(): 
            return None, "Dosya bulunamadı."
        
        ext = path.suffix.lower()
        mime_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".pdf": "application/pdf", ".txt": "text/plain"
        }
        mime_type = mime_types.get(ext, "application/octet-stream")
        
        try:
            return {"mime_type": mime_type, "data": path.read_bytes()}, None
        except Exception as e: 
            return None, str(e)

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Yapay zeka çıktısı içindeki JSON bloklarını daha güvenli ayıklar."""
        try:
            # 1. Kod bloğu içindeki JSON'u ara
            json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            content = json_block.group(1) if json_block else text
            
            # 2. JSON karakterlerini temizle ve bul
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        except Exception as e:
            logger.debug(f"JSON ayıklama hatası: {e}")
            return None

    def _build_core_prompt(self, agent_name, user_text, sec_result, op_result=None):
        """Ajanlar için dinamik Sistem Talimatlarını (System Prompt) inşa eder."""
        status_code, user_obj, sub_status = sec_result
        user_name = user_obj["name"] if user_obj else "Misafir"
        
        agent_def = AGENTS_CONFIG.get(agent_name, {})
        base_sys = agent_def.get('sys', "Yardımcı bir yapay zekasın.")

        bio_content = self._read_user_bio(user_obj) if status_code in ["ONAYLI", "SES_ONAYLI"] else ""
        time_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        team_str = ", ".join([n for n in AGENTS_CONFIG.keys() if n != agent_name])

        prompt_parts = [
            f"### KİMLİK VE ROL ###\nSen LotusAI işletim sisteminin {agent_name} isimli uzman ajanısın.",
            f"Görev Tanımın: {base_sys}",
            f"\n### ORTAM BİLGİLERİ ###\nŞu anki zaman: {time_str}\nKullanıcı: {user_name} (Güvenlik Durumu: {status_code})\nDiğer Aktif Ajanlar: {team_str}",
            f"Donanım Durumu: {self.device.upper()} Aktif"
        ]

        if sub_status == "TANIŞMA_MODU":
            prompt_parts.append("\n⚠️ GÜVENLİK PROTOKOLÜ: Kullanıcı henüz tam doğrulanmadı. Sadece tanışma ve temel bilgilendirme yap. Sistem yetkilerini kullandırma.")
        
        # Ajan bazlı canlı veri (Context)
        agent_instance = getattr(self, agent_name.lower(), None)
        if agent_instance and hasattr(agent_instance, "get_context_data"):
            try:
                # GAYA gibi ajanlar kullanıcı metnine göre context üretebilir
                if agent_name == "GAYA":
                    ctx = agent_instance.get_context_data(user_text)
                else:
                    ctx = agent_instance.get_context_data()
                
                if ctx: prompt_parts.append(f"\n### CANLI VERİ BAĞLAMI ###\n{ctx}")
            except Exception as e:
                logger.error(f"Context hatası ({agent_name}): {e}")

        if op_result:
            prompt_parts.append(f"\n### OPERASYONEL ANALİZ SONUCU ###\nSistem bu görevi senin için önceden işledi, bu sonucu yanıtına dahil et:\n{op_result}")
        
        if bio_content and status_code in ["ONAYLI", "SES_ONAYLI"]:
            prompt_parts.append(f"\n### YÖNETİCİ HAKKINDA ÖZEL BİLGİLER (KİŞİSELLEŞTİRME) ###\n{bio_content[:3000]}")

        # Stil Notu
        prompt_parts.append("\n### YANIT STİLİ ###\nProfesyonel, net ve LotusAI kimliğine uygun konuş. Gereksiz giriş cümlelerinden kaçın.")

        return "\n".join(prompt_parts)

    async def _handle_visual_tasks(self, clean_input, file_path, agent_name):
        """Görsel analiz ve belge işleme süreçlerini yönetir."""
        gemini_file_part = None
        op_result = None
        
        if file_path:
            gemini_file_part, _ = self._load_file_for_gemini(file_path)
        
        fatura_triggers = ["fatura", "fiş", "dekont", "hesap", "oku", "işle", "ne yazıyor", "analiz et"]
        is_visual_request = any(t in clean_input for t in fatura_triggers) or file_path is not None
        
        if is_visual_request and Config.AI_PROVIDER == "gemini":
            target_agent = "KERBEROS" if agent_name in ["KERBEROS", "ATLAS"] else "GAYA"
            
            # Eğer dosya yoksa ama kamera varsa anlık görüntü al
            if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
                frame = self.tools['camera'].get_frame()
                if frame is not None:
                    # Görüntü işleme sırasında GPU hızlandırması potansiyeli (Gelecek planı için)
                    _, buffer = cv2.imencode('.jpg', frame)
                    gemini_file_part = {"mime_type": "image/jpeg", "data": buffer.tobytes()}
            
            if gemini_file_part:
                prompt = "Bu görseli en ince ayrıntısına kadar analiz et. Eğer bu bir fatura veya finansal belge ise şu bilgileri JSON formatında çıkar: { 'firma': '', 'toplam_tutar': '', 'tarih': '', 'urunler': [], 'is_invoice': true }. Eğer belge değilse 'description' anahtarıyla görseli açıkla."
                json_resp = await self._query_gemini(target_agent, "Görsel analiz uzmanısın.", [], prompt, gemini_file_part)
                analysis_data = self._extract_json_from_text(json_resp["content"])
                
                if analysis_data:
                    agent_obj = getattr(self, target_agent.lower(), None)
                    if analysis_data.get('is_invoice'):
                        if target_agent == "KERBEROS" and hasattr(agent_obj, "audit_invoice"):
                            op_result = agent_obj.audit_invoice(analysis_data)
                        elif target_agent == "GAYA" and hasattr(agent_obj, "process_invoice_result"):
                            op_result = agent_obj.process_invoice_result(analysis_data)
                    else:
                        op_result = f"Görsel Analizi: {analysis_data.get('description', 'Analiz yapılamadı.')}"
        
        return gemini_file_part, op_result

    async def get_response(self, agent_name: str, user_text: str, sec_result, file_path: str = None):
        """Kullanıcı mesajına göre en uygun cevabı üretir (Ana Giriş Noktası)."""
        clean_input = self.nlp.clean_text(user_text)
        status, user_obj, sub_status = sec_result
        
        # 1. GÜVENLİK KONTROLÜ
        if status not in ["ONAYLI", "SES_ONAYLI"]:
            agent_name = "KERBEROS"
            if sub_status == "KAMERA_YOK":
                return {"agent": "KERBEROS", "content": "Güvenlik protokolü gereği sizi tanımam gerekiyor. Lütfen kameranızı aktif edin veya sesli doğrulama yapın."}

        # 2. SELAMLAŞMA VE HIZLI YANITLAR
        welcome_keywords = ["selam", "merhaba", "geldim", "buradayım", "hey lotus"]
        if status in ["ONAYLI", "SES_ONAYLI"] and len(clean_input.split()) <= 3:
            if any(w in clean_input for w in welcome_keywords):
                name = user_obj["name"] if user_obj else "Halil Bey"
                return {"agent": agent_name, "content": f"Hoş geldiniz {name}. LotusAI {self.device.upper()} destekli sistemleriyle aktif. Size nasıl yardımcı olabilirim?"}

        # 3. GÖRSEL/OPERASYONEL İŞLEMLER
        gemini_file_part, op_result = await self._handle_visual_tasks(clean_input, file_path, agent_name)

        # 4. AJAN ÖZEL FONKSİYONLARI (Dynamic Call)
        if not op_result:
            agent_obj = getattr(self, agent_name.lower(), None)
            if agent_obj and hasattr(agent_obj, "auto_handle"):
                try:
                    op_result = await agent_obj.auto_handle(clean_input)
                except Exception as e:
                    logger.error(f"Ajan otomatik işlem hatası ({agent_name}): {e}")

        # 5. YAPAY ZEKA (LLM) İLE CEVAP ÜRETİMİ
        sys_prompt = self._build_core_prompt(agent_name, clean_input, sec_result, op_result)
        
        try:
            recent, _ = self.memory.load_context(agent_name, clean_input)
        except:
            recent = []

        if Config.AI_PROVIDER == "gemini":
            return await self._query_gemini(agent_name, sys_prompt, recent, user_text, gemini_file_part)
        else:
            return await self._query_ollama(agent_name, sys_prompt, recent, user_text)

    async def get_team_response(self, user_text, sec_result):
        """Tüm ekipten brifing alır."""
        if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
            return [{"agent": "KERBEROS", "content": "Güvenlik onayı yetersiz, ekip brifingi başlatılamaz."}]

        active_agents = [n for n, loaded in AGENTS_LOADED.items() if loaded]
        tasks = []
        
        for agent in active_agents:
            sys_prompt = self._build_core_prompt(agent, user_text, sec_result) + "\n\n[GÖREV]: Konu hakkında kendi uzmanlık alanından tek cümlelik, çok kısa bir brifing ver."
            if Config.AI_PROVIDER == "gemini":
                tasks.append(self._query_gemini(agent, sys_prompt, [], user_text))
            else:
                tasks.append(self._query_ollama(agent, sys_prompt, [], user_text))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def _query_gemini(self, agent, sys_prompt, history, user_text, image_data=None):
        """Exponential Backoff ile Gemini API çağrısı."""
        api_key = Config.GEMINI_KEYS.get(agent) or Config.GEMINI_KEYS.get("ATLAS", "")
        if not api_key: return {"agent": agent, "content": "⚠️ Sistem hatası: API Anahtarı yapılandırılmamış."}
        
        genai.configure(api_key=api_key)
        
        gemini_hist = []
        for h in history:
            role = "user" if h["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [h["content"]]})

        for i in range(5): # 5 deneme
            try:
                model = genai.GenerativeModel(
                    model_name=Config.GEMINI_MODEL, 
                    system_instruction=sys_prompt
                )
                
                contents = [user_text]
                if image_data: contents.append(image_data)
                
                chat = model.start_chat(history=gemini_hist)
                # Thread kullanımı (API kütüphanesi senkron olduğu için)
                resp = await asyncio.to_thread(chat.send_message, contents)
                
                reply = resp.text.strip()
                self.memory.save(agent, "user", user_text)
                self.memory.save(agent, "model", reply)
                
                return {"agent": agent, "content": reply}
                
            except Exception as e:
                wait_time = 2 ** i
                logger.warning(f"Gemini denemesi {i+1} başarısız: {e}")
                if i == 4:
                    logger.error(f"❌ Gemini kritik hatası ({agent}): {e}")
                    return {"agent": agent, "content": "Şu an merkezi sinir sistemime (Gemini) ulaşamıyorum. Lütfen kısa süre sonra tekrar deneyin."}
                await asyncio.sleep(wait_time)

    async def _query_ollama(self, agent, sys_prompt, history, user_text):
        """Yerel Ollama sunucusuna istek gönderir (Ollama GPU kullanımını kendi yönetir)."""
        msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_text}]
        
        for i in range(2):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        Config.OLLAMA_URL, 
                        json={"model": Config.TEXT_MODEL, "messages": msgs, "stream": False}, 
                        timeout=120
                    ) as resp:
                        if resp.status == 200:
                            res = await resp.json()
                            reply = res.get("message", {}).get("content", "").strip()
                            self.memory.save(agent, "user", user_text)
                            self.memory.save(agent, "assistant", reply)
                            return {"agent": agent, "content": reply}
                        else:
                            logger.error(f"Ollama HTTP hatası: {resp.status}")
            except Exception as e:
                if i == 1:
                    logger.error(f"❌ Ollama bağlantı hatası: {e}")
                    return {"agent": agent, "content": "LotusAI yerel sunucusu (Ollama) şu an kapalı veya yanıt vermiyor."}
                await asyncio.sleep(2)



# import aiohttp
# import asyncio
# import io
# import json
# import os
# import re
# import time
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, Dict, List, Any, Union
# from PIL import Image
# import google.generativeai as genai

# # --- LOGLAMA YAPILANDIRMASI ---
# logger = logging.getLogger("LotusAI.Engine")

# # OpenCV Kontrolü (Görsel işlemler ve kamera için)
# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     logger.warning("⚠️ 'opencv-python' yüklü değil, kamera fonksiyonları kısıtlı çalışacaktır.")

# # --- KONFİGÜRASYON VE MODÜLLER ---
# from agents.definitions import AGENTS_CONFIG
# from config import Config
# from managers.nlp import NLPManager 

# # --- AJANLARIN GÜVENLİ YÜKLENMESİ ---
# AGENTS_LOADED = {
#     "ATLAS": False, "GAYA": False, "POYRAZ": False, 
#     "KURT": False, "SIDAR": False, "KERBEROS": False
# }

# try:
#     from agents.atlas import AtlasAgent
#     AGENTS_LOADED["ATLAS"] = True
# except ImportError as e: logger.error(f"❌ ATLAS Ajanı yüklenemedi: {e}")

# try:
#     from agents.gaya import GayaAgent
#     AGENTS_LOADED["GAYA"] = True
# except ImportError as e: logger.error(f"❌ GAYA Ajanı yüklenemedi: {e}")

# try:
#     from agents.poyraz import PoyrazAgent
#     AGENTS_LOADED["POYRAZ"] = True
# except ImportError as e: logger.error(f"❌ POYRAZ Ajanı yüklenemedi: {e}")

# try:
#     from agents.kurt import KurtAgent
#     AGENTS_LOADED["KURT"] = True
# except ImportError as e: logger.error(f"❌ KURT Ajanı yüklenemedi: {e}")

# try:
#     from agents.sidar import SidarAgent
#     AGENTS_LOADED["SIDAR"] = True
# except ImportError as e: logger.error(f"❌ SIDAR Ajanı yüklenemedi: {e}")

# try:
#     from agents.kerberos import KerberosAgent
#     AGENTS_LOADED["KERBEROS"] = True
# except ImportError as e: logger.error(f"❌ KERBEROS Ajanı yüklenemedi: {e}")


# class AgentEngine:
#     """
#     LotusAI Karar ve Cevap Üretim Motoru.
#     Ajanlar arası koordinasyonu, görsel analizi ve LLM (Gemini/Ollama) iletişimini yönetir.
#     """
#     def __init__(self, memory_manager, tools_dict: Dict[str, Any]):
#         self.memory = memory_manager
#         self.tools = tools_dict
#         self.app_id = getattr(Config, "APP_ID", "lotus-ai-core")
        
#         # NLP Yöneticisi (Metin temizleme ve anlama için)
#         self.nlp = tools_dict.get('nlp') or NLPManager()
        
#         # --- AJANLARI BAŞLAT ---
#         self.atlas = AtlasAgent(memory_manager, tools_dict) if AGENTS_LOADED["ATLAS"] else None
#         self.gaya = GayaAgent(tools_dict, self.nlp) if AGENTS_LOADED["GAYA"] else None
        
#         if "poyraz_special" in tools_dict:
#             self.poyraz = tools_dict["poyraz_special"]
#         else:
#             self.poyraz = PoyrazAgent(self.nlp, tools_dict) if AGENTS_LOADED["POYRAZ"] else None
            
#         self.kurt = KurtAgent(tools_dict) if AGENTS_LOADED["KURT"] else None
        
#         if "sidar_special" in tools_dict:
#             self.sidar = tools_dict["sidar_special"]
#         else:
#             sidar_tools = {k: tools_dict.get(k) for k in ['code', 'system', 'security']}
#             self.sidar = SidarAgent(sidar_tools) if AGENTS_LOADED["SIDAR"] else None

#         self.kerberos = KerberosAgent(tools_dict) if AGENTS_LOADED["KERBEROS"] else None

#     def determine_agent(self, text: str) -> Optional[str]:
#         """Kullanıcı girdisindeki anahtar kelimelere göre en uygun ajanı seçer."""
#         if not text: return "ATLAS"
#         clean_text = self.nlp.clean_text(text)
        
#         for name, data in AGENTS_CONFIG.items():
#             triggers = data.get("wake_words", []) + data.get("keys", [])
#             if any(k in clean_text for k in triggers):
#                 return name
        
#         return "ATLAS"

#     def _read_user_bio(self, user_obj: Optional[Dict] = None) -> str:
#         """Kullanıcı biyografisini okur (Kişiselleştirilmiş yanıtlar için)."""
#         bio_file = user_obj.get("bio_file") if user_obj else "halil_bio.txt"
#         bio_path = Path(Config.WORK_DIR) / bio_file
        
#         if not bio_path.exists():
#             bio_path = Path(Config.WORK_DIR) / "halil_bio.txt"

#         if bio_path.exists():
#             try:
#                 return bio_path.read_text(encoding="utf-8")
#             except Exception as e:
#                 logger.error(f"Biyografi okuma hatası: {e}")
#         return ""

#     def _load_file_for_gemini(self, file_path: Union[str, Path]):
#         """Dosyayı Gemini API'nin anlayacağı multimodal formatına hazırlar."""
#         path = Path(file_path)
#         if not path.exists(): 
#             return None, "Dosya bulunamadı."
        
#         ext = path.suffix.lower()
#         mime_types = {
#             ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
#             ".pdf": "application/pdf", ".txt": "text/plain"
#         }
#         mime_type = mime_types.get(ext, "application/octet-stream")
        
#         try:
#             return {"mime_type": mime_type, "data": path.read_bytes()}, None
#         except Exception as e: 
#             return None, str(e)

#     def _extract_json_from_text(self, text: str) -> Optional[Dict]:
#         """Yapay zeka çıktısı içindeki JSON bloklarını daha güvenli ayıklar."""
#         try:
#             # Önce kod bloğu içindeki json'u ara
#             json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
#             if json_block:
#                 return json.loads(json_block.group(1))
            
#             # Kod bloğu yoksa ilk { ve son } arasını al
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             return json.loads(match.group()) if match else None
#         except Exception as e:
#             logger.debug(f"JSON ayıklama hatası: {e}")
#             return None

#     def _build_core_prompt(self, agent_name, user_text, sec_result, op_result=None):
#         """Ajanlar için dinamik Sistem Talimatlarını (System Prompt) inşa eder."""
#         status_code, user_obj, sub_status = sec_result
#         user_name = user_obj["name"] if user_obj else "Misafir"
        
#         agent_def = AGENTS_CONFIG.get(agent_name, {})
#         base_sys = agent_def.get('sys', "Yardımcı bir yapay zekasın.")

#         bio_content = self._read_user_bio(user_obj) if status_code in ["ONAYLI", "SES_ONAYLI"] else ""
#         time_str = datetime.now().strftime("%d.%m.%Y %H:%M")
#         team_str = ", ".join([n for n in AGENTS_CONFIG.keys() if n != agent_name])

#         prompt_parts = [
#             f"### KİMLİK VE ROL ###\nSen LotusAI sisteminin {agent_name} isimli ajanısın. {base_sys}",
#             f"\n### ORTAM BİLGİLERİ ###\nŞu anki zaman: {time_str}\nKullanıcı: {user_name} (Güvenlik: {status_code})\nDiğer Aktif Ajanlar: {team_str}",
#         ]

#         if sub_status == "TANIŞMA_MODU":
#             prompt_parts.append("\n⚠️ GÜVENLİK UYARISI: Kullanıcı henüz tam doğrulanmadı. Sadece tanış ve nazik ol.")
        
#         # Ajan bazlı canlı veri (Context)
#         agent_instance = getattr(self, agent_name.lower(), None)
#         if agent_instance and hasattr(agent_instance, "get_context_data"):
#             try:
#                 ctx = agent_instance.get_context_data(user_text) if agent_name == "GAYA" else agent_instance.get_context_data()
#                 if ctx: prompt_parts.append(f"\n### CANLI VERİ BAĞLAMI ###\n{ctx}")
#             except Exception as e:
#                 logger.error(f"Context hatası ({agent_name}): {e}")

#         if op_result:
#             prompt_parts.append(f"\n### OPERASYONEL ANALİZ SONUCU ###\n{op_result}")
        
#         if bio_content and status_code in ["ONAYLI", "SES_ONAYLI"]:
#             prompt_parts.append(f"\n### YÖNETİCİ HAKKINDA ÖZEL BİLGİLER ###\n{bio_content[:2000]}")

#         return "\n".join(prompt_parts)

#     async def _handle_visual_tasks(self, clean_input, file_path, agent_name):
#         """Görsel analiz ve belge işleme süreçlerini yönetir."""
#         gemini_file_part = None
#         op_result = None
        
#         if file_path:
#             gemini_file_part, _ = self._load_file_for_gemini(file_path)
        
#         fatura_triggers = ["fatura", "fiş", "dekont", "hesap", "oku", "işle"]
#         is_invoice_request = any(t in clean_input for t in fatura_triggers)
        
#         if is_invoice_request and Config.AI_PROVIDER == "gemini":
#             target_agent = "KERBEROS" if agent_name in ["KERBEROS", "ATLAS"] else "GAYA"
            
#             # Kamera entegrasyonu
#             if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
#                 frame = self.tools['camera'].get_frame()
#                 if frame is not None:
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     gemini_file_part = {"mime_type": "image/jpeg", "data": buffer.tobytes()}
            
#             if gemini_file_part:
#                 prompt = "Bu görseli analiz et. Eğer bir fatura/fiş ise şu bilgileri JSON olarak çıkar: { 'firma': '', 'toplam_tutar': '', 'tarih': '', 'urunler': [] }. Değilse görseli açıkla."
#                 json_resp = await self._query_gemini(target_agent, "Veri ayıklama asistanısın.", [], prompt, gemini_file_part)
#                 invoice_data = self._extract_json_from_text(json_resp["content"])
                
#                 if invoice_data:
#                     agent_obj = getattr(self, target_agent.lower(), None)
#                     if target_agent == "KERBEROS" and hasattr(agent_obj, "audit_invoice"):
#                         op_result = agent_obj.audit_invoice(invoice_data)
#                     elif target_agent == "GAYA" and hasattr(agent_obj, "process_invoice_result"):
#                         op_result = agent_obj.process_invoice_result(invoice_data)
        
#         return gemini_file_part, op_result

#     async def get_response(self, agent_name: str, user_text: str, sec_result, file_path: str = None):
#         """Kullanıcı mesajına göre en uygun cevabı üretir (Ana Giriş Noktası)."""
#         clean_input = self.nlp.clean_text(user_text)
#         status, user_obj, sub_status = sec_result
        
#         # 1. GÜVENLİK KONTROLÜ
#         if status not in ["ONAYLI", "SES_ONAYLI"]:
#             agent_name = "KERBEROS"
#             if sub_status == "KAMERA_YOK":
#                 return {"agent": "KERBEROS", "content": "Güvenlik protokolü: Kimliğinizi doğrulamak için lütfen kamerayı aktif edin."}

#         # 2. SELAMLAŞMA
#         welcome_keywords = ["selam", "merhaba", "geldim", "buradayım"]
#         if status in ["ONAYLI", "SES_ONAYLI"] and len(clean_input.split()) <= 3:
#             if any(w in clean_input for w in welcome_keywords):
#                 name = user_obj["name"] if user_obj else "Halil Bey"
#                 return {"agent": agent_name, "content": f"Hoş geldiniz {name}. LotusAI aktif ve emirlerinizi bekliyor."}

#         # 3. GÖRSEL/OPERASYONEL İŞLEMLER
#         gemini_file_part, op_result = await self._handle_visual_tasks(clean_input, file_path, agent_name)

#         # 4. AJAN ÖZEL FONKSİYONLARI (Dynamic Call)
#         if not op_result:
#             agent_obj = getattr(self, agent_name.lower(), None)
#             if agent_obj:
#                 # Her ajanın kendi özel 'handle_task' metoduna sahip olduğunu varsayıyoruz (yoksa LLM'e geçer)
#                 if hasattr(agent_obj, "auto_handle"):
#                     op_result = await agent_obj.auto_handle(clean_input)

#         # 5. YAPAY ZEKA (LLM) İLE CEVAP ÜRETİMİ
#         sys_prompt = self._build_core_prompt(agent_name, clean_input, sec_result, op_result)
        
#         try:
#             recent, _ = self.memory.load_context(agent_name, clean_input)
#         except:
#             recent = []

#         if Config.AI_PROVIDER == "gemini":
#             return await self._query_gemini(agent_name, sys_prompt, recent, user_text, gemini_file_part)
#         else:
#             return await self._query_ollama(agent_name, sys_prompt, recent, user_text)

#     async def get_team_response(self, user_text, sec_result):
#         """Tüm ekipten brifing alır."""
#         if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
#             return [{"agent": "KERBEROS", "content": "Güvenlik onayı yetersiz."}]

#         active_agents = [n for n, loaded in AGENTS_LOADED.items() if loaded]
#         tasks = []
        
#         for agent in active_agents:
#             sys_prompt = self._build_core_prompt(agent, user_text, sec_result) + "\n\n[GÖREV]: Çok KISA bir yorum yap."
#             if Config.AI_PROVIDER == "gemini":
#                 tasks.append(self._query_gemini(agent, sys_prompt, [], user_text))
#             else:
#                 tasks.append(self._query_ollama(agent, sys_prompt, [], user_text))
            
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         return [r for r in results if isinstance(r, dict)]

#     async def _query_gemini(self, agent, sys_prompt, history, user_text, image_data=None):
#         """Exponential Backoff ile Gemini API çağrısı."""
#         api_key = Config.GEMINI_KEYS.get(agent) or Config.GEMINI_KEYS.get("ATLAS", "")
#         if not api_key: return {"agent": agent, "content": "⚠️ API Anahtarı bulunamadı."}
        
#         genai.configure(api_key=api_key)
        
#         gemini_hist = []
#         for h in history:
#             role = "user" if h["role"] == "user" else "model"
#             gemini_hist.append({"role": role, "parts": [h["content"]]})

#         for i in range(5):
#             try:
#                 model = genai.GenerativeModel(
#                     model_name=Config.GEMINI_MODEL, 
#                     system_instruction=sys_prompt
#                 )
                
#                 contents = [user_text]
#                 if image_data: contents.append(image_data)
                
#                 chat = model.start_chat(history=gemini_hist)
#                 resp = await asyncio.to_thread(chat.send_message, contents)
                
#                 reply = resp.text.strip()
#                 self.memory.save(agent, "user", user_text)
#                 self.memory.save(agent, "model", reply)
                
#                 return {"agent": agent, "content": reply}
                
#             except Exception as e:
#                 wait_time = 2 ** i
#                 if i == 4:
#                     logger.error(f"❌ Gemini hatası ({agent}): {e}")
#                     return {"agent": agent, "content": "Şu an bağlantı kurulamıyor. Lütfen tekrar deneyin."}
#                 await asyncio.sleep(wait_time)

#     async def _query_ollama(self, agent, sys_prompt, history, user_text):
#         """Yerel Ollama sunucusuna istek gönderir."""
#         msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_text}]
        
#         # Ollama için de basit bir retry ekledik
#         for i in range(2):
#             try:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.post(
#                         Config.OLLAMA_URL, 
#                         json={"model": Config.TEXT_MODEL, "messages": msgs, "stream": False}, 
#                         timeout=120
#                     ) as resp:
#                         if resp.status == 200:
#                             res = await resp.json()
#                             reply = res.get("message", {}).get("content", "").strip()
#                             self.memory.save(agent, "user", user_text)
#                             self.memory.save(agent, "assistant", reply)
#                             return {"agent": agent, "content": reply}
#             except Exception as e:
#                 if i == 1:
#                     logger.error(f"❌ Ollama hatası: {e}")
#                     return {"agent": agent, "content": "Yerel sunucu yanıt vermiyor."}
#                 await asyncio.sleep(2)


# import aiohttp
# import asyncio
# import io
# import json
# import os
# import re
# import time
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, Dict, List, Any, Union
# from PIL import Image
# import google.generativeai as genai

# # --- LOGLAMA YAPILANDIRMASI ---
# # Sistemin arka planda ne yaptığını izlemek için standart loglama kullanıyoruz.
# logger = logging.getLogger("LotusAI.Engine")

# # OpenCV Kontrolü (Görsel işlemler ve kamera için)
# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     logger.warning("⚠️ 'opencv-python' yüklü değil, kamera fonksiyonları kısıtlı çalışacaktır.")

# # --- KONFİGÜRASYON VE MODÜLLER ---
# from agents.definitions import AGENTS_CONFIG
# from config import Config
# from managers.nlp import NLPManager 

# # --- AJANLARIN GÜVENLİ YÜKLENMESİ ---
# # Herhangi bir ajan dosyasında hata olsa bile ana motorun çökmesini engelliyoruz.
# AGENTS_LOADED = {
#     "ATLAS": False, "GAYA": False, "POYRAZ": False, 
#     "KURT": False, "SIDAR": False, "KERBEROS": False
# }

# try:
#     from agents.atlas import AtlasAgent
#     AGENTS_LOADED["ATLAS"] = True
# except ImportError as e: logger.error(f"❌ ATLAS Ajanı yüklenemedi: {e}")

# try:
#     from agents.gaya import GayaAgent
#     AGENTS_LOADED["GAYA"] = True
# except ImportError as e: logger.error(f"❌ GAYA Ajanı yüklenemedi: {e}")

# try:
#     from agents.poyraz import PoyrazAgent
#     AGENTS_LOADED["POYRAZ"] = True
# except ImportError as e: logger.error(f"❌ POYRAZ Ajanı yüklenemedi: {e}")

# try:
#     from agents.kurt import KurtAgent
#     AGENTS_LOADED["KURT"] = True
# except ImportError as e: logger.error(f"❌ KURT Ajanı yüklenemedi: {e}")

# try:
#     from agents.sidar import SidarAgent
#     AGENTS_LOADED["SIDAR"] = True
# except ImportError as e: logger.error(f"❌ SIDAR Ajanı yüklenemedi: {e}")

# try:
#     from agents.kerberos import KerberosAgent
#     AGENTS_LOADED["KERBEROS"] = True
# except ImportError as e: logger.error(f"❌ KERBEROS Ajanı yüklenemedi: {e}")


# class AgentEngine:
#     """
#     LotusAI Karar ve Cevap Üretim Motoru.
#     Ajanlar arası koordinasyonu, görsel analizi ve LLM (Gemini/Ollama) iletişimini yönetir.
#     """
#     def __init__(self, memory_manager, tools_dict: Dict[str, Any]):
#         self.memory = memory_manager
#         self.tools = tools_dict
        
#         # NLP Yöneticisi (Metin temizleme ve anlama için)
#         self.nlp = tools_dict.get('nlp') or NLPManager()
        
#         # --- AJANLARI BAŞLAT ---
#         # Ajanlara hem hafıza hem de kullanabilecekleri araçları (tools_dict) enjekte ediyoruz.
#         self.atlas = AtlasAgent(memory_manager, tools_dict) if AGENTS_LOADED["ATLAS"] else None
#         self.gaya = GayaAgent(tools_dict, self.nlp) if AGENTS_LOADED["GAYA"] else None
        
#         if "poyraz_special" in tools_dict:
#             self.poyraz = tools_dict["poyraz_special"]
#         else:
#             self.poyraz = PoyrazAgent(self.nlp, tools_dict) if AGENTS_LOADED["POYRAZ"] else None
            
#         self.kurt = KurtAgent(tools_dict) if AGENTS_LOADED["KURT"] else None
        
#         if "sidar_special" in tools_dict:
#             self.sidar = tools_dict["sidar_special"]
#         else:
#             # Sidar için sadece ilgili teknik araçları ayırıyoruz
#             sidar_tools = {k: tools_dict.get(k) for k in ['code', 'system', 'security']}
#             self.sidar = SidarAgent(sidar_tools) if AGENTS_LOADED["SIDAR"] else None

#         self.kerberos = KerberosAgent(tools_dict) if AGENTS_LOADED["KERBEROS"] else None

#     def determine_agent(self, text: str) -> Optional[str]:
#         """Kullanıcı girdisindeki anahtar kelimelere göre en uygun ajanı seçer."""
#         if not text: return None
#         clean_text = self.nlp.clean_text(text)
        
#         for name, data in AGENTS_CONFIG.items():
#             triggers = data.get("wake_words", []) + data.get("keys", [])
#             if any(k in clean_text for k in triggers):
#                 return name
        
#         # Eğer özel bir ajan tetiklenmediyse, lider (ATLAS) cevap verir.
#         return "ATLAS"

#     def _read_user_bio(self, user_obj: Optional[Dict] = None) -> str:
#         """Patron (Halil Bey) veya kullanıcı biyografisini güvenli bir şekilde okur."""
#         # Eğer kullanıcı objesinde özel bir biyografi dosyası belirtilmişse onu oku
#         bio_file = user_obj.get("bio_file") if user_obj else "halil_bio.txt"
#         bio_path = Path(Config.WORK_DIR) / bio_file
        
#         if not bio_path.exists():
#             # Varsayılana geri dön
#             bio_path = Path(Config.WORK_DIR) / "halil_bio.txt"

#         if bio_path.exists():
#             try:
#                 return bio_path.read_text(encoding="utf-8")
#             except Exception as e:
#                 logger.error(f"Biyografi okuma hatası: {e}")
#         return ""

#     def _load_file_for_gemini(self, file_path: Union[str, Path]):
#         """Dosyayı Gemini API'nin anlayacağı multimodal (görsel/belge) formatına hazırlar."""
#         path = Path(file_path)
#         if not path.exists(): 
#             return None, "Dosya bulunamadı."
        
#         ext = path.suffix.lower()
#         mime_types = {
#             ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
#             ".pdf": "application/pdf", ".txt": "text/plain"
#         }
#         mime_type = mime_types.get(ext, "application/octet-stream")
        
#         try:
#             return {"mime_type": mime_type, "data": path.read_bytes()}, None
#         except Exception as e: 
#             return None, str(e)

#     def _extract_json_from_text(self, text: str) -> Optional[Dict]:
#         """Yapay zeka çıktısı içindeki JSON bloklarını ayıklar (Fatura işleme vb. için)."""
#         try:
#             # Markdown içindeki ```json ... ``` bloklarını veya çıplak { ... } yapılarını bulur
#             match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
#             return json.loads(match.group()) if match else None
#         except: return None

#     def _build_core_prompt(self, agent_name, user_text, sec_result, op_result=None):
#         """Ajanlar için dinamik Sistem Talimatlarını (System Prompt) inşa eder."""
#         status_code, user_obj, sub_status = sec_result
#         user_name = user_obj["name"] if user_obj else "Misafir"
        
#         # Temel ajan kimliği
#         agent_def = AGENTS_CONFIG.get(agent_name, {})
#         base_sys = agent_def.get('sys', "Yardımcı bir yapay zekasın.")

#         # Dinamik Ortam Bilgileri
#         bio_content = self._read_user_bio(user_obj) if status_code in ["ONAYLI", "SES_ONAYLI"] else ""
#         time_str = datetime.now().strftime("%d.%m.%Y %H:%M")
#         team_str = ", ".join([n for n in AGENTS_CONFIG.keys() if n != agent_name])

#         prompt_parts = [
#             f"### KİMLİK VE ROL ###\n{base_sys}",
#             f"\n### ORTAM BİLGİLERİ ###\nZaman: {time_str}\nKullanıcı: {user_name} (Güvenlik Durumu: {status_code})\nEkip Arkadaşların: {team_str}",
#         ]

#         if sub_status == "TANIŞMA_MODU":
#             prompt_parts.append("\n⚠️ GÜVENLİK UYARISI: Kullanıcı henüz tam doğrulanmadı. Sadece tanış ve nazik ol, kritik veri paylaşma.")
        
#         # Ajan bazlı canlı veri (Context) ekleme
#         agent_instance = getattr(self, agent_name.lower(), None)
#         if agent_instance and hasattr(agent_instance, "get_context_data"):
#             try:
#                 # Bazı ajanlar (Gaya gibi) girdi metnine göre bağlam üretir
#                 ctx = agent_instance.get_context_data(user_text) if agent_name == "GAYA" else agent_instance.get_context_data()
#                 if ctx: prompt_parts.append(f"\n### CANLI VERİ BAĞLAMI ###\n{ctx}")
#             except Exception as e:
#                 logger.error(f"Context hatası ({agent_name}): {e}")

#         # Eğer bir operasyon (fatura okuma, rezervasyon vb.) yapıldıysa sonucunu prompta ekle
#         if op_result:
#             prompt_parts.append(f"\n### OPERASYONEL ANALİZ SONUCU ###\n{op_result}")
        
#         # Patron/Yönetici bilgisi (Hassas içerik)
#         if bio_content and status_code in ["ONAYLI", "SES_ONAYLI"]:
#             prompt_parts.append(f"\n### YÖNETİCİ HAKKINDA ÖZEL BİLGİLER ###\n{bio_content[:1500]}")

#         return "\n".join(prompt_parts)

#     async def _handle_visual_tasks(self, clean_input, file_path, agent_name):
#         """Görsel analiz (Multimodal) ve fatura işleme süreçlerini yönetir."""
#         gemini_file_part = None
#         op_result = None
        
#         # 1. Dosya Hazırlığı
#         if file_path:
#             gemini_file_part, _ = self._load_file_for_gemini(file_path)
        
#         # 2. Fatura/Fiş Tetikleyici Kontrolü
#         fatura_triggers = ["fatura", "fiş", "dekont", "hesap", "oku", "işle"]
#         is_invoice_request = any(t in clean_input for t in fatura_triggers)
        
#         if is_invoice_request and Config.AI_PROVIDER == "gemini":
#             target_agent = "KERBEROS" if agent_name in ["KERBEROS", "ATLAS"] else "GAYA"
            
#             # Eğer yüklenmiş dosya yoksa ama kamera aktifse anlık görüntü al
#             if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
#                 frame = self.tools['camera'].get_frame()
#                 if frame is not None:
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     gemini_file_part = {"mime_type": "image/jpeg", "data": buffer.tobytes()}
            
#             # Gemini ile görselden veri ayıkla
#             if gemini_file_part:
#                 prompt = "Bu görseldeki faturayı analiz et ve SADECE JSON formatında döndür: { 'firma': '', 'toplam_tutar': '', 'urunler': [] }"
#                 json_resp = await self._query_gemini(target_agent, "Veri ayıklama asistanısın.", [], prompt, gemini_file_part)
#                 invoice_data = self._extract_json_from_text(json_resp["content"])
                
#                 if invoice_data:
#                     agent_obj = getattr(self, target_agent.lower(), None)
#                     if target_agent == "KERBEROS" and agent_obj:
#                         op_result = agent_obj.audit_invoice(invoice_data)
#                     elif target_agent == "GAYA" and agent_obj:
#                         op_result = agent_obj.process_invoice_result(invoice_data)
        
#         return gemini_file_part, op_result

#     async def get_response(self, agent_name: str, user_text: str, sec_result, file_path: str = None):
#         """Kullanıcı mesajına göre en uygun cevabı üretir (Ana Giriş Noktası)."""
#         clean_input = self.nlp.clean_text(user_text)
#         status, user_obj, sub_status = sec_result
        
#         # 1. GÜVENLİK KONTROLÜ (Doğrulanmamış kullanıcıyla sadece Kerberos muhatap olur)
#         if status not in ["ONAYLI", "SES_ONAYLI"]:
#             agent_name = "KERBEROS"
#             if sub_status == "KAMERA_YOK":
#                 return {"agent": "KERBEROS", "content": "Güvenlik protokolü gereği sizi görmem gerekiyor. Lütfen kamerayı aktif edin."}

#         # 2. SELAMLAŞMA VE KARŞILAMA (Dosya 1'den gelen özellik)
#         welcome_keywords = ["selam", "merhaba", "geldim", "buradayım", "gördün mü"]
#         if status in ["ONAYLI", "SES_ONAYLI"] and len(clean_input.split()) <= 3:
#             if any(w in clean_input for w in welcome_keywords):
#                 name = user_obj["name"] if user_obj else "Halil Bey"
#                 return {"agent": agent_name, "content": f"Sizi görüyorum {name}, hoş geldiniz. Sistem tüm fonksiyonlarıyla emrinizde."}

#         # 3. GÖRSEL/FATURA İŞLEME
#         gemini_file_part, op_result = await self._handle_visual_tasks(clean_input, file_path, agent_name)

#         # 4. ÖZEL AJAN TETİKLEYİCİLERİ (Ajanların kendi metodlarını çalıştırması)
#         if not op_result:
#             agent_obj = getattr(self, agent_name.lower(), None)
#             if agent_name == "GAYA" and agent_obj:
#                 if "rezervasyon" in clean_input:
#                     op_result = agent_obj.handle_reservation(clean_input, user_obj["name"] if user_obj else "Misafir")
#             elif agent_name == "POYRAZ" and agent_obj:
#                 if any(k in clean_input for k in ["analiz", "yorum", "rapor", "performans"]):
#                     op_result = agent_obj.analyze_business_performance()
#             elif agent_name == "SIDAR" and agent_obj:
#                 if any(k in clean_input for k in ["sistem", "sağlık", "durum", "arşiv"]):
#                     op_result = agent_obj.perform_system_check()

#         # 5. YAPAY ZEKA (LLM) İLE CEVAP ÜRETİMİ
#         sys_prompt = self._build_core_prompt(agent_name, clean_input, sec_result, op_result)
        
#         # Geçmiş hafızayı yükle
#         try:
#             recent, _ = self.memory.load_context(agent_name, clean_input)
#         except:
#             recent = []

#         if Config.AI_PROVIDER == "gemini":
#             return await self._query_gemini(agent_name, sys_prompt, recent, user_text, gemini_file_part)
#         else:
#             return await self._query_ollama(agent_name, sys_prompt, recent, user_text)

#     async def get_team_response(self, user_text, sec_result):
#         """Ekip Brifingi: Tüm aktif ajanlardan aynı anda fikir alır."""
#         if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
#             return [{"agent": "KERBEROS", "content": "Güvenlik onayı olmadan ekip brifingi veremem."}]

#         tasks = []
#         # Sadece yüklenmiş olan ajanları listeye al
#         active_agents = [n for n, loaded in AGENTS_LOADED.items() if loaded]
        
#         for agent in active_agents:
#             sys_prompt = self._build_core_prompt(agent, user_text, sec_result) + "\n\n[GÖREV]: Çok KISA (en fazla 2 cümle) bir fikir beyan et."
#             if Config.AI_PROVIDER == "gemini":
#                 tasks.append(self._query_gemini(agent, sys_prompt, [], user_text))
#             else:
#                 tasks.append(self._query_ollama(agent, sys_prompt, [], user_text))
            
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         return [r for r in results if isinstance(r, dict)]

#     async def _query_gemini(self, agent, sys_prompt, history, user_text, image_data=None):
#         """
#         Üstel Geri Çekilme (Exponential Backoff) ile Gemini API çağrısı yapar.
#         Ağ hatalarında veya yoğunlukta otomatik olarak bekleyip tekrar dener.
#         """
#         api_key = Config.GEMINI_KEYS.get(agent) or Config.GEMINI_KEYS.get("ATLAS", "")
#         if not api_key: return {"agent": agent, "content": "⚠️ API Anahtarı eksik, yapılandırmayı kontrol edin."}
        
#         genai.configure(api_key=api_key)
        
#         # Geçmişi Gemini'nin beklediği 'role' yapısına çeviriyoruz
#         gemini_hist = []
#         for h in history:
#             role = "user" if h["role"] == "user" else "model"
#             gemini_hist.append({"role": role, "parts": [h["content"]]})

#         # Retry Döngüsü (Hata payına karşı 5 deneme)
#         for i in range(5):
#             try:
#                 # Modeli yapılandır (Config'den çekilen model ismini kullan)
#                 model = genai.GenerativeModel(
#                     model_name=Config.GEMINI_MODEL, 
#                     system_instruction=sys_prompt
#                 )
                
#                 contents = [user_text]
#                 if image_data: contents.append(image_data)
                
#                 chat = model.start_chat(history=gemini_hist)
#                 # API çağrısını asenkron thread içinde yapıyoruz
#                 resp = await asyncio.to_thread(chat.send_message, contents)
                
#                 reply = resp.text.strip()
                
#                 # Başarılı cevabı hafızaya kaydet
#                 self.memory.save(agent, "user", user_text)
#                 self.memory.save(agent, "model", reply)
                
#                 return {"agent": agent, "content": reply}
                
#             except Exception as e:
#                 wait_time = 2 ** i # 1, 2, 4, 8, 16 saniye bekleme
#                 if i == 4: # Son deneme de başarısızsa
#                     logger.error(f"❌ Gemini kritik hata ({agent}): {e}")
#                     return {"agent": agent, "content": "Bulut sunucusu şu an yanıt vermiyor, yerel modele geçmeyi deneyebilirsiniz."}
#                 await asyncio.sleep(wait_time)

#     async def _query_ollama(self, agent, sys_prompt, history, user_text):
#         """Yerel Ollama sunucusuna (Llama/Mistral vb.) istek gönderir."""
#         msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_text}]
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     Config.OLLAMA_URL, 
#                     json={"model": Config.TEXT_MODEL, "messages": msgs, "stream": False}, 
#                     timeout=90
#                 ) as resp:
#                     if resp.status == 200:
#                         res = await resp.json()
#                         reply = res.get("message", {}).get("content", "").strip()
                        
#                         self.memory.save(agent, "user", user_text)
#                         self.memory.save(agent, "assistant", reply)
                        
#                         return {"agent": agent, "content": reply}
#                     return {"agent": agent, "content": f"Yerel model hatası (Kod: {resp.status})"}
#         except Exception as e:
#             logger.error(f"❌ Ollama erişim hatası: {e}")
#             return {"agent": agent, "content": "Yerel yapay zeka sunucusuna ulaşılamıyor."}





# import aiohttp
# import asyncio
# import io
# import json
# import os
# import re
# import time
# from datetime import datetime
# from PIL import Image
# import google.generativeai as genai

# # OpenCV (Kamera) Modülü Kontrollü İçe Aktarma
# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     print("⚠️ UYARI: 'opencv-python' yüklü değil, kamera fonksiyonları çalışmayabilir.")

# # --- Konfigürasyon ve Tanımlar ---
# from agents.definitions import AGENTS_CONFIG
# from config import Config
# from managers.nlp import NLPManager 

# # --- AJAN MODÜLLERİ ---
# try:
#     from agents.atlas import AtlasAgent
#     from agents.gaya import GayaAgent
#     from agents.poyraz import PoyrazAgent
#     from agents.kurt import KurtAgent
#     from agents.sidar import SidarAgent
#     from agents.kerberos import KerberosAgent
# except ImportError as e:
#     print(f"KRİTİK HATA: Ajan dosyaları (agents/...) eksik veya hatalı: {e}")

# class AgentEngine:
#     """
#     Lotus AI'ın Karar ve Cevap Üretim Motoru.
#     Tüm ajanların beyin takımıdır. Özel işleri ilgili ajan modülüne devreder.
#     API (Gemini/Ollama) iletişimini ve Prompt mühendisliğini burası yönetir.
#     """
#     def __init__(self, memory_manager, tools_dict):
#         self.memory = memory_manager
#         self.tools = tools_dict
        
#         # NLP Manager'ı tools içinde varsa oradan al, yoksa yeni oluştur
#         if 'nlp' in tools_dict:
#             self.nlp = tools_dict['nlp']
#         else:
#             self.nlp = NLPManager()
        
#         # --- Ajanları Başlat ---
#         # 1. ATLAS (Hafıza Yöneticisi)
#         self.atlas = AtlasAgent(memory_manager)
        
#         # 2. GAYA (Operasyon Yöneticisi)
#         self.gaya = GayaAgent(tools_dict, self.nlp)
        
#         # 3. POYRAZ (Veri Analisti) - ENJEKSİYON KONTROLÜ
#         # lotus_system.py içinde oluşturulup 'poyraz_special' olarak gönderiliyor.
#         if "poyraz_special" in tools_dict:
#             self.poyraz = tools_dict["poyraz_special"]
#         else:
#             # Fallback (Eğer main dosyasından gelmediyse)
#             self.poyraz = PoyrazAgent(self.nlp)
            
#         # 4. KURT (Saha Operasyon)
#         self.kurt = KurtAgent(tools_dict)
        
#         # 5. SİDAR (Güvenlik ve Arşiv) - ENJEKSİYON KONTROLÜ
#         # lotus_system.py içinde oluşturulup 'sidar_special' olarak gönderiliyor.
#         if "sidar_special" in tools_dict:
#             self.sidar = tools_dict["sidar_special"]
#         else:
#             # Fallback: Eski usül (Hata verebilir çünkü yeni Sidar parametre istiyor)
#             print("⚠️ UYARI: SidarAgent main dosyasından inject edilmedi, başlatılamayabilir.")
#             try:
#                 self.sidar = SidarAgent(tools_dict) 
#             except TypeError:
#                 print("❌ SidarAgent başlatılamadı (Eksik parametre).")
#                 self.sidar = None

#         # 6. KERBEROS (Güvenlik Şefi)
#         self.kerberos = KerberosAgent(tools_dict)
        
#         # Medya Yöneticisi kontrolü
#         if 'media' not in self.tools:
#             try:
#                 from managers.media import MediaManager
#                 self.tools['media'] = MediaManager()
#             except ImportError: 
#                 pass 
#             except Exception as e:
#                 print(f"⚠️ MediaManager başlatılamadı: {e}")

#     def determine_agent(self, text):
#         """Kullanıcı girdisine göre hangi ajanın devreye gireceğini belirler."""
#         if not text:
#             return None
        
#         clean_text = self.nlp.clean_text(text)
        
#         # Tanımlardaki anahtar kelimeleri tara
#         for name, data in AGENTS_CONFIG.items():
#             triggers = data.get("wake_words", []) + data.get("keys", [])
#             # Basit eşleşme kontrolü
#             if any(k in clean_text for k in triggers):
#                 return name
#         return None

#     def _read_user_bio(self, bio_filename):
#         """Kullanıcı biyografisini okur (Kişiselleştirilmiş deneyim için)."""
#         content = "Kullanıcı hakkında biyografik bilgi bulunamadı."
        
#         if bio_filename and os.path.exists(bio_filename):
#             try:
#                 with open(bio_filename, "r", encoding="utf-8") as f: 
#                     return f.read()
#             except: pass
        
#         default_bio = os.path.join(Config.WORK_DIR, "halil_bio.txt")
#         if os.path.exists(default_bio):
#             try:
#                 with open(default_bio, "r", encoding="utf-8") as f: 
#                     content = f.read()
#             except: pass
            
#         return content

#     def _load_file_for_gemini(self, file_path):
#         """Dosyayı Gemini API formatına hazırlar."""
#         if not file_path or not os.path.exists(file_path): 
#             return None, "Dosya bulunamadı."
        
#         low = file_path.lower()
#         mime_type = "application/octet-stream"
        
#         if low.endswith(".png"): mime_type = "image/png"
#         elif low.endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"
#         elif low.endswith(".pdf"): mime_type = "application/pdf"
#         elif low.endswith(".txt"): mime_type = "text/plain"
#         elif low.endswith(".json"): mime_type = "application/json"
        
#         try:
#             with open(file_path, "rb") as f: 
#                 file_data = f.read()
#             return {"mime_type": mime_type, "data": file_data}, None
#         except Exception as e: 
#             return None, str(e)

#     def _extract_json_from_text(self, text):
#         """AI yanıtından JSON bloğunu ayıklar."""
#         try:
#             text = text.strip()
#             if "```" in text:
#                 parts = text.split("```")
#                 for part in parts:
#                     if "{" in part: 
#                         text = part.replace("json", "").strip()
#                         break
            
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             return json.loads(match.group()) if match else json.loads(text)
#         except: 
#             return None

#     def _build_core_prompt(self, agent_name, user_text, sec_result, vision_desc=None, research_data=None):
#         """
#         Prompt iskeletini oluşturur.
#         Tüm bağlamı (Context) burada birleştirir.
#         """
#         status_code, user_obj, sub_status = sec_result
#         user_name = user_obj["name"] if user_obj else "Yabancı"
#         user_level = user_obj.get("level", 0) if user_obj else 0
        
#         # Eğer güvenlik ihlali varsa Biyo okuma
#         bio_content = ""
#         if status_code in ["ONAYLI", "SES_ONAYLI"]:
#             bio_content = self._read_user_bio(user_obj.get("bio_file") if user_obj else "halil_bio.txt")
        
#         menu_summary = "Menü verisi yok."
#         if 'operations' in self.tools:
#             menu_summary = self.tools['operations'].get_context_summary()

#         locale_months = {
#             1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
#             7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
#         }
#         now = datetime.now()
#         time_str = f"{now.day} {locale_months.get(now.month)} {now.year}, Saat: {now.strftime('%H:%M')}"
#         team_str = ", ".join([n for n in AGENTS_CONFIG.keys() if n != agent_name])
        
#         base_sys = AGENTS_CONFIG.get(agent_name, {}).get('sys', "Sen yardımcı bir yapay zekasın.")
        
#         # --- PROMPT İNŞASI ---
#         prompt_parts = [
#             f"### SİSTEM KİMLİĞİ VE ROLÜN ###",
#             f"{base_sys}",
            
#             f"\n### ORTAM VE BAĞLAM ###",
#             f"Tarih/Saat: {time_str}",
#             f"Kullanıcı: {user_name} (Durum: {status_code})",
#             f"Diğer Ajanlar: {team_str}",
#         ]
        
#         # --- ÖZEL DURUM TALİMATLARI ---
        
#         # 1. YABANCI (TANIŞMA MODU)
#         if sub_status == "TANIŞMA_MODU":
#             prompt_parts.append("\n⚠️ DİKKAT: Kullanıcıyı kamera görüyor ama veritabanında KAYITLI DEĞİL. İşletme verisi paylaşma. Sadece sohbet et, adını sor ve tanış.")

#         # 2. PATRON (SAMİMİYET MODU)
#         elif status_code in ["ONAYLI", "SES_ONAYLI"] and user_level >= 5:
#             prompt_parts.append(f"\n⚠️ İLİŞKİ DURUMU: Karşındaki kişi PATRONUN {user_name}. Onu yıllardır tanıyorsun. Asla kendini resmi bir dille tanıtma. Samimi ve esprili cevap ver.")

#         # 3. SES İLE TANIMA (Görüntü Yok)
#         if status_code == "SES_ONAYLI":
#             prompt_parts.append(f"\n🎙️ BİLGİ: Kullanıcıyı kamerada GÖREMİYORSUN ama ses imzasından kimliğini ({user_name}) doğruladın.")

#         if status_code in ["ONAYLI", "SES_ONAYLI"]:
#             prompt_parts.append(f"\n### İŞLETME BİLGİLERİ ###\nŞirket: Nilüfer Bağevi\nHizmetler: {menu_summary}")
#             if bio_content: 
#                 prompt_parts.append(f"\n### PATRON HAKKINDA ###\n{bio_content[:1500]}...")

#         # --- AJANA ÖZEL BAĞLAM (Dynamic Context) ---
#         ctx = ""
#         try:
#             if agent_name == "ATLAS": ctx = self.atlas.get_context_data()
#             elif agent_name == "GAYA": ctx = self.gaya.get_context_data(user_text)
#             elif agent_name == "POYRAZ": ctx = self.poyraz.get_context_data() if self.poyraz else ""
#             elif agent_name == "KURT": ctx = self.kurt.get_context_data()
#             elif agent_name == "SİDAR": ctx = self.sidar.get_context_data() if self.sidar else ""
#             elif agent_name == "KERBEROS": ctx = self.kerberos.get_context_data()
#         except Exception as e:
#             print(f"Bağlam hatası ({agent_name}): {e}")
        
#         if ctx: prompt_parts.append(ctx)

#         if research_data: prompt_parts.append(f"\n### ARAŞTIRMA/VERİ SONUCU ###\n{research_data}")
#         if vision_desc: prompt_parts.append(f"\n### GÖRSEL ALGISI ###\n{vision_desc}")
        
#         prompt_parts.append(f"\n### KULLANICI MESAJI ###\n'{user_text}'")
        
#         return "\n".join(prompt_parts)

#     def _prepare_context(self, agent_name, user_text, sec_result, vision_desc=None, research_data=None):
#         core_prompt = self._build_core_prompt(agent_name, user_text, sec_result, vision_desc, research_data)
#         if Config.AI_PROVIDER != "gemini":
#             core_prompt += f"\n\nÖNEMLİ: Asla 'User:', 'System:' etiketleri kullanma. Sadece {agent_name} rolünde konuş."
#         return core_prompt

#     async def get_response(self, agent_name, user_text, sec_result, file_path=None):
#         """
#         Sistemin kalbi. Girdiyi işler, ajanı seçer ve cevabı üretir.
#         """
#         clean_input = self.nlp.clean_text(user_text)
#         status, user_obj, sub_status = sec_result
        
#         # --- 1. GÜVENLİK KİLİDİ ---
#         # Eğer güvenlik durumu "ONAYLI" veya "SES_ONAYLI" değilse Kerberos devreye girer.
#         if status not in ["ONAYLI", "SES_ONAYLI"]:
#             agent_name = "KERBEROS"
            
#             # Senaryo A: Yüz yok / Ses Yok / Kamera Kapalı
#             if sub_status == "KAMERA_YOK":
#                 return {
#                     "agent": "KERBEROS", 
#                     "content": "KİMLİK TESPİT EDİLEMEDİ! Güvenlik protokolü gereği yüzünüzü görmeden veya sesinizi tanımadan işlem yapamam."
#                 }
            
#             # Senaryo B: Yabancı (Tanışma Modu)
#             print(f"🔒 Yabancı Tespit Edildi (Tanışma Modu)")

#         # --- 2. KARŞILAMA/RESET MODU ---
#         confirmation_words = ["baktim", "baktım", "geldim", "buradayim", "buradayım", "actim", "açtım", "gördün mü"]
#         is_short_confirmation = len(clean_input.split()) <= 3 and any(w in clean_input for w in confirmation_words)
        
#         if status in ["ONAYLI", "SES_ONAYLI"] and is_short_confirmation:
#             msg = f"Sizi net görüyorum {user_obj['name']}" if status == "ONAYLI" else f"Sesinizden tanıdım {user_obj['name']}"
#             return {
#                 "agent": agent_name, 
#                 "content": f"{msg}, tekrar hoş geldiniz. Kaldığımız yerden devam edebiliriz."
#             }

#         # --- STANDART İŞLEYİŞ ---
        
#         # 3. Sosyal Medya Mesajı Kontrolü
#         if any(sm in user_text for sm in ["WHATSAPP MESAJI", "INSTAGRAM MESAJI", "FACEBOOK MESAJI"]):
#             agent_name = "GAYA"

#         # 4. Dosya Hazırlığı
#         gemini_file_part = None
#         if file_path:
#             gemini_file_part, err = self._load_file_for_gemini(file_path)
#             if gemini_file_part: 
#                 user_text += f" [DOSYA EKLENDİ: {os.path.basename(file_path)}]"
#             else: 
#                 print(f"Dosya hatası: {err}")

#         # 5. Fatura Modu
#         fatura_triggers = ["fatura", "fiş", "hesap", "dekont"]
#         is_invoice = any(t in clean_input for t in fatura_triggers) and ("oku" in clean_input or "işle" in clean_input or "bak" in clean_input)
#         if file_path and any(t in clean_input for t in fatura_triggers): 
#             is_invoice = True

#         op_result = None
        
#         # --- FATURA İŞLEME MANTIĞI (KERBEROS ENTEGRASYONU) ---
#         if is_invoice:
#             # Eğer halihazırda Kerberos ile konuşuluyorsa veya genel komutsa
#             if agent_name == "KERBEROS" or agent_name == "ATLAS":
#                 agent_name = "KERBEROS" # İşi Kerberos'a ver
#             else:
#                 agent_name = "GAYA" # Diğer durumlarda Gaya baksın

#             if Config.AI_PROVIDER == "gemini":
#                 if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
#                     try:
#                         print("📸 Fatura için kamera açılıyor...")
#                         frame = self.tools['camera'].get_frame()
#                         if frame is not None:
#                             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                             img_byte_arr = io.BytesIO()
#                             pil_image.save(img_byte_arr, format='JPEG')
#                             gemini_file_part = {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
#                     except Exception as e:
#                         print(f"Kamera hatası: {e}")
                
#                 if gemini_file_part:
#                     prompt = "Bu fatura/fiş görselinden SADECE JSON verisi üret. Format: { \"firma\": \"...\", \"toplam_tutar\": \"...\", \"urunler\": [] }"
#                     json_resp = await self._query_gemini(agent_name, prompt, [], "Analiz et", gemini_file_part)
#                     invoice_data = self._extract_json_from_text(json_resp["content"])
                    
#                     # Hangi ajan seçildiyse onun fonksiyonunu çağır
#                     if agent_name == "KERBEROS":
#                         op_result = self.kerberos.audit_invoice(invoice_data)
#                     else:
#                         op_result = self.gaya.process_invoice_result(invoice_data)
#                 else:
#                     op_result = "Fatura okumak için dosya yüklemeli veya kameraya göstermelisin."
#             else:
#                 op_result = "Fatura okuma için Online Mod (Gemini) gerekli."

#         # 6. Genel Vision
#         if not op_result and Config.AI_PROVIDER == "gemini" and not gemini_file_part:
#             vision_triggers = ["bak", "gör", "nedir", "bu ne", "fotoğraf", "kamerayı aç"]
#             if any(k in clean_input for k in vision_triggers):
#                 if 'camera' in self.tools and CV2_AVAILABLE:
#                     try:
#                         frame = self.tools['camera'].get_frame()
#                         if frame is not None:
#                             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                             img_byte_arr = io.BytesIO()
#                             pil_image.save(img_byte_arr, format='JPEG')
#                             gemini_file_part = {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
#                     except: pass

#         # 7. Operasyonel İşlemler
#         if not op_result:
#             if agent_name == "GAYA" and "rezervasyon" in clean_input:
#                 user_name = sec_result[1]["name"] if sec_result[1] else "Misafir"
#                 op_result = self.gaya.handle_reservation(clean_input, user_name)
#             elif agent_name == "GAYA" and "panelleri aç" in clean_input and 'media' in self.tools:
#                  op_result = self.tools['media'].open_delivery_panels()
            
#             # YENİ: POYRAZ ANALİZ TETİKLEME
#             elif agent_name == "POYRAZ" and self.poyraz and ("analiz" in clean_input or "yorum" in clean_input or "durum" in clean_input):
#                 op_result = self.poyraz.analyze_platform_reviews("Google Maps & Yemeksepeti")

#             # YENİ: SİDAR ARŞİV TETİKLEME
#             elif agent_name == "SİDAR" and self.sidar and ("arşiv" in clean_input or "belge" in clean_input or "tara" in clean_input):
#                  op_result = self.sidar.scan_new_documents()

#         # 8. Cevap Üretimi
#         sys_prompt = self._prepare_context(agent_name, clean_input, sec_result, None, op_result)
        
#         try: 
#             recent, _ = self.memory.load_context(agent_name, clean_input)
#         except: 
#             recent = []

#         if Config.AI_PROVIDER == "gemini":
#             return await self._query_gemini(agent_name, sys_prompt, recent, user_text, gemini_file_part)
#         else:
#             return await self._query_ollama(agent_name, sys_prompt, recent, user_text)

#     async def get_team_response(self, user_text, sec_result):
#         """
#         Tüm ekibin kısa cevap vermesini sağlar.
#         """
#         agents_to_respond = ["ATLAS", "GAYA", "KURT", "POYRAZ", "SİDAR", "KERBEROS"]
        
#         # Güvenlik Kilidi Takım İçin de Geçerli
#         if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
#              return [{"agent": "KERBEROS", "content": "KİMLİK TESPİT EDİLEMEDİ! Lütfen kameraya bakın."}]

#         tasks = []
#         for agent in agents_to_respond:
#             if agent not in AGENTS_CONFIG: continue
            
#             sys_prompt = self._prepare_context(agent, user_text, sec_result)
#             sys_prompt += "\n\n[GÖREV]: Kullanıcı TÜM EKİBE seslendi. SADECE 1 CÜMLE ile karakterine uygun cevap ver."
            
#             if Config.AI_PROVIDER == "gemini":
#                 tasks.append(self._query_gemini(agent, sys_prompt, [], user_text))
#             else:
#                 tasks.append(self._query_ollama(agent, sys_prompt, [], user_text))
            
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         valid_results = []
#         for res in results:
#             if isinstance(res, dict) and 'content' in res:
#                 valid_results.append(res)
#             elif isinstance(res, Exception):
#                 print(f"Takım cevabında hata: {res}")
                
#         return valid_results

#     async def _query_gemini(self, agent, sys_prompt, history, user_text, image_data=None, retries=2):
#         """Gemini API sorgusu (Yeniden deneme mekanizmalı)."""
#         for attempt in range(retries + 1):
#             try:
#                 api_key = Config.GEMINI_KEYS.get(agent) or Config.GEMINI_KEYS.get("ATLAS")
#                 if not api_key: 
#                     return {"agent": agent, "content": "HATA: Gemini API Anahtarı eksik."}
                
#                 genai.configure(api_key=api_key)
                
#                 gemini_history = [{"role": "user" if i["role"]=="user" else "model", "parts": [i["content"]]} for i in history]
                
#                 model = genai.GenerativeModel(model_name=Config.GEMINI_MODEL, system_instruction=sys_prompt)
#                 chat = model.start_chat(history=gemini_history)
                
#                 parts = [user_text]
#                 if image_data: parts.append(image_data)
                
#                 resp = await asyncio.to_thread(chat.send_message, parts)
#                 reply_text = resp.text.strip()
                
#                 self.memory.save(agent, "user", user_text)
#                 self.memory.save(agent, "model", reply_text)
                
#                 return {"agent": agent, "content": reply_text}
                
#             except Exception as e:
#                 err_str = str(e)
#                 if attempt < retries:
#                     print(f"⚠️ {agent} API Hatası (Deneme {attempt+1}): {err_str} - Tekrar deneniyor...")
#                     await asyncio.sleep(1) 
#                 else:
#                     print(f"❌ {agent} API Hatası: {err_str}")
#                     return {"agent": agent, "content": "Bağlantıda sorun yaşıyorum, tekrar dener misin?"}

#     async def _query_ollama(self, agent, sys_prompt, history, user_text):
#         """Yerel LLM (Ollama) ile konuşma."""
#         msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_text}]
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(Config.OLLAMA_URL, json={"model": Config.TEXT_MODEL, "messages": msgs, "stream": False}, timeout=90) as resp:
#                     if resp.status == 200:
#                         res = await resp.json()
#                         reply = res.get("message", {}).get("content", "")
                        
#                         self.memory.save(agent, "user", user_text)
#                         self.memory.save(agent, "assistant", reply)
                        
#                         return {"agent": agent, "content": reply}
                    
#                     return {"agent": agent, "content": f"Yerel model hatası (Kod: {resp.status})."}
#         except Exception as e:
#             return {"agent": agent, "content": f"Ollama sunucusuna ulaşılamadı: {e}"}
        
        


# import aiohttp
# import asyncio
# import io
# import json
# import os
# import re
# import time
# from datetime import datetime
# from PIL import Image
# import google.generativeai as genai

# # OpenCV (Kamera) Modülü Kontrollü İçe Aktarma
# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     print("⚠️ UYARI: 'opencv-python' yüklü değil, kamera fonksiyonları çalışmayabilir.")

# # --- Konfigürasyon ve Tanımlar ---
# from agents.definitions import AGENTS_CONFIG
# from config import Config
# from managers.nlp import NLPManager 

# # --- AJAN MODÜLLERİ ---
# try:
#     from agents.atlas import AtlasAgent
#     from agents.gaya import GayaAgent
#     from agents.poyraz import PoyrazAgent
#     from agents.kurt import KurtAgent
#     from agents.sidar import SidarAgent
#     from agents.kerberos import KerberosAgent
# except ImportError as e:
#     print(f"KRİTİK HATA: Ajan dosyaları (agents/...) eksik veya hatalı: {e}")

# class AgentEngine:
#     """
#     Lotus AI'ın Karar ve Cevap Üretim Motoru.
#     Tüm ajanların beyin takımıdır. Özel işleri ilgili ajan modülüne devreder.
#     API (Gemini/Ollama) iletişimini ve Prompt mühendisliğini burası yönetir.
#     """
#     def __init__(self, memory_manager, tools_dict):
#         self.memory = memory_manager
#         self.tools = tools_dict
#         self.nlp = NLPManager()
        
#         # --- Ajanları Başlat ---
#         self.atlas = AtlasAgent(memory_manager)
#         self.gaya = GayaAgent(tools_dict, self.nlp)
#         self.poyraz = PoyrazAgent(tools_dict)
#         self.kurt = KurtAgent(tools_dict)
#         self.sidar = SidarAgent(tools_dict)
#         self.kerberos = KerberosAgent(tools_dict)
        
#         # Medya Yöneticisi kontrolü
#         if 'media' not in self.tools:
#             try:
#                 from managers.media import MediaManager
#                 self.tools['media'] = MediaManager()
#             except ImportError: 
#                 pass 
#             except Exception as e:
#                 print(f"⚠️ MediaManager başlatılamadı: {e}")

#     def determine_agent(self, text):
#         """Kullanıcı girdisine göre hangi ajanın devreye gireceğini belirler."""
#         if not text:
#             return None
        
#         clean_text = self.nlp.clean_text(text)
        
#         # Tanımlardaki anahtar kelimeleri tara
#         for name, data in AGENTS_CONFIG.items():
#             triggers = data.get("wake_words", []) + data.get("keys", [])
#             # Basit eşleşme kontrolü
#             if any(k in clean_text for k in triggers):
#                 return name
#         return None

#     def _read_user_bio(self, bio_filename):
#         """Kullanıcı biyografisini okur (Kişiselleştirilmiş deneyim için)."""
#         content = "Kullanıcı hakkında biyografik bilgi bulunamadı."
        
#         if bio_filename and os.path.exists(bio_filename):
#             try:
#                 with open(bio_filename, "r", encoding="utf-8") as f: 
#                     return f.read()
#             except: pass
        
#         default_bio = os.path.join(Config.WORK_DIR, "halil_bio.txt")
#         if os.path.exists(default_bio):
#             try:
#                 with open(default_bio, "r", encoding="utf-8") as f: 
#                     content = f.read()
#             except: pass
            
#         return content

#     def _load_file_for_gemini(self, file_path):
#         """Dosyayı Gemini API formatına hazırlar."""
#         if not file_path or not os.path.exists(file_path): 
#             return None, "Dosya bulunamadı."
        
#         low = file_path.lower()
#         mime_type = "application/octet-stream"
        
#         if low.endswith(".png"): mime_type = "image/png"
#         elif low.endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"
#         elif low.endswith(".pdf"): mime_type = "application/pdf"
#         elif low.endswith(".txt"): mime_type = "text/plain"
#         elif low.endswith(".json"): mime_type = "application/json"
        
#         try:
#             with open(file_path, "rb") as f: 
#                 file_data = f.read()
#             return {"mime_type": mime_type, "data": file_data}, None
#         except Exception as e: 
#             return None, str(e)

#     def _extract_json_from_text(self, text):
#         """AI yanıtından JSON bloğunu ayıklar."""
#         try:
#             text = text.strip()
#             if "```" in text:
#                 parts = text.split("```")
#                 for part in parts:
#                     if "{" in part: 
#                         text = part.replace("json", "").strip()
#                         break
            
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             return json.loads(match.group()) if match else json.loads(text)
#         except: 
#             return None

#     def _build_core_prompt(self, agent_name, user_text, sec_result, vision_desc=None, research_data=None):
#         """
#         Prompt iskeletini oluşturur.
#         Tüm bağlamı (Context) burada birleştirir.
#         """
#         status_code, user_obj, msg = sec_result
#         user_name = user_obj["name"] if user_obj else "Yabancı"
        
#         # Eğer güvenlik ihlali varsa Biyo okuma
#         bio_content = ""
#         if status_code == "ONAYLI":
#             bio_content = self._read_user_bio(user_obj.get("bio_file") if user_obj else "halil_bio.txt")
        
#         menu_summary = "Menü verisi yok."
#         if 'operations' in self.tools:
#             menu_summary = self.tools['operations'].get_context_summary()

#         locale_months = {
#             1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
#             7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
#         }
#         now = datetime.now()
#         time_str = f"{now.day} {locale_months.get(now.month)} {now.year}, Saat: {now.strftime('%H:%M')}"
#         team_str = ", ".join([n for n in AGENTS_CONFIG.keys() if n != agent_name])
        
#         base_sys = AGENTS_CONFIG.get(agent_name, {}).get('sys', "Sen yardımcı bir yapay zekasın.")
        
#         # --- PROMPT İNŞASI ---
#         prompt_parts = [
#             f"### SİSTEM KİMLİĞİ VE ROLÜN ###",
#             f"{base_sys}",
            
#             f"\n### ORTAM VE BAĞLAM ###",
#             f"Tarih/Saat: {time_str}",
#             f"Kullanıcı: {user_name} (Durum: {status_code})",
#             f"Diğer Ajanlar: {team_str}",
#         ]

#         if status_code == "ONAYLI":
#             prompt_parts.append(f"\n### İŞLETME BİLGİLERİ ###\nŞirket: Nilüfer Bağevi\nHizmetler: {menu_summary}")
#             if bio_content: 
#                 prompt_parts.append(f"\n### PATRON HAKKINDA ###\n{bio_content[:1500]}...")

#         # --- AJANA ÖZEL BAĞLAM (Dynamic Context) ---
#         ctx = ""
#         try:
#             if agent_name == "ATLAS": ctx = self.atlas.get_context_data()
#             elif agent_name == "GAYA": ctx = self.gaya.get_context_data(user_text)
#             elif agent_name == "POYRAZ": ctx = self.poyraz.get_context_data()
#             elif agent_name == "KURT": ctx = self.kurt.get_context_data()
#             elif agent_name == "SİDAR": ctx = self.sidar.get_context_data()
#             elif agent_name == "KERBEROS": ctx = self.kerberos.get_context_data()
#         except Exception as e:
#             print(f"Bağlam hatası ({agent_name}): {e}")
        
#         if ctx: prompt_parts.append(ctx)

#         if research_data: prompt_parts.append(f"\n### ARAŞTIRMA/VERİ SONUCU ###\n{research_data}")
#         if vision_desc: prompt_parts.append(f"\n### GÖRSEL ALGISI ###\n{vision_desc}")
        
#         prompt_parts.append(f"\n### KULLANICI MESAJI ###\n'{user_text}'")
        
#         return "\n".join(prompt_parts)

#     def _prepare_context(self, agent_name, user_text, sec_result, vision_desc=None, research_data=None):
#         core_prompt = self._build_core_prompt(agent_name, user_text, sec_result, vision_desc, research_data)
#         if Config.AI_PROVIDER != "gemini":
#             core_prompt += f"\n\nÖNEMLİ: Asla 'User:', 'System:' etiketleri kullanma. Sadece {agent_name} rolünde konuş."
#         return core_prompt

#     async def get_response(self, agent_name, user_text, sec_result, file_path=None):
#         """
#         Sistemin kalbi. Girdiyi işler, ajanı seçer ve cevabı üretir.
#         """
#         clean_input = self.nlp.clean_text(user_text)
        
#         # --- GÜVENLİK VE KİMLİK KONTROLÜ (OVERRIDE) ---
#         # Eğer güvenlik durumu "ONAYLI" veya "SES_ONAYLI" değilse
#         # Hangi ajanı isterse istesin (Atlas vb.) devreye KERBEROS girer.
#         if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
#             agent_name = "KERBEROS"
#             # Kullanıcının ne dediğini Kerberos'a iletme, ona talimat ver
#             original_text = user_text
#             user_text = (
#                 f"[SİSTEM UYARISI]: Görüntüde yüz algılanamadı veya tanınamadı. "
#                 f"Kullanıcı şunu sordu: '{original_text}'. "
#                 f"BUNU CEVAPLAMA. Sert bir dille kimlik sor, kameraya bakmasını söyle veya yüzünü göster de."
#             )
#             print(f"🔒 Güvenlik Kilidi Devrede: Ajan {agent_name} olarak değiştirildi.")

#         # 1. Sosyal Medya Mesajı Kontrolü
#         if any(sm in user_text for sm in ["WHATSAPP MESAJI", "INSTAGRAM MESAJI", "FACEBOOK MESAJI"]):
#             agent_name = "GAYA"

#         # 2. Dosya Hazırlığı
#         gemini_file_part = None
#         if file_path:
#             gemini_file_part, err = self._load_file_for_gemini(file_path)
#             if gemini_file_part: 
#                 user_text += f" [DOSYA EKLENDİ: {os.path.basename(file_path)}]"
#             else: 
#                 print(f"Dosya hatası: {err}")

#         # 3. Fatura Modu (Gaya ve Kerberos için)
#         fatura_triggers = ["fatura", "fiş", "hesap"]
#         is_invoice = any(t in clean_input for t in fatura_triggers) and ("oku" in clean_input or "işle" in clean_input)
#         if file_path and any(t in clean_input for t in fatura_triggers): 
#             is_invoice = True

#         op_result = None
        
#         if (agent_name == "GAYA" or agent_name == "KERBEROS") and is_invoice:
#             agent_name = "GAYA" # Faturaları Gaya işler
#             if Config.AI_PROVIDER == "gemini":
#                 if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
#                     try:
#                         print("📸 Fatura için kamera açılıyor...")
#                         frame = self.tools['camera'].get_frame()
#                         if frame is not None:
#                             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                             img_byte_arr = io.BytesIO()
#                             pil_image.save(img_byte_arr, format='JPEG')
#                             gemini_file_part = {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
#                     except Exception as e:
#                         print(f"Kamera hatası: {e}")
                
#                 if gemini_file_part:
#                     prompt = "Bu fatura/fiş görselinden SADECE JSON verisi üret. Format: { \"firma\": \"...\", \"toplam_tutar\": \"...\", \"urunler\": [] }"
#                     json_resp = await self._query_gemini("GAYA", prompt, [], "Analiz et", gemini_file_part)
#                     invoice_data = self._extract_json_from_text(json_resp["content"])
                    
#                     op_result = self.gaya.process_invoice_result(invoice_data)
#                 else:
#                     op_result = "Fatura okumak için dosya yüklemeli veya kameraya göstermelisin."
#             else:
#                 op_result = "Fatura okuma için Online Mod (Gemini) gerekli."

#         # 4. Genel Vision
#         if not op_result and Config.AI_PROVIDER == "gemini" and not gemini_file_part:
#             vision_triggers = ["bak", "gör", "nedir", "bu ne", "fotoğraf", "kamerayı aç"]
#             if any(k in clean_input for k in vision_triggers):
#                 if 'camera' in self.tools and CV2_AVAILABLE:
#                     try:
#                         frame = self.tools['camera'].get_frame()
#                         if frame is not None:
#                             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                             img_byte_arr = io.BytesIO()
#                             pil_image.save(img_byte_arr, format='JPEG')
#                             gemini_file_part = {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
#                     except: pass

#         # 5. Operasyonel İşlemler
#         if not op_result:
#             if agent_name == "GAYA" and "rezervasyon" in clean_input:
#                 user_name = sec_result[1]["name"] if sec_result[1] else "Misafir"
#                 op_result = self.gaya.handle_reservation(clean_input, user_name)
            
#             elif agent_name == "GAYA" and "panelleri aç" in clean_input and 'media' in self.tools:
#                  op_result = self.tools['media'].open_delivery_panels()

#         # 6. Cevap Üretimi
#         sys_prompt = self._prepare_context(agent_name, clean_input, sec_result, None, op_result)
        
#         try: 
#             recent, _ = self.memory.load_context(agent_name, clean_input)
#         except: 
#             recent = []

#         if Config.AI_PROVIDER == "gemini":
#             return await self._query_gemini(agent_name, sys_prompt, recent, user_text, gemini_file_part)
#         else:
#             return await self._query_ollama(agent_name, sys_prompt, recent, user_text)

#     async def get_team_response(self, user_text, sec_result):
#         """
#         Tüm ekibin kısa cevap vermesini sağlar.
#         """
#         agents_to_respond = ["ATLAS", "GAYA", "KURT", "POYRAZ", "SİDAR", "KERBEROS"]
        
#         # Güvenlik Kilidi Takım İçin de Geçerli
#         if sec_result[0] not in ["ONAYLI", "SES_ONAYLI"]:
#              return [{"agent": "KERBEROS", "content": "KİMLİK TESPİT EDİLEMEDİ! Lütfen kameraya bakın."}]

#         tasks = []
#         for agent in agents_to_respond:
#             if agent not in AGENTS_CONFIG: continue
            
#             sys_prompt = self._prepare_context(agent, user_text, sec_result)
#             sys_prompt += "\n\n[GÖREV]: Kullanıcı TÜM EKİBE seslendi. SADECE 1 CÜMLE ile karakterine uygun cevap ver."
            
#             if Config.AI_PROVIDER == "gemini":
#                 tasks.append(self._query_gemini(agent, sys_prompt, [], user_text))
#             else:
#                 tasks.append(self._query_ollama(agent, sys_prompt, [], user_text))
            
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         valid_results = []
#         for res in results:
#             if isinstance(res, dict) and 'content' in res:
#                 valid_results.append(res)
#             elif isinstance(res, Exception):
#                 print(f"Takım cevabında hata: {res}")
                
#         return valid_results

#     async def _query_gemini(self, agent, sys_prompt, history, user_text, image_data=None, retries=2):
#         """Gemini API sorgusu (Yeniden deneme mekanizmalı)."""
#         for attempt in range(retries + 1):
#             try:
#                 api_key = Config.GEMINI_KEYS.get(agent) or Config.GEMINI_KEYS.get("ATLAS")
#                 if not api_key: 
#                     return {"agent": agent, "content": "HATA: Gemini API Anahtarı eksik."}
                
#                 genai.configure(api_key=api_key)
                
#                 gemini_history = [{"role": "user" if i["role"]=="user" else "model", "parts": [i["content"]]} for i in history]
                
#                 model = genai.GenerativeModel(model_name=Config.GEMINI_MODEL, system_instruction=sys_prompt)
#                 chat = model.start_chat(history=gemini_history)
                
#                 parts = [user_text]
#                 if image_data: parts.append(image_data)
                
#                 resp = await asyncio.to_thread(chat.send_message, parts)
#                 reply_text = resp.text.strip()
                
#                 self.memory.save(agent, "user", user_text)
#                 self.memory.save(agent, "model", reply_text)
                
#                 return {"agent": agent, "content": reply_text}
                
#             except Exception as e:
#                 err_str = str(e)
#                 if attempt < retries:
#                     print(f"⚠️ {agent} API Hatası (Deneme {attempt+1}): {err_str} - Tekrar deneniyor...")
#                     await asyncio.sleep(1) 
#                 else:
#                     print(f"❌ {agent} API Hatası: {err_str}")
#                     return {"agent": agent, "content": "Bağlantıda sorun yaşıyorum, tekrar dener misin?"}

#     async def _query_ollama(self, agent, sys_prompt, history, user_text):
#         """Yerel LLM (Ollama) ile konuşma."""
#         msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_text}]
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(Config.OLLAMA_URL, json={"model": Config.TEXT_MODEL, "messages": msgs, "stream": False}, timeout=90) as resp:
#                     if resp.status == 200:
#                         res = await resp.json()
#                         reply = res.get("message", {}).get("content", "")
                        
#                         self.memory.save(agent, "user", user_text)
#                         self.memory.save(agent, "assistant", reply)
                        
#                         return {"agent": agent, "content": reply}
                    
#                     return {"agent": agent, "content": f"Yerel model hatası (Kod: {resp.status})."}
#         except Exception as e:
#             return {"agent": agent, "content": f"Ollama sunucusuna ulaşılamadı: {e}"}