"""
LotusAI Agent Engine - Multi-Agent Koordinasyon Motoru
Sürüm: 2.5.3
Açıklama: Agent seçimi, LLM iletişimi, görsel analiz ve dinamik yanıt üretimi
"""

import aiohttp
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# GOOGLE AI SDK
# ═══════════════════════════════════════════════════════════════
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("⚠️ google-generativeai yüklü değil, Gemini devre dışı")

# ═══════════════════════════════════════════════════════════════
# GÖRÜNTÜ İŞLEME
# ═══════════════════════════════════════════════════════════════
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("⚠️ Pillow yüklü değil, görüntü işleme kısıtlı")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("⚠️ opencv-python yüklü değil, kamera desteği yok")

# ═══════════════════════════════════════════════════════════════
# GPU DESTEĞI
# ═══════════════════════════════════════════════════════════════
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# CORE MODÜLLER
# ═══════════════════════════════════════════════════════════════
from config import Config

logger = logging.getLogger("LotusAI.Engine")


# ═══════════════════════════════════════════════════════════════
# SABITLER
# ═══════════════════════════════════════════════════════════════
class EngineConfig:
    """Engine çalışma parametreleri"""
    # Retry ayarları
    MAX_RETRIES: int = 5
    INITIAL_BACKOFF: float = 2.0
    MAX_BACKOFF: float = 32.0
    
    # Timeout'lar
    OLLAMA_TIMEOUT: int = 120
    GEMINI_TIMEOUT: int = 60
    
    # Metin limitleri
    BIO_MAX_CHARS: int = 3000
    CONTEXT_MAX_ITEMS: int = 10
    
    # Dosya tipleri
    SUPPORTED_IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp'}
    SUPPORTED_DOC_TYPES = {'.pdf', '.txt'}
    
    # Prompt temaları
    WELCOME_KEYWORDS = ["selam", "merhaba", "geldim", "buradayım", "hey lotus"]
    VISUAL_TRIGGERS = ["fatura", "fiş", "dekont", "hesap", "oku", "işle", 
                      "ne yazıyor", "analiz et", "göster"]


# ═══════════════════════════════════════════════════════════════
# YARDIMCI SINIFLAR
# ═══════════════════════════════════════════════════════════════
@dataclass
class AgentResponse:
    """Agent yanıt yapısı"""
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SecurityStatus(Enum):
    """Güvenlik durumları"""
    APPROVED = "ONAYLI"
    VOICE_APPROVED = "SES_ONAYLI"
    PENDING = "BEKLEMEDE"
    DENIED = "REDDEDİLDİ"
    NO_CAMERA = "KAMERA_YOK"
    INTRO_MODE = "TANIŞMA_MODU"


# ═══════════════════════════════════════════════════════════════
# AGENT YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════════
class AgentLoader:
    """Agent modüllerini dinamik olarak yükler"""
    
    _loaded_agents: Dict[str, Any] = {}
    _load_status: Dict[str, bool] = {}
    
    AGENT_MAP = {
        "ATLAS": ("atlas", "AtlasAgent"),
        "GAYA": ("gaya", "GayaAgent"),
        "POYRAZ": ("poyraz", "PoyrazAgent"),
        "KURT": ("kurt", "KurtAgent"),
        "SIDAR": ("sidar", "SidarAgent"),
        "KERBEROS": ("kerberos", "KerberosAgent")
    }
    
    @classmethod
    def load_agent(cls, agent_name: str) -> Optional[type]:
        """
        Agent class'ını yükle
        
        Args:
            agent_name: Agent adı (ATLAS, GAYA, vb.)
        
        Returns:
            Agent class veya None
        """
        if agent_name in cls._loaded_agents:
            return cls._loaded_agents[agent_name]
        
        if agent_name not in cls.AGENT_MAP:
            logger.warning(f"Bilinmeyen agent: {agent_name}")
            return None
        
        module_name, class_name = cls.AGENT_MAP[agent_name]
        
        try:
            module = __import__(f"agents.{module_name}", fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            cls._loaded_agents[agent_name] = agent_class
            cls._load_status[agent_name] = True
            logger.info(f"✅ {agent_name} agent yüklendi")
            
            return agent_class
        
        except (ImportError, AttributeError) as e:
            logger.debug(f"Agent yüklenemedi ({agent_name}): {e}")
            cls._load_status[agent_name] = False
            return None
    
    @classmethod
    def load_all(cls) -> Dict[str, bool]:
        """Tüm agent'ları yükle ve durum döndür"""
        for agent_name in cls.AGENT_MAP.keys():
            cls.load_agent(agent_name)
        return cls._load_status.copy()
    
    @classmethod
    def is_loaded(cls, agent_name: str) -> bool:
        """Agent yüklü mü kontrol et"""
        return cls._load_status.get(agent_name, False)


# ═══════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════
try:
    from agents.definitions import AGENTS_CONFIG
except ImportError:
    AGENTS_CONFIG = {}
    logger.warning("⚠️ agents/definitions.py bulunamadı, varsayılan config kullanılıyor")


# ═══════════════════════════════════════════════════════════════
# NLP MANAGER
# ═══════════════════════════════════════════════════════════════
try:
    from managers.nlp import NLPManager
    NLP_AVAILABLE = True
except ImportError:
    NLPManager = None
    NLP_AVAILABLE = False
    logger.warning("⚠️ NLPManager bulunamadı")


# ═══════════════════════════════════════════════════════════════
# ANA ENGINE SINIFI
# ═══════════════════════════════════════════════════════════════
class AgentEngine:
    """
    LotusAI Multi-Agent Koordinasyon Motoru
    
    Sorumluluklar:
    - Agent seçimi ve yönlendirme
    - LLM iletişimi (Gemini/Ollama)
    - Görsel analiz ve dosya işleme
    - Dinamik prompt oluşturma
    - Memory management
    - Context oluşturma
    """
    
    def __init__(self, memory_manager: Any, tools_dict: Dict[str, Any]):
        """
        Engine başlatıcı
        
        Args:
            memory_manager: Hafıza yöneticisi
            tools_dict: Manager ve araç dictionary'si
        """
        self.memory = memory_manager
        self.tools = tools_dict
        self.app_id = Config.PROJECT_NAME.lower().replace(" ", "-")
        
        # GPU durumu
        self.device = "cuda" if (Config.USE_GPU and TORCH_AVAILABLE) else "cpu"
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.info(f"🚀 Engine GPU aktif: {Config.GPU_INFO}")
            except Exception as e:
                logger.warning(f"GPU temizleme hatası: {e}")
                self.device = "cpu"
        
        # NLP Manager
        self.nlp: Optional[Any] = None
        if NLP_AVAILABLE:
            self.nlp = tools_dict.get('nlp') or NLPManager()
        
        # Agent'ları yükle
        self._initialize_agents()
        
        logger.info(f"✅ AgentEngine hazır (Device: {self.device.upper()})")
    
    def _initialize_agents(self) -> None:
        """Tüm agent'ları başlat"""
        # Agent loader'ı kullan
        AgentLoader.load_all()
        
        # Agent instance'larını oluştur
        self.agents: Dict[str, Any] = {}
        
        # ATLAS
        if AgentLoader.is_loaded("ATLAS"):
            AtlasAgent = AgentLoader.load_agent("ATLAS")
            self.agents["ATLAS"] = AtlasAgent(self.memory, self.tools)
        
        # GAYA
        if AgentLoader.is_loaded("GAYA"):
            GayaAgent = AgentLoader.load_agent("GAYA")
            self.agents["GAYA"] = GayaAgent(self.tools, self.nlp)
        
        # POYRAZ (özel durum - tools'dan gelebilir)
        if "poyraz_special" in self.tools:
            self.agents["POYRAZ"] = self.tools["poyraz_special"]
        elif AgentLoader.is_loaded("POYRAZ"):
            PoyrazAgent = AgentLoader.load_agent("POYRAZ")
            self.agents["POYRAZ"] = PoyrazAgent(self.nlp, self.tools)
        
        # KURT
        if AgentLoader.is_loaded("KURT"):
            KurtAgent = AgentLoader.load_agent("KURT")
            self.agents["KURT"] = KurtAgent(self.tools)
        
        # SIDAR (özel durum - tools'dan gelebilir)
        if "sidar_special" in self.tools:
            self.agents["SIDAR"] = self.tools["sidar_special"]
        elif AgentLoader.is_loaded("SIDAR"):
            SidarAgent = AgentLoader.load_agent("SIDAR")
            sidar_tools = {
                k: self.tools.get(k) 
                for k in ['code', 'system', 'security', 'memory']
            }
            self.agents["SIDAR"] = SidarAgent(sidar_tools)
        
        # KERBEROS
        if AgentLoader.is_loaded("KERBEROS"):
            KerberosAgent = AgentLoader.load_agent("KERBEROS")
            self.agents["KERBEROS"] = KerberosAgent(self.tools)
        
        logger.info(f"Aktif agent'lar: {', '.join(self.agents.keys())}")
    
    def determine_agent(self, text: str) -> Optional[str]:
        """
        Kullanıcı girdisine göre en uygun agent'ı seç
        
        Args:
            text: Kullanıcı metni
        
        Returns:
            Agent adı veya None
        """
        if not text:
            return "ATLAS"
        
        # Metni temizle
        clean_text = self.nlp.clean_text(text.lower()) if self.nlp else text.lower()
        
        # Öncelikli kontroller
        priority_checks = {
            "SIDAR": ["sistem", "kod", "hata", "terminal", "log", "debug"],
            "KERBEROS": ["kimsin", "yetki", "güvenlik", "kilit", "doğrula", "tanı"]
        }
        
        for agent, keywords in priority_checks.items():
            if any(k in clean_text for k in keywords):
                if agent in self.agents:
                    return agent
        
        # AGENTS_CONFIG'den kontrol
        for agent_name, agent_data in AGENTS_CONFIG.items():
            if agent_name not in self.agents:
                continue
            
            triggers = agent_data.get("wake_words", []) + agent_data.get("keys", [])
            if any(k.lower() in clean_text for k in triggers):
                return agent_name
        
        return "ATLAS"
    
    def _read_user_bio(self, user_obj: Optional[Dict] = None) -> str:
        """
        Kullanıcı biyografisini oku
        
        Args:
            user_obj: Kullanıcı bilgileri
        
        Returns:
            Bio metni veya boş string
        """
        bio_file = user_obj.get("bio_file", "halil_bio.txt") if user_obj else "halil_bio.txt"
        bio_path = Config.WORK_DIR / bio_file
        
        # Fallback
        if not bio_path.exists():
            bio_path = Config.WORK_DIR / "halil_bio.txt"
        
        if bio_path.exists():
            try:
                content = bio_path.read_text(encoding="utf-8")
                return content[:EngineConfig.BIO_MAX_CHARS]
            except Exception as e:
                logger.error(f"Bio okuma hatası: {e}")
        
        return ""
    
    def _load_file_for_gemini(
        self, 
        file_path: Union[str, Path]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Dosyayı Gemini multimodal formatına hazırla
        
        Args:
            file_path: Dosya yolu
        
        Returns:
            Tuple[file data dict, error message]
        """
        path = Path(file_path)
        
        if not path.exists():
            return None, "Dosya bulunamadı"
        
        # MIME type belirleme
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".txt": "text/plain"
        }
        
        mime_type = mime_map.get(ext, "application/octet-stream")
        
        try:
            data = path.read_bytes()
            return {"mime_type": mime_type, "data": data}, None
        
        except Exception as e:
            return None, f"Dosya okuma hatası: {str(e)}"
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        LLM çıktısından JSON çıkar
        
        Args:
            text: LLM yanıtı
        
        Returns:
            JSON dict veya None
        """
        if not text:
            return None
        
        try:
            # 1. Markdown kod bloğu
            json_block = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```',
                text,
                re.DOTALL | re.IGNORECASE
            )
            
            if json_block:
                return json.loads(json_block.group(1))
            
            # 2. Düz JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return None
        
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse hatası: {e}")
            return None
        
        except Exception as e:
            logger.error(f"JSON extraction hatası: {e}")
            return None
    
    def _build_core_prompt(
        self,
        agent_name: str,
        user_text: str,
        sec_result: Tuple[str, Optional[Dict], str],
        op_result: Optional[str] = None
    ) -> str:
        """
        Agent için sistem prompt'u oluştur
        
        Args:
            agent_name: Agent adı
            user_text: Kullanıcı mesajı
            sec_result: Güvenlik sonucu (status, user_obj, sub_status)
            op_result: Operasyonel sonuç (opsiyonel)
        
        Returns:
            Sistem prompt'u
        """
        status_code, user_obj, sub_status = sec_result
        user_name = user_obj.get("name", "Misafir") if user_obj else "Misafir"
        
        # Agent tanımı
        agent_def = AGENTS_CONFIG.get(agent_name, {})
        base_sys = agent_def.get('sys', "Yardımcı bir yapay zeka sistemsin.")
        
        # Bio içeriği
        bio_content = ""
        if status_code in ["ONAYLI", "SES_ONAYLI"]:
            bio_content = self._read_user_bio(user_obj)
        
        # Zaman bilgisi
        time_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        
        # Diğer agent'lar
        team_list = [name for name in self.agents.keys() if name != agent_name]
        team_str = ", ".join(team_list) if team_list else "Yalnız"
        
        # Prompt oluştur
        sections = [
            "### KİMLİK VE ROL ###",
            f"Sen LotusAI işletim sisteminin **{agent_name}** isimli uzman ajanısın.",
            f"Görev Tanımın: {base_sys}",
            "",
            "### ORTAM BİLGİLERİ ###",
            f"Tarih/Saat: {time_str}",
            f"Kullanıcı: {user_name}",
            f"Güvenlik Durumu: {status_code}",
            f"Diğer Aktif Ajanlar: {team_str}",
            f"Donanım: {self.device.upper()}"
        ]
        
        # Güvenlik uyarısı
        if sub_status == "TANIŞMA_MODU":
            sections.extend([
                "",
                "⚠️ GÜVENLİK PROTOKOLÜ:",
                "Kullanıcı henüz tam doğrulanmadı. Sadece tanışma ve temel bilgilendirme yap.",
                "Sistem yetkilerini kullandırma."
            ])
        
        # Agent context
        agent_instance = self.agents.get(agent_name)
        if agent_instance and hasattr(agent_instance, "get_context_data"):
            try:
                if agent_name == "GAYA":
                    context = agent_instance.get_context_data(user_text)
                else:
                    context = agent_instance.get_context_data()
                
                if context:
                    sections.extend([
                        "",
                        "### CANLI VERİ BAĞLAMI ###",
                        context
                    ])
            
            except Exception as e:
                logger.error(f"Context alma hatası ({agent_name}): {e}")
        
        # Operasyonel sonuç
        if op_result:
            sections.extend([
                "",
                "### OPERASYONEL ANALİZ SONUCU ###",
                "Sistem bu görevi önceden işledi, sonucu yanıtına dahil et:",
                op_result
            ])
        
        # Bio
        if bio_content:
            sections.extend([
                "",
                "### KULLANICI HAKKINDA ÖZEL BİLGİLER ###",
                bio_content
            ])
        
        # Stil notu
        sections.extend([
            "",
            "### YANIT STİLİ ###",
            "Profesyonel, net ve LotusAI kimliğine uygun konuş.",
            "Gereksiz giriş cümlelerinden kaçın."
        ])
        
        return "\n".join(sections)
    
    async def _handle_visual_tasks(
        self,
        clean_input: str,
        file_path: Optional[str],
        agent_name: str
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Görsel analiz ve belge işleme
        
        Args:
            clean_input: Temizlenmiş kullanıcı metni
            file_path: Dosya yolu (opsiyonel)
            agent_name: Mevcut agent
        
        Returns:
            Tuple[gemini file part, operation result]
        """
        gemini_file_part = None
        op_result = None
        
        # Dosya yükle
        if file_path:
            gemini_file_part, error = self._load_file_for_gemini(file_path)
            if error:
                logger.warning(f"Dosya yükleme hatası: {error}")
        
        # Görsel istek kontrolü
        is_visual_request = (
            any(trigger in clean_input for trigger in EngineConfig.VISUAL_TRIGGERS) or
            file_path is not None
        )
        
        if not is_visual_request or Config.AI_PROVIDER != "gemini":
            return gemini_file_part, op_result
        
        # Target agent belirle
        target_agent = "KERBEROS" if agent_name in ["KERBEROS", "ATLAS"] else "GAYA"
        
        # Kameradan görüntü al (dosya yoksa)
        if not gemini_file_part and 'camera' in self.tools and CV2_AVAILABLE:
            frame = self.tools['camera'].get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                gemini_file_part = {
                    "mime_type": "image/jpeg",
                    "data": buffer.tobytes()
                }
        
        # Görsel analiz
        if gemini_file_part:
            analysis_prompt = (
                "Bu görseli detaylıca analiz et. "
                "Eğer fatura/finansal belge ise şu JSON formatında çıkar: "
                "{ 'firma': '', 'toplam_tutar': '', 'tarih': '', 'urunler': [], 'is_invoice': true }. "
                "Değilse 'description' anahtarıyla görseli açıkla."
            )
            
            response = await self._query_gemini(
                target_agent,
                "Görsel analiz uzmanısın.",
                [],
                analysis_prompt,
                gemini_file_part
            )
            
            analysis_data = self._extract_json_from_text(response["content"])
            
            if analysis_data:
                agent_instance = self.agents.get(target_agent)
                
                if analysis_data.get('is_invoice'):
                    # Fatura işleme
                    if target_agent == "KERBEROS" and hasattr(agent_instance, "audit_invoice"):
                        op_result = agent_instance.audit_invoice(analysis_data)
                    elif target_agent == "GAYA" and hasattr(agent_instance, "process_invoice_result"):
                        op_result = agent_instance.process_invoice_result(analysis_data)
                else:
                    # Genel görsel
                    description = analysis_data.get('description', 'Analiz yapılamadı')
                    op_result = f"Görsel Analizi: {description}"
        
        return gemini_file_part, op_result
    
    async def get_response(
        self,
        agent_name: str,
        user_text: str,
        sec_result: Tuple[str, Optional[Dict], str],
        file_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Kullanıcı mesajına yanıt üret (Ana giriş noktası)
        
        Args:
            agent_name: Seçilen agent
            user_text: Kullanıcı mesajı
            sec_result: Güvenlik sonucu
            file_path: Dosya yolu (opsiyonel)
        
        Returns:
            Agent yanıtı dict
        """
        # Metni temizle
        clean_input = self.nlp.clean_text(user_text) if self.nlp else user_text.lower()
        
        status, user_obj, sub_status = sec_result
        
        # 1. GÜVENLİK KONTROLÜ
        if status not in ["ONAYLI", "SES_ONAYLI"]:
            agent_name = "KERBEROS"
            
            if sub_status == "KAMERA_YOK":
                return {
                    "agent": "KERBEROS",
                    "content": (
                        "Güvenlik protokolü gereği sizi tanımam gerekiyor. "
                        "Lütfen kameranızı aktive edin veya sesli doğrulama yapın."
                    )
                }
        
        # 2. SELAMLAŞMA KONTROLÜ
        if status in ["ONAYLI", "SES_ONAYLI"] and len(clean_input.split()) <= 3:
            if any(word in clean_input for word in EngineConfig.WELCOME_KEYWORDS):
                name = user_obj.get("name", "Halil Bey") if user_obj else "Halil Bey"
                return {
                    "agent": agent_name,
                    "content": (
                        f"Hoş geldiniz {name}. LotusAI {self.device.upper()} "
                        f"destekli sistemleriyle aktif. Size nasıl yardımcı olabilirim?"
                    )
                }
        
        # 3. GÖRSEL/DOSYA İŞLEME
        gemini_file_part, op_result = await self._handle_visual_tasks(
            clean_input,
            file_path,
            agent_name
        )
        
        # 4. AGENT ÖZEL FONKSİYON
        if not op_result:
            agent_instance = self.agents.get(agent_name)
            if agent_instance and hasattr(agent_instance, "auto_handle"):
                try:
                    op_result = await agent_instance.auto_handle(clean_input)
                except Exception as e:
                    logger.error(f"Agent auto_handle hatası ({agent_name}): {e}")
        
        # 5. LLM YANIT ÜRETİMİ
        sys_prompt = self._build_core_prompt(agent_name, clean_input, sec_result, op_result)
        
        # Memory'den geçmiş al
        try:
            history, _, _ = self.memory.load_context(
                agent_name,
                clean_input,
                max_items=EngineConfig.CONTEXT_MAX_ITEMS
            )
        except Exception as e:
            logger.warning(f"Memory yükleme hatası: {e}")
            history = []
        
        # Provider'a göre query
        if Config.AI_PROVIDER == "gemini" and GENAI_AVAILABLE:
            return await self._query_gemini(
                agent_name,
                sys_prompt,
                history,
                user_text,
                gemini_file_part
            )
        else:
            return await self._query_ollama(
                agent_name,
                sys_prompt,
                history,
                user_text
            )
    
    async def get_team_response(
        self,
        user_text: str,
        sec_result: Tuple[str, Optional[Dict], str]
    ) -> List[Dict[str, str]]:
        """
        Tüm ekipten brifing al
        
        Args:
            user_text: Kullanıcı sorusu
            sec_result: Güvenlik sonucu
        
        Returns:
            Agent yanıtları listesi
        """
        status, _, _ = sec_result
        
        if status not in ["ONAYLI", "SES_ONAYLI"]:
            return [{
                "agent": "KERBEROS",
                "content": "Güvenlik onayı yetersiz, ekip brifingi başlatılamaz."
            }]
        
        # Tüm agent'lar için task oluştur
        tasks = []
        
        for agent_name in self.agents.keys():
            sys_prompt = self._build_core_prompt(
                agent_name,
                user_text,
                sec_result
            ) + "\n\n[GÖREV]: Konu hakkında kendi uzmanlık alanından tek cümlelik kısa brifing ver."
            
            if Config.AI_PROVIDER == "gemini" and GENAI_AVAILABLE:
                task = self._query_gemini(agent_name, sys_prompt, [], user_text)
            else:
                task = self._query_ollama(agent_name, sys_prompt, [], user_text)
            
            tasks.append(task)
        
        # Paralel çalıştır
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hataları filtrele
        return [r for r in results if isinstance(r, dict)]
    
    async def _query_gemini(
        self,
        agent: str,
        sys_prompt: str,
        history: List[Dict],
        user_text: str,
        image_data: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Gemini API'ye istek gönder (Exponential backoff)
        
        Args:
            agent: Agent adı
            sys_prompt: Sistem prompt'u
            history: Sohbet geçmişi
            user_text: Kullanıcı mesajı
            image_data: Görsel data (opsiyonel)
        
        Returns:
            Agent yanıtı
        """
        if not GENAI_AVAILABLE:
            return {
                "agent": agent,
                "content": "⚠️ Gemini kütüphanesi yüklü değil"
            }
        
        # API Key al
        agent_settings = Config.get_agent_settings(agent)
        api_key = agent_settings.get("key")
        model_name = agent_settings.get("model", Config.GEMINI_MODEL_DEFAULT)
        
        if not api_key:
            return {
                "agent": agent,
                "content": "⚠️ API anahtarı yapılandırılmamış"
            }
        
        # Gemini yapılandır
        genai.configure(api_key=api_key)
        
        # History formatla
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Retry döngüsü
        for attempt in range(EngineConfig.MAX_RETRIES):
            try:
                # Model oluştur
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=sys_prompt
                )
                
                # İçerik hazırla
                contents = [user_text]
                if image_data:
                    contents.append(image_data)
                
                # Chat başlat
                chat = model.start_chat(history=gemini_history)
                
                # Yanıt al (async thread)
                response = await asyncio.to_thread(
                    chat.send_message,
                    contents
                )
                
                reply = response.text.strip()
                
                # Memory'ye kaydet
                if self.memory:
                    self.memory.save(agent, "user", user_text)
                    self.memory.save(agent, "model", reply)
                
                return {"agent": agent, "content": reply}
            
            except Exception as e:
                wait_time = min(
                    EngineConfig.INITIAL_BACKOFF * (2 ** attempt),
                    EngineConfig.MAX_BACKOFF
                )
                
                logger.warning(
                    f"Gemini deneme {attempt + 1}/{EngineConfig.MAX_RETRIES} "
                    f"başarısız: {e}"
                )
                
                if attempt == EngineConfig.MAX_RETRIES - 1:
                    logger.error(f"Gemini kritik hatası ({agent}): {e}")
                    return {
                        "agent": agent,
                        "content": (
                            "Şu an merkezi sinir sistemime (Gemini) ulaşamıyorum. "
                            "Lütfen kısa süre sonra tekrar deneyin."
                        )
                    }
                
                await asyncio.sleep(wait_time)
        
        # Buraya normalde gelmemeli
        return {"agent": agent, "content": "Beklenmeyen hata"}
    
    async def _query_ollama(
        self,
        agent: str,
        sys_prompt: str,
        history: List[Dict],
        user_text: str
    ) -> Dict[str, str]:
        """
        Ollama API'ye istek gönder
        
        Args:
            agent: Agent adı
            sys_prompt: Sistem prompt'u
            history: Sohbet geçmişi
            user_text: Kullanıcı mesajı
        
        Returns:
            Agent yanıtı
        """
        # Mesaj listesi oluştur
        messages = [
            {"role": "system", "content": sys_prompt}
        ] + history + [
            {"role": "user", "content": user_text}
        ]
        
        # Retry döngüsü
        for attempt in range(2):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        Config.OLLAMA_URL,
                        json={
                            "model": Config.TEXT_MODEL,
                            "messages": messages,
                            "stream": False
                        },
                        timeout=EngineConfig.OLLAMA_TIMEOUT
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            reply = data.get("message", {}).get("content", "").strip()
                            
                            # Memory'ye kaydet
                            if self.memory:
                                self.memory.save(agent, "user", user_text)
                                self.memory.save(agent, "assistant", reply)
                            
                            return {"agent": agent, "content": reply}
                        
                        else:
                            logger.error(f"Ollama HTTP hatası: {response.status}")
            
            except asyncio.TimeoutError:
                logger.error("Ollama timeout")
            
            except Exception as e:
                logger.error(f"Ollama bağlantı hatası: {e}")
            
            if attempt == 0:
                await asyncio.sleep(2)
        
        # Tüm denemeler başarısız
        return {
            "agent": agent,
            "content": (
                "LotusAI yerel sunucusu (Ollama) şu an kapalı veya yanıt vermiyor. "
                "Terminal'de 'ollama serve' komutunu çalıştırın."
            )
        }

