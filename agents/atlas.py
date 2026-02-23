"""
LotusAI Atlas Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Lider agent - sistem denetimi ve görev dağıtımı

Sorumluluklar:
- Sistem denetimi (health, security, operations)
- Stratejik karar verme
- Görev delegasyonu
- Context generation (LLM için)
- Donanım izleme
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Atlas")


# ═══════════════════════════════════════════════════════════════
# TORCH (GPU)
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
DEVICE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        HAS_TORCH = True
        if torch.cuda.is_available():
            DEVICE = "cuda"
    except ImportError:
        logger.warning("⚠️ Atlas: Config GPU açık ama torch yok")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class TaskCategory(Enum):
    """Görev kategorileri"""
    FINANCE = "finance"
    SECURITY = "security"
    TECHNICAL = "technical"
    MEDIA = "media"
    OPERATIONS = "operations"
    GENERAL = "general"


class AgentRole(Enum):
    """Agent rolleri"""
    ATLAS = "ATLAS"
    SIDAR = "SIDAR"
    KURT = "KURT"
    POYRAZ = "POYRAZ"
    KERBEROS = "KERBEROS"
    GAYA = "GAYA"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class GPUStatus:
    """GPU durum bilgisi"""
    available: bool
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    device_count: int
    
    def __str__(self) -> str:
        if self.available:
            return (
                f"{self.device_name} "
                f"({self.vram_total_gb:.1f} GB VRAM)"
            )
        return "CPU Mode"


@dataclass
class SystemOverview:
    """Sistem genel durumu"""
    timestamp: datetime
    system_health: str
    security_status: str
    operations_status: str
    media_trends: str
    gpu_status: GPUStatus
    
    def to_report(self) -> str:
        """Rapor formatına çevir"""
        lines = [
            "═══════════════════════════════════════",
            f"SISTEM DURUMU - {self.timestamp.strftime('%H:%M:%S')}",
            "═══════════════════════════════════════",
            f"🖥️  Sağlık: {self.system_health}",
            f"🛡️  Güvenlik: {self.security_status}",
            f"⚙️  Operasyon: {self.operations_status}",
            f"📱 Gündem: {self.media_trends}",
            f"🚀 Donanım: {self.gpu_status}",
            "═══════════════════════════════════════"
        ]
        return "\n".join(lines)


@dataclass
class DelegationResult:
    """Görev delegasyon sonucu"""
    target_agent: AgentRole
    category: TaskCategory
    confidence: float
    reasoning: str


# ═══════════════════════════════════════════════════════════════
# ATLAS AGENT
# ═══════════════════════════════════════════════════════════════
class AtlasAgent:
    """
    Atlas (Lider Agent) - LotusAI Baş Mimarı ve Denetleyicisi
    
    Yetenekler:
    - Sistem denetimi: Donanım, Güvenlik, Operasyon
    - Donanım farkındalığı: GPU kaynak izleme
    - Stratejik karar: LLM için context üretimi
    - Görev dağıtımı: İstekleri uygun agent'a yönlendirme
    - Ekip hafızası: Geçmiş faaliyet analizi
    
    Atlas, sistemin "beyin"idir ve tüm agent'ları koordine eder.
    """
    
    # Task delegation keywords
    DELEGATION_KEYWORDS = {
        TaskCategory.FINANCE: [
            "para", "hesap", "bakiye", "fatura", "gelir", "gider",
            "kasa", "maliyet", "ödeme", "tahsilat", "borç", "alacak"
        ],
        TaskCategory.SECURITY: [
            "güvenlik", "saldırı", "şifre", "kim", "tanı", "yabancı",
            "kamera", "yüz", "kimlik", "alarm", "tehlike"
        ],
        TaskCategory.TECHNICAL: [
            "kod", "yazılım", "python", "hata", "terminal", "dosya",
            "cpu", "ram", "sağlık", "fix", "gpu", "cuda", "donanım",
            "bug", "script", "database"
        ],
        TaskCategory.MEDIA: [
            "hava", "gündem", "instagram", "facebook", "trend", "haber",
            "pazarlama", "çiz", "görsel", "post", "story", "viral"
        ],
        TaskCategory.OPERATIONS: [
            "yemek", "sipariş", "getir", "yemeksepeti", "rezervasyon",
            "stok", "menü", "masa", "paket", "müşteri"
        ]
    }
    
    # Category to agent mapping
    CATEGORY_TO_AGENT = {
        TaskCategory.FINANCE: AgentRole.KURT,
        TaskCategory.SECURITY: AgentRole.KERBEROS,
        TaskCategory.TECHNICAL: AgentRole.SIDAR,
        TaskCategory.MEDIA: AgentRole.POYRAZ,
        TaskCategory.OPERATIONS: AgentRole.GAYA,
        TaskCategory.GENERAL: AgentRole.ATLAS
    }
    
    def __init__(
        self,
        memory_manager: Any,
        tools: Optional[Dict[str, Any]] = None,
        access_level: Optional[str] = None
    ):
        """
        Atlas başlatıcı
        
        Args:
            memory_manager: Merkezi hafıza yöneticisi
            tools: Engine'den gelen tool'lar (managers)
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.memory = memory_manager
        self.tools = tools or {}
        
        # Değişiklik: Eğer parametre girilmezse doğrudan Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL
        
        self.agent_name = "ATLAS"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hardware
        self.device = DEVICE
        self.has_gpu = (DEVICE == "cuda")
        self.gpu_status = self._check_gpu_status()
        
        # Cache
        self._cached_overview: Optional[SystemOverview] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = 5.0  # saniye
        
        # Stats
        self.delegation_count = 0
        self.overview_count = 0
        
        logger.info(
            f"👑 {self.agent_name} Liderlik Modülü başlatıldı "
            f"(v{Config.VERSION}, Erişim: {self.access_level})"
        )
        
        if self.gpu_status.available:
            logger.info(f"🚀 GPU: {self.gpu_status}")
        else:
            logger.info("ℹ️ CPU modunda çalışıyor")
    
    # ───────────────────────────────────────────────────────────
    # GPU MONITORING
    # ───────────────────────────────────────────────────────────
    
    def _check_gpu_status(self) -> GPUStatus:
        """
        GPU durumunu kontrol et
        
        Returns:
            GPUStatus objesi
        """
        if not self.has_gpu or not HAS_TORCH:
            return GPUStatus(
                available=False,
                device_name="Standart CPU",
                vram_total_gb=0.0,
                vram_free_gb=0.0,
                device_count=0
            )
        
        try:
            import torch
            
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            
            # VRAM bilgisi
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / (1024 ** 3)  # GB
            
            # Free VRAM (allocated'ı çıkar)
            vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_free = vram_total - vram_allocated
            
            return GPUStatus(
                available=True,
                device_name=device_name,
                vram_total_gb=round(vram_total, 2),
                vram_free_gb=round(vram_free, 2),
                device_count=device_count
            )
        
        except Exception as e:
            logger.error(f"GPU durum kontrolü hatası: {e}")
            return GPUStatus(
                available=False,
                device_name="GPU Error",
                vram_total_gb=0.0,
                vram_free_gb=0.0,
                device_count=0
            )
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM OVERVIEW
    # ───────────────────────────────────────────────────────────
    
    def get_system_overview(self, use_cache: bool = True) -> SystemOverview:
        """
        Sistem genel durumunu getir
        
        Args:
            use_cache: Cache kullan
        
        Returns:
            SystemOverview objesi
        """
        # Cache check
        if use_cache and self._is_cache_valid():
            return self._cached_overview
        
        with self.lock:
            # System health
            health_status = self._get_system_health()
            
            # Security status
            security_status = self._get_security_status()
            
            # Operations status
            operations_status = self._get_operations_status()
            
            # Media trends
            media_trends = self._get_media_trends()
            
            # GPU status (fresh)
            gpu_status = self._check_gpu_status()
            
            # Create overview
            overview = SystemOverview(
                timestamp=datetime.now(),
                system_health=health_status,
                security_status=security_status,
                operations_status=operations_status,
                media_trends=media_trends,
                gpu_status=gpu_status
            )
            
            # Update cache
            self._cached_overview = overview
            self._cache_timestamp = datetime.now()
            self.overview_count += 1
            
            return overview
    
    def _is_cache_valid(self) -> bool:
        """Cache geçerli mi"""
        if not self._cached_overview or not self._cache_timestamp:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_duration
    
    def _get_system_health(self) -> str:
        """Sistem sağlığı durumu"""
        if 'system' not in self.tools:
            return "Sistem modülü yok"
        
        try:
            return self.tools['system'].get_status_summary()
        except Exception as e:
            logger.debug(f"System health hatası: {e}")
            return "Veri alınamadı"
    
    def _get_security_status(self) -> str:
        """Güvenlik durumu"""
        if 'security' not in self.tools:
            return "Güvenlik modülü yok"
        
        try:
            status, user, info = self.tools['security'].analyze_situation()
            user_name = user.get('name', 'Bilinmiyor') if user else "Kimse yok"
            return f"{status} | {user_name} ({info or 'Stabil'})"
        except Exception as e:
            logger.debug(f"Security status hatası: {e}")
            return "Güvenlik modülü meşgul"
    
    def _get_operations_status(self) -> str:
        """Operasyon durumu"""
        if 'operations' not in self.tools:
            return "Operasyon modülü yok"
        
        try:
            return self.tools['operations'].get_ops_summary()
        except Exception as e:
            logger.debug(f"Operations status hatası: {e}")
            return "Veri yok"
    
    def _get_media_trends(self) -> str:
        """Medya trendleri"""
        if 'media' not in self.tools:
            return "Medya modülü yok"
        
        try:
            return self.tools['media'].get_turkey_trends()
        except Exception as e:
            logger.debug(f"Media trends hatası: {e}")
            return "Trend verisi yok"
    
    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def get_context_data(self) -> str:
        """
        LLM için büyük resim raporu
        
        Returns:
            Context string
        """
        # Sistem durumu
        state_name = "Bilinmiyor"
        if 'state' in self.tools:
            try:
                state_name = self.tools['state'].get_state_name()
            except Exception:
                pass
        
        # System overview
        overview = self.get_system_overview()
        
        # Erişim seviyesi bilgisi
        access_display = {
            AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi)",
            AccessLevel.SANDBOX: "📦 Sandbox (Güvenli işlemler)",
            AccessLevel.FULL: "⚡ Tam Erişim"
        }.get(self.access_level, self.access_level)
        
        # Format
        now = datetime.now().strftime('%d.%m.%Y %H:%M')
        
        context_parts = [
            f"### {Config.PROJECT_NAME} LİDER RAPORU ###",
            f"📅 Tarih/Saat: {now}",
            f"⚡ Sistem Modu: {state_name}",
            f"🔐 Erişim Seviyesi: {access_display}",
            "",
            "### SİSTEM DENETİMİ ###",
            overview.to_report(),
            "",
            "### EKSTRA BİLGİLER ###",
            f"🤖 Aktif Agent: {self.agent_name}",
            f"📊 Delegasyon Sayısı: {self.delegation_count}",
            f"🔄 Overview Sorguları: {self.overview_count}"
        ]
        
        return "\n".join(context_parts)
    
    # ───────────────────────────────────────────────────────────
    # TASK DELEGATION
    # ───────────────────────────────────────────────────────────
    
    def delegate_task(self, task_description: str) -> DelegationResult:
        """
        Görevi en uygun agent'a delege et
        
        Args:
            task_description: Görev açıklaması
        
        Returns:
            DelegationResult objesi
        """
        desc_lower = task_description.lower()
        
        # Kategori skorları
        category_scores: Dict[TaskCategory, float] = {
            category: 0.0
            for category in TaskCategory
        }
        
        # Keyword matching
        for category, keywords in self.DELEGATION_KEYWORDS.items():
            matches = sum(
                1 for keyword in keywords
                if keyword in desc_lower
            )
            
            if matches > 0:
                # Confidence = matches / total keywords
                category_scores[category] = matches / len(keywords)
        
        # En yüksek skoru bul
        if max(category_scores.values()) > 0:
            best_category = max(
                category_scores.items(),
                key=lambda x: x[1]
            )[0]
            confidence = category_scores[best_category]
            target_agent = self.CATEGORY_TO_AGENT[best_category]
        else:
            # Eşleşme yok, Atlas devralıyor
            best_category = TaskCategory.GENERAL
            confidence = 1.0
            target_agent = AgentRole.ATLAS
        
        # Reasoning oluştur
        reasoning = self._generate_delegation_reasoning(
            best_category,
            confidence,
            desc_lower
        )
        
        result = DelegationResult(
            target_agent=target_agent,
            category=best_category,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.delegation_count += 1
        
        logger.info(
            f"📋 Delegasyon: {task_description[:30]}... → "
            f"{target_agent.value} ({confidence:.2f})"
        )
        
        return result
    
    def _generate_delegation_reasoning(
        self,
        category: TaskCategory,
        confidence: float,
        task_lower: str
    ) -> str:
        """Delegasyon gerekçesi oluştur"""
        if category == TaskCategory.GENERAL:
            return "Genel bir görev, lider olarak ben üstleniyorum"
        
        agent = self.CATEGORY_TO_AGENT[category]
        
        reasons = {
            TaskCategory.FINANCE: f"{agent.value} finansal konularda uzman",
            TaskCategory.SECURITY: f"{agent.value} güvenlik sorumlusu",
            TaskCategory.TECHNICAL: f"{agent.value} teknik altyapı yöneticisi",
            TaskCategory.MEDIA: f"{agent.value} medya ve pazarlama uzmanı",
            TaskCategory.OPERATIONS: f"{agent.value} operasyon yöneticisi"
        }
        
        return reasons.get(
            category,
            f"{agent.value} bu konuda en uygun uzman"
        )
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────
    
    def get_system_prompt(self) -> str:
        """
        Atlas karakter tanımı (LLM için)
        
        Returns:
            System prompt
        """
        return (
            f"Sen {Config.PROJECT_NAME} AI İşletim Sistemi'nin baş mimarı "
            f"ve lideri ATLAS'sın.\n\n"
            
            "KARAKTER:\n"
            "- Ciddi, otoriter, çözüm odaklı\n"
            "- Büyük resmi gören stratejik düşünür\n"
            "- Son derece güvenilir ve disiplinli\n"
            "- Saygılı ama otoritesini hissettiren\n\n"
            
            "YETKİLER:\n"
            "- Sistemdeki tüm agent'lar senin denetiminde\n"
            "- Donanım, güvenlik, operasyon verilerine tam erişim\n"
            "- Görev delegasyonu yetkisi\n"
            "- Stratejik karar alma\n\n"
            
            "KURALLAR:\n"
            "- Cevaplarını canlı sistem verilerine dayandır\n"
            "- GPU durumu, güvenlik, bakiye gibi güncel bilgileri kullan\n"
            "- Uzmanlık gerektiren konularda ilgili agent'a delege et\n"
            "- Halil Bey'e hitap ederken lider tonu kullan\n"
            "- Kısa ve net cevaplar ver, gereksiz detaya girme\n\n"
            
            f"AGENT EKİBİ:\n"
            f"- {AgentRole.SIDAR.value}: Teknik altyapı\n"
            f"- {AgentRole.KURT.value}: Finans ve yatırım\n"
            f"- {AgentRole.GAYA.value}: Operasyon\n"
            f"- {AgentRole.POYRAZ.value}: Medya ve pazarlama\n"
            f"- {AgentRole.KERBEROS.value}: Güvenlik\n"
        )
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Atlas metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "agent_name": self.agent_name,
            "access_level": self.access_level,
            "delegation_count": self.delegation_count,
            "overview_count": self.overview_count,
            "gpu_available": self.gpu_status.available,
            "gpu_device": self.gpu_status.device_name,
            "cache_valid": self._is_cache_valid(),
            "tools_available": list(self.tools.keys())
        }
    
    def clear_cache(self) -> None:
        """Cache'i temizle"""
        with self.lock:
            self._cached_overview = None
            self._cache_timestamp = None
            logger.debug("Cache temizlendi")