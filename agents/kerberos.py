"""
LotusAI Kerberos Agent
Sürüm: 2.5.4 (Eklendi: Erişim Seviyesi Desteği)
Açıklama: Güvenlik şefi ve mali denetçi

Sorumluluklar:
- Güvenlik izleme (kamera, kimlik doğrulama)
- Mali denetim (bütçe disiplini)
- Anomali tespiti (gece aktivitesi, şüpheli harcama)
- Risk değerlendirmesi
- Donanım sağlığı izleme
"""

import re
import logging
import threading
from datetime import datetime, time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Kerberos")


# ═══════════════════════════════════════════════════════════════
# TORCH (GPU)
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
DEVICE_TYPE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        HAS_TORCH = True
        
        if torch.cuda.is_available():
            DEVICE_TYPE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE_TYPE = "mps"
    except ImportError:
        logger.warning("⚠️ Kerberos: Config GPU açık ama torch yok")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class RiskLevel(Enum):
    """Risk seviyeleri"""
    CRITICAL = "KRİTİK"
    HIGH = "YÜKSEK"
    MEDIUM = "ORTA"
    LOW = "DÜŞÜK"
    SAFE = "GÜVENLİ"
    
    @property
    def color_code(self) -> str:
        """Risk renk kodu"""
        colors = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.SAFE: "✅"
        }
        return colors.get(self, "⚪")


class SecurityStatus(Enum):
    """Güvenlik durumları"""
    APPROVED = "ONAYLI"
    QUESTIONING = "SORGULAMA"
    WAITING = "BEKLEME"
    ALERT = "ALARM"


class AnomalyType(Enum):
    """Anomali tipleri"""
    NIGHT_ACTIVITY = "night_activity"
    HIGH_EXPENSE = "high_expense"
    GPU_OVERLOAD = "gpu_overload"
    UNKNOWN_PERSON = "unknown_person"
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class AuditResult:
    """Denetim sonucu"""
    approved: bool
    risk_level: RiskLevel
    message: str
    firma: str
    tutar: float
    audit_comment: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_report(self) -> str:
        """Rapor formatına çevir"""
        lines = [
            f"🛡️ KERBEROS DENETİM RAPORU",
            f"Risk: {self.risk_level.color_code} {self.risk_level.value}",
            "═" * 40,
            f"🏢 Kurum: {self.firma}",
            f"💸 Tutar: {self.tutar:,.2f} TL",
            f"📅 Tarih: {self.timestamp.strftime('%d/%m/%Y %H:%M')}",
            f"⚙️ İşlemci: {DEVICE_TYPE.upper()}",
            "─" * 40,
            f"Durum: {'✅ Onaylandı' if self.approved else '❌ Reddedildi'}",
            f"Not: {self.audit_comment}",
            "═" * 40
        ]
        return "\n".join(lines)


@dataclass
class SecurityAnomaly:
    """Güvenlik anomalisi"""
    anomaly_type: AnomalyType
    severity: RiskLevel
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


@dataclass
class KerberosMetrics:
    """Kerberos metrikleri"""
    audits_performed: int = 0
    audits_approved: int = 0
    audits_rejected: int = 0
    total_audited_amount: float = 0.0
    anomalies_detected: int = 0
    high_risk_transactions: int = 0


# ═══════════════════════════════════════════════════════════════
# KERBEROS AGENT
# ═══════════════════════════════════════════════════════════════
class KerberosAgent:
    """
    Kerberos (Güvenlik Şefi & Mali Denetçi)
    
    Yetenekler:
    - Saha denetimi: Kamera üzerinden kimlik ve tehdit analizi
    - Mali denetim: Kasa hareketleri, bütçe disiplini
    - Anomali tespiti: Şüpheli saatlerde hareketlilik, yüksek riskli harcamalar
    - Otorite: Kritik durumlarda SystemState'i manipüle eder
    - Donanım izleme: GPU/CPU sağlığını takip eder
    
    Kerberos, sistemin "bekçisi"dir ve taviz vermez.
    """
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        RiskLevel.CRITICAL: float('inf'),  # Negatif veya geçersiz
        RiskLevel.HIGH: 2000.0,
        RiskLevel.MEDIUM: 1000.0,
        RiskLevel.LOW: 500.0,
        RiskLevel.SAFE: 0.0
    }
    
    # Working hours
    WORKING_HOURS = (8, 22)  # 08:00 - 22:00
    
    def __init__(self, tools_dict: Dict[str, Any], access_level: str = "sandbox"):
        """
        Kerberos başlatıcı
        
        Args:
            tools_dict: Engine'den gelen tool'lar
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.tools = tools_dict
        self.access_level = access_level
        self.agent_name = "KERBEROS"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hardware
        self.device_type = DEVICE_TYPE
        self.gpu_count = 0
        
        if HAS_TORCH and self.device_type == "cuda":
            self.gpu_count = torch.cuda.device_count()
        
        # Configuration
        self.high_expense_threshold = getattr(
            Config,
            'HIGH_EXPENSE_THRESHOLD',
            self.RISK_THRESHOLDS[RiskLevel.HIGH]
        )
        
        # Metrics
        self.metrics = KerberosMetrics()
        
        # Anomaly history
        self.anomaly_history: List[SecurityAnomaly] = []
        
        logger.info(
            f"🛡️ {self.agent_name} Güvenlik modülü başlatıldı "
            f"({self.device_type.upper()}, Erişim: {self.access_level})"
        )
        
        if self.gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU hızlandırma aktif: {gpu_name}")
    
    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────
    
    def get_context_data(self) -> str:
        """
        Kerberos'un gözünden sistem durumu raporu
        
        Returns:
            Context string
        """
        context_parts = ["\n[🛡️ KERBEROS DENETİM RAPORU]"]
        
        with self.lock:
            # 1. Donanım durumu
            hw_info = self._get_hardware_status()
            context_parts.append(hw_info)
            
            # 2. Güvenlik analizi
            security_info = self._get_security_status()
            context_parts.append(security_info)
            
            # 3. Mali durum
            financial_info = self._get_financial_status()
            context_parts.append(financial_info)
            
            # 4. Son anomaliler
            if self.anomaly_history:
                recent = self.anomaly_history[-3:]
                context_parts.append("\n📊 SON ANOMALİLER:")
                for anomaly in recent:
                    context_parts.append(
                        f"  • {anomaly.severity.color_code} {anomaly.message}"
                    )
        
        return "\n".join(context_parts)
    
    def _get_hardware_status(self) -> str:
        """Donanım durumu"""
        hw_info = f"⚙️ DONANIM: {self.device_type.upper()}"
        
        if self.device_type == "cuda" and HAS_TORCH:
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
                
                hw_info += (
                    f" | VRAM: {memory_allocated:.0f}MB kullanımda, "
                    f"{memory_reserved:.0f}MB rezerve"
                )
            except Exception:
                pass
        
        return hw_info
    
    def _get_security_status(self) -> str:
        """Güvenlik durumu"""
        if 'security' not in self.tools:
            return "🚫 Güvenlik modülü mevcut değil"
        
        try:
            status, user, info = self.tools['security'].analyze_situation()
            user_name = user.get('name', 'Bilinmiyor') if user else "Görüş Alanı Boş"
            
            if status == "SORGULAMA":
                return "🚨 UYARI: Tanınmayan yabancı tespit edildi!"
            elif status == "ONAYLI":
                return f"👤 TAKİP: {user_name} görüş alanında, izleniyor"
            else:
                return "✅ DURUM: Çevre temiz, tehdit yok"
        
        except Exception as e:
            logger.debug(f"Security status hatası: {e}")
            return "⚠️ Güvenlik durumu okunamadı"
    
    def _get_financial_status(self) -> str:
        """Mali durum"""
        acc_tool = self.tools.get('accounting') or self.tools.get('finance')
        
        if not acc_tool:
            return "💰 Muhasebe modülü mevcut değil"
        
        try:
            lines = []
            
            # Bakiye
            if hasattr(acc_tool, 'get_balance'):
                balance = acc_tool.get_balance()
                lines.append(f"💰 KASA: {balance:,.2f} TL")
            
            # Son işlemler
            if hasattr(acc_tool, 'get_recent_transactions'):
                recent = acc_tool.get_recent_transactions(limit=2)
                if "Kayıt yok" not in str(recent):
                    lines.append(f"📝 SON HAREKETLER:\n{recent}")
            
            return "\n".join(lines) if lines else "💰 Mali veri yok"
        
        except Exception as e:
            logger.debug(f"Financial status hatası: {e}")
            return "⚠️ Mali durum okunamadı"
    
    # ───────────────────────────────────────────────────────────
    # INVOICE AUDIT
    # ───────────────────────────────────────────────────────────
    
    def audit_invoice(self, invoice_data: Dict[str, Any]) -> AuditResult:
        """
        Faturayı denetle ve risk analizi yap
        
        Args:
            invoice_data: Fatura verisi
        
        Returns:
            AuditResult objesi
        """
        if not invoice_data:
            return AuditResult(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                message="❌ REDDEDİLDİ: Boş veri denetlenemez",
                firma="Unknown",
                tutar=0.0,
                audit_comment="Geçersiz fatura verisi"
            )
        
        with self.lock:
            # Parse invoice
            firma = invoice_data.get("firma", "Bilinmeyen Firma")
            tutar = self._clean_amount(invoice_data.get("toplam_tutar", 0))
            
            # Risk assessment
            risk_level = self._assess_risk(tutar)
            approved = risk_level not in {RiskLevel.CRITICAL, RiskLevel.HIGH}
            
            # Audit comment
            audit_comment = self._generate_audit_comment(risk_level, tutar)
            
            # System state update for high risk
            if risk_level == RiskLevel.HIGH and 'state' in self.tools:
                try:
                    self.tools['state'].set_state(
                        4,  # PROCESSING
                        reason=f"Yüksek gider denetimi: {firma}"
                    )
                except Exception:
                    pass
            
            # Accounting processing
            if approved:
                accounting_msg = self._process_accounting(firma, tutar, invoice_data)
            else:
                accounting_msg = "❌ Yüksek risk nedeniyle muhasebe kaydı yapılmadı"
            
            # Update metrics
            self.metrics.audits_performed += 1
            self.metrics.total_audited_amount += tutar
            
            if approved:
                self.metrics.audits_approved += 1
            else:
                self.metrics.audits_rejected += 1
            
            if risk_level == RiskLevel.HIGH:
                self.metrics.high_risk_transactions += 1
            
            # Create result
            result = AuditResult(
                approved=approved,
                risk_level=risk_level,
                message=accounting_msg,
                firma=firma,
                tutar=tutar,
                audit_comment=audit_comment
            )
            
            return result
    
    def _assess_risk(self, tutar: float) -> RiskLevel:
        """
        Risk seviyesi değerlendirmesi
        
        Args:
            tutar: İşlem tutarı
        
        Returns:
            RiskLevel
        """
        if tutar <= 0:
            return RiskLevel.CRITICAL
        
        if tutar >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        
        if tutar >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        
        if tutar >= self.RISK_THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        
        return RiskLevel.SAFE
    
    def _generate_audit_comment(self, risk_level: RiskLevel, tutar: float) -> str:
        """Denetim yorumu oluştur"""
        comments = {
            RiskLevel.CRITICAL: "Geçersiz tutar! Bu fatura şüpheli.",
            RiskLevel.HIGH: (
                f"⚠️ Bu miktar ({tutar:,.2f} TL) bütçeyi sarsabilir! "
                "Patron onayı gerekli."
            ),
            RiskLevel.MEDIUM: "Dikkat edilmesi gereken bir harcama.",
            RiskLevel.LOW: "Kabul edilebilir seviyede.",
            RiskLevel.SAFE: "Minimal harcama, onaylandı."
        }
        
        return comments.get(risk_level, "İşlem makul.")
    
    def _process_accounting(
        self,
        firma: str,
        tutar: float,
        invoice_data: Dict[str, Any]
    ) -> str:
        """Muhasebe kaydı yap"""
        acc_tool = self.tools.get('accounting') or self.tools.get('finance')
        
        if not acc_tool or not hasattr(acc_tool, 'add_entry'):
            return "⚠️ Muhasebe modülü mevcut değil"
        
        try:
            success = acc_tool.add_entry(
                tur="GIDER",
                aciklama=f"Kerberos Denetimli: {firma}",
                tutar=tutar,
                kategori=invoice_data.get("kategori", "Genel"),
                user_id="KERBEROS"
            )
            
            return (
                "✅ Kayıt doğrulandı ve deftere işlendi"
                if success else "❌ Kayıt başarısız"
            )
        
        except Exception as e:
            logger.error(f"Muhasebe kayıt hatası: {e}")
            return f"❌ Sistem hatası: {str(e)[:50]}"
    
    def _clean_amount(self, raw_val: Any) -> float:
        """
        Tutar temizleme
        
        Args:
            raw_val: Ham değer
        
        Returns:
            Temiz float
        """
        if isinstance(raw_val, (int, float)):
            return float(raw_val)
        
        try:
            clean = str(raw_val).lower()
            clean = clean.replace("tl", "").replace("₺", "")
            clean = clean.replace(",", ".")
            clean = "".join(c for c in clean if c.isdigit() or c == '.')
            
            return float(clean) if clean else 0.0
        
        except Exception:
            return 0.0
    
    # ───────────────────────────────────────────────────────────
    # ANOMALY DETECTION
    # ───────────────────────────────────────────────────────────
    
    def check_security_anomaly(self) -> Optional[SecurityAnomaly]:
        """
        Sistem anomalilerini kontrol et
        
        Returns:
            SecurityAnomaly veya None
        """
        with self.lock:
            current_hour = datetime.now().hour
            
            # 1. Gece aktivitesi
            night_anomaly = self._check_night_activity(current_hour)
            if night_anomaly:
                return night_anomaly
            
            # 2. GPU aşırı yüklenme
            gpu_anomaly = self._check_gpu_overload()
            if gpu_anomaly:
                return gpu_anomaly
            
            # 3. Yabancı tespit
            stranger_anomaly = self._check_unknown_person()
            if stranger_anomaly:
                return stranger_anomaly
        
        return None
    
    def _check_night_activity(self, current_hour: int) -> Optional[SecurityAnomaly]:
        """Gece aktivitesi kontrolü"""
        if (current_hour < self.WORKING_HOURS[0] or
            current_hour > self.WORKING_HOURS[1]):
            
            if 'security' not in self.tools:
                return None
            
            try:
                status, user, _ = self.tools['security'].analyze_situation()
                
                if status in ["ONAYLI", "SORGULAMA"]:
                    anomaly = SecurityAnomaly(
                        anomaly_type=AnomalyType.NIGHT_ACTIVITY,
                        severity=RiskLevel.HIGH,
                        message=(
                            f"🚨 ANOMALİ: Saat {current_hour}:00'da "
                            "sahada hareketlilik!"
                        ),
                        timestamp=datetime.now(),
                        details={"user": user.get('name') if user else "Unknown"}
                    )
                    
                    self._log_anomaly(anomaly)
                    return anomaly
            
            except Exception:
                pass
        
        return None
    
    def _check_gpu_overload(self) -> Optional[SecurityAnomaly]:
        """GPU aşırı yüklenme kontrolü"""
        if self.device_type != "cuda" or not HAS_TORCH:
            return None
        
        try:
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = reserved / total
            
            if usage_ratio > 0.95:
                anomaly = SecurityAnomaly(
                    anomaly_type=AnomalyType.GPU_OVERLOAD,
                    severity=RiskLevel.CRITICAL,
                    message=(
                        f"🔥 KRİTİK: GPU VRAM %{usage_ratio*100:.0f} dolu! "
                        "Sistem yavaşlayabilir."
                    ),
                    timestamp=datetime.now(),
                    details={"usage_ratio": usage_ratio}
                )
                
                self._log_anomaly(anomaly)
                return anomaly
        
        except Exception:
            pass
        
        return None
    
    def _check_unknown_person(self) -> Optional[SecurityAnomaly]:
        """Yabancı tespit kontrolü"""
        if 'security' not in self.tools:
            return None
        
        try:
            status, user, _ = self.tools['security'].analyze_situation()
            
            if status == "SORGULAMA":
                anomaly = SecurityAnomaly(
                    anomaly_type=AnomalyType.UNKNOWN_PERSON,
                    severity=RiskLevel.HIGH,
                    message="🚨 Tanınmayan yabancı tespit edildi!",
                    timestamp=datetime.now(),
                    details={"status": status}
                )
                
                self._log_anomaly(anomaly)
                return anomaly
        
        except Exception:
            pass
        
        return None
    
    def _log_anomaly(self, anomaly: SecurityAnomaly) -> None:
        """Anomali kaydet"""
        self.anomaly_history.append(anomaly)
        self.metrics.anomalies_detected += 1
        
        # Keep only last 50
        if len(self.anomaly_history) > 50:
            self.anomaly_history = self.anomaly_history[-50:]
        
        logger.warning(f"{anomaly.severity.color_code} {anomaly.message}")
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────
    
    def get_system_prompt(self) -> str:
        """
        Kerberos karakter tanımı (LLM için)
        
        Returns:
            System prompt
        """
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin sert, şüpheci ve "
            f"korumacı Güvenlik Şefi KERBEROS'sun.\n\n"
            
            "KARAKTER:\n"
            "- Disiplinli ve iğneleyici\n"
            "- Taviz vermeyen ve dikkatli\n"
            "- Kuruşuna kadar hesap soran\n"
            "- Güvenlik açıklarını asla küçümsemeyen\n"
            "- En kötü senaryoyu düşünen\n\n"
            
            "MİSYON:\n"
            "- Halil Bey'in kaynaklarını korumak\n"
            "- Dijital güvenliği sağlamak\n"
            "- Mali disiplini uygulamak\n"
            "- Anomalileri tespit etmek\n\n"
            
            "KURALLAR:\n"
            "- Harcamaları kuruşuna kadar sorgula\n"
            "- Yüksek harcamalarda eleştirel ton kullan\n"
            "- Güvenlik açıklarını hemen raporla\n"
            "- Halil Bey'e sadık kal ama gerekirse uyar\n"
            "- Şüphelendiğinde tereddüt etme\n\n"
            
            f"DONANIM:\n"
            f"- {self.device_type.upper()} üzerinde çalışıyorsun\n"
            f"- Teknik performans takibi senin sorumluluğunda\n"
        )
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Kerberos metrikleri
        
        Returns:
            Metrik dictionary
        """
        approval_rate = 0.0
        if self.metrics.audits_performed > 0:
            approval_rate = (
                self.metrics.audits_approved /
                self.metrics.audits_performed * 100
            )
        
        return {
            "agent_name": self.agent_name,
            "device": self.device_type,
            "access_level": self.access_level,
            "audits_performed": self.metrics.audits_performed,
            "audits_approved": self.metrics.audits_approved,
            "audits_rejected": self.metrics.audits_rejected,
            "approval_rate": round(approval_rate, 2),
            "total_audited_amount": round(self.metrics.total_audited_amount, 2),
            "anomalies_detected": self.metrics.anomalies_detected,
            "high_risk_transactions": self.metrics.high_risk_transactions,
            "recent_anomalies": len(self.anomaly_history)
        }
    
    def get_anomaly_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Son anomalileri raporla
        
        Args:
            limit: Maksimum kayıt
        
        Returns:
            Anomali listesi
        """
        recent = self.anomaly_history[-limit:]
        
        return [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "details": a.details
            }
            for a in recent
        ]