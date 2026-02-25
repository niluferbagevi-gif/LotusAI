"""
LotusAI agents/gaya.py - Gaya Agent
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu & Mimari Optimizasyon)
Açıklama: Operasyon, finans ve iletişim uzmanı

Sorumluluklar:
- Fatura işleme
- Rezervasyon yönetimi
- Stok takibi
- Sosyal medya iletişimi
- Müşteri ilişkileri
- auto_handle: Operasyonel sorguları LLM'e gitmeden anlık karşılama
"""

import os
import re
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from decimal import Decimal, InvalidOperation

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Gaya")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class CommunicationChannel(Enum):
    """İletişim kanalları"""
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    MESSENGER = "messenger"
    EMAIL = "email"
    PHONE = "phone"
    DIRECT = "direct"


class TaskType(Enum):
    """Görev tipleri"""
    INVOICE = "invoice"
    RESERVATION = "reservation"
    SOCIAL_MEDIA = "social_media"
    CUSTOMER_SERVICE = "customer_service"
    STOCK_UPDATE = "stock_update"
    FINANCIAL = "financial"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class InvoiceData:
    """Fatura verisi"""
    firma: str
    toplam_tutar: float
    urunler: List[Dict[str, Any]]
    tarih: Optional[str] = None
    fatura_no: Optional[str] = None

    def __post_init__(self):
        if self.toplam_tutar < 0:
            raise ValueError("Tutar negatif olamaz")
        if not self.firma:
            self.firma = "Bilinmeyen Tedarikçi"


@dataclass
class ReservationData:
    """Rezervasyon verisi"""
    name: str
    time_slot: str
    kisi_sayisi: str
    phone: Optional[str] = None
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ProcessingResult:
    """İşlem sonucu"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class GayaMetrics:
    """Gaya metrikleri"""
    invoices_processed: int = 0
    reservations_handled: int = 0
    social_interactions: int = 0
    stock_updates: int = 0
    total_amount_processed: float = 0.0
    auto_handle_count: int = 0


# ═══════════════════════════════════════════════════════════════
# GAYA AGENT
# ═══════════════════════════════════════════════════════════════
class GayaAgent:
    """
    Gaya (Operasyon & İletişim Uzmanı)

    Yetenekler:
    - GPU hızlandırmalı NLP (rezervasyon analizi) - Merkezi NLP üzerinden
    - Fatura işleme ve muhasebe entegrasyonu
    - Çok kanallı iletişim yönetimi
    - Stok takibi
    - Sosyal medya içerik önerileri
    - auto_handle: Operasyonel sorguları LLM'siz karşılama

    Erişim Seviyesi Kuralları (OpenClaw):
    - restricted: Sadece bilgi verir. Fatura, rezervasyon işlemi yapamaz.
    - sandbox: Fatura ve rezervasyon işleyebilir.
    - full: Tüm yetkiler.

    Gaya, Lotus Bağevi'nin "marka yüzü" ve operasyonel kalbidir.
    """

    # Communication keywords
    CHANNEL_KEYWORDS = {
        CommunicationChannel.WHATSAPP: ["whatsapp", "wp", "mesaj"],
        CommunicationChannel.INSTAGRAM: ["instagram", "insta", "ig", "story", "dm"],
        CommunicationChannel.FACEBOOK: ["facebook", "fb", "messenger"],
        CommunicationChannel.MESSENGER: ["messenger", "msj"],
        CommunicationChannel.EMAIL: ["email", "mail", "e-posta"],
        CommunicationChannel.PHONE: ["telefon", "aradı", "call"]
    }

    # Task keywords
    TASK_KEYWORDS = {
        TaskType.INVOICE: ["fatura", "fiş", "dekont", "ödeme", "harcama"],
        TaskType.RESERVATION: ["masa", "rezervasyon", "yer", "ayırt", "geleceğiz"],
        TaskType.SOCIAL_MEDIA: ["post", "paylaş", "story", "reel", "içerik"],
        TaskType.STOCK_UPDATE: ["stok", "ürün", "malzeme", "tedarik"]
    }

    # auto_handle tetikleyicileri
    AUTO_METRIC_TRIGGERS = [
        "istatistik", "metrik", "kaç fatura", "kaç rezervasyon",
        "ne kadar işledim", "performans", "özet"
    ]
    AUTO_STOCK_TRIGGERS = [
        "stok durumu", "stok özeti", "malzeme durumu", "envanter"
    ]
    AUTO_SOCIAL_TRIGGERS = [
        "içerik öner", "post öner", "ne paylaşayım", "sosyal medya öner"
    ]
    AUTO_RESERVATION_STATUS_TRIGGERS = [
        "rezervasyonlar", "masa durumu", "bugünkü rezervasyon", "kim geliyor"
    ]

    def __init__(
        self,
        tools_dict: Dict[str, Any],
        nlp_manager: Optional[Any] = None,
        access_level: Optional[str] = None
    ):
        """
        Gaya başlatıcı

        Args:
            tools_dict: Engine'den gelen tool'lar
            nlp_manager: NLP yöneticisi (rezervasyon için)
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.tools = tools_dict
        self.nlp = nlp_manager

        # Parametre girilmezse Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL

        self.agent_name = "GAYA"

        # Thread safety
        self.lock = threading.RLock()

        # Hardware Info (Config üzerinden güvenli okuma)
        self.device_status = "GPU AKTİF" if Config.USE_GPU else "CPU MODU"

        # Metrics
        self.metrics = GayaMetrics()

        logger.info(
            f"🌸 {self.agent_name} Operasyon modülü başlatıldı "
            f"({self.device_status}, Erişim: {self.access_level.upper()})"
        )

    # ───────────────────────────────────────────────────────────
    # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
    # ───────────────────────────────────────────────────────────

    async def auto_handle(self, text: str) -> Optional[str]:
        """
        Kullanıcı metnini analiz ederek Gaya'nın araçlarını
        otomatik çalıştırır. engine.py tarafından çağrılır.

        LLM'e gitmeden önce anlık, deterministik yanıt üretir.
        Eşleşme yoksa None döner ve akış LLM'e devam eder.

        Args:
            text: Kullanıcı metni (temizlenmiş)

        Returns:
            Yanıt string veya None
        """
        text_lower = text.lower()
        self.metrics.auto_handle_count += 1

        # 1. Metrik / İstatistik sorgusu
        if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
            return self._format_metrics_report()

        # 2. Sosyal medya içerik önerisi
        if any(t in text_lower for t in self.AUTO_SOCIAL_TRIGGERS):
            return self.get_social_content_idea()

        # 3. Stok durumu sorgulama
        if any(t in text_lower for t in self.AUTO_STOCK_TRIGGERS):
            if self._has_tool('operations'):
                try:
                    stock_info = self.tools['operations'].get_stock_summary()
                    return f"📦 STOK DURUMU:\n{stock_info}"
                except Exception as e:
                    logger.debug(f"Stok sorgulama hatası: {e}")
                    return "⚠️ Stok bilgisi alınamadı."
            return "⚠️ Operasyon modülü aktif değil."

        # 4. Rezervasyon listesi sorgulama
        if any(t in text_lower for t in self.AUTO_RESERVATION_STATUS_TRIGGERS):
            if self.access_level == AccessLevel.RESTRICTED:
                return "🔒 Kısıtlı modda rezervasyon detayları görüntülenemez."
            if self._has_tool('operations'):
                try:
                    reservations = self.tools['operations'].get_reservations_summary()
                    return f"📅 BUGÜNKÜ REZERVASYONLAR:\n{reservations}"
                except Exception as e:
                    logger.debug(f"Rezervasyon sorgulama hatası: {e}")
                    return "⚠️ Rezervasyon bilgisi alınamadı."
            return "⚠️ Operasyon modülü aktif değil."

        # 5. Erişim seviyesi kısıtlaması hatırlatması
        if any(t in text_lower for t in ["fatura işle", "rezervasyon ekle", "kaydet"]):
            if self.access_level == AccessLevel.RESTRICTED:
                return (
                    "🔒 Kısıtlı (Restricted) moddayım.\n"
                    "Fatura işleme veya rezervasyon kaydetme yapamam.\n"
                    "Bu işlemler için Sandbox veya Tam Erişim modu gereklidir."
                )

        # Eşleşme yok — LLM devralır
        return None

    def _format_metrics_report(self) -> str:
        """Metrik raporunu formatla"""
        return (
            f"📊 GAYA OPERASYON METRİKLERİ\n"
            f"{'═' * 35}\n"
            f"🧾 İşlenen Fatura   : {self.metrics.invoices_processed}\n"
            f"📅 Rezervasyon      : {self.metrics.reservations_handled}\n"
            f"📱 Sosyal İletişim  : {self.metrics.social_interactions}\n"
            f"📦 Stok Güncelleme : {self.metrics.stock_updates}\n"
            f"💰 Toplam İşlem     : {self.metrics.total_amount_processed:,.2f} TL\n"
            f"{'═' * 35}"
        )

    # ───────────────────────────────────────────────────────────
    # CONTEXT GENERATION
    # ───────────────────────────────────────────────────────────

    def get_context_data(self, user_text: str) -> str:
        """
        Mesaj içeriğine göre bağlam oluştur.
        engine.py GAYA için bu metodu user_text ile çağırır.

        Args:
            user_text: Kullanıcı mesajı

        Returns:
            Context string
        """
        text_lower = user_text.lower()
        context_parts = []

        with self.lock:
            # Kanal tespiti
            channel = self._detect_channel(text_lower)
            if channel:
                context_parts.append(
                    f"\n📍 KANAL: {channel.value.upper()}\n"
                    "Yanıt kısa, öz ve ilgi çekici olmalı (CTA içermeli)"
                )

            # Task tespiti
            task_type = self._detect_task_type(text_lower)
            if task_type:
                context_parts.append(self._get_task_context(task_type))

            # Tool availability
            available_tools = list(self.tools.keys())
            context_parts.append(
                f"\n🔧 Mevcut Araçlar: {', '.join(available_tools)}"
            )

            # Erişim seviyesi bilgisi
            access_display = {
                AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi)",
                AccessLevel.SANDBOX: "📦 Sandbox (Güvenli işlemler)",
                AccessLevel.FULL: "⚡ Tam Erişim"
            }.get(self.access_level, self.access_level)

            context_parts.append(
                f"\n🔐 ERİŞİM SEVİYEN: {access_display}\n"
                "Kısıtlı modda işlem yapamazsın; sadece bilgi verirsin.\n"
                "Sandbox modunda güvenli işlemlere (rezervasyon, fatura) izin verilir.\n"
                "Tam modda tüm yetkiler açıktır."
            )

        return "\n".join(context_parts)

    def _detect_channel(self, text_lower: str) -> Optional[CommunicationChannel]:
        """İletişim kanalını tespit et"""
        for channel, keywords in self.CHANNEL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return channel
        return None

    def _detect_task_type(self, text_lower: str) -> Optional[TaskType]:
        """Görev tipini tespit et"""
        for task_type, keywords in self.TASK_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return task_type
        return None

    def _get_task_context(self, task_type: TaskType) -> str:
        """Task-specific context"""
        contexts = {
            TaskType.INVOICE: (
                "\n📝 GÖREV: Fatura İşleme\n"
                "Verileri 'AccountingManager' ve 'OperationsManager'a işle"
            ),
            TaskType.RESERVATION: (
                "\n📅 GÖREV: Rezervasyon\n"
                "Kişi sayısı, saat ve iletişim bilgilerini doğrula"
            ),
            TaskType.SOCIAL_MEDIA: (
                "\n📱 GÖREV: Sosyal Medya\n"
                "İçerik güncel trendlere uygun olmalı"
            ),
            TaskType.STOCK_UPDATE: (
                "\n📦 GÖREV: Stok Güncelleme\n"
                "Envanter sistemini güncelle"
            )
        }
        return contexts.get(task_type, "")

    # ───────────────────────────────────────────────────────────
    # INVOICE PROCESSING
    # ───────────────────────────────────────────────────────────

    def process_invoice_result(
        self,
        invoice_data: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Fatura verisini işle

        Args:
            invoice_data: AI vision'dan gelen fatura verisi

        Returns:
            ProcessingResult objesi
        """
        if not invoice_data:
            return ProcessingResult(
                success=False,
                message="⚠️ Fatura verisi sağlanamadı",
                error="No data"
            )

        # Kısıtlı modda fatura işleme yapılamaz
        if self.access_level == AccessLevel.RESTRICTED:
            return ProcessingResult(
                success=False,
                message=(
                    "🔒 Kısıtlı erişim modunda fatura işlenemiyor.\n"
                    "Sandbox veya Tam Erişim modu gereklidir."
                ),
                error="Access restricted"
            )

        with self.lock:
            try:
                firma = invoice_data.get('firma', 'Bilinmeyen Tedarikçi')
                raw_tutar = invoice_data.get('toplam_tutar', '0')
                tutar = self._clean_price(raw_tutar)
                urunler = invoice_data.get('urunler', [])

                results = []

                # 1. Stok güncelleme
                if urunler and self._has_tool('operations'):
                    stock_result = self._update_stock(urunler)
                    results.append(stock_result)
                elif urunler:
                    results.append("⚠️ Stok güncelleme yapılamadı (operations tool yok)")

                # 2. Muhasebe kaydı
                if tutar > 0:
                    accounting_result = self._record_accounting(firma, tutar)
                    results.append(accounting_result)
                else:
                    results.append("⚠️ Tutar belirsiz, muhasebe kaydı atlandı")

                self.metrics.invoices_processed += 1
                self.metrics.total_amount_processed += tutar

                report_lines = [
                    f"🧾 FATURA İŞLEME ÖZETİ ({firma})",
                    "═" * 40,
                    f"💰 Tutar: {tutar:,.2f} TL",
                    f"📦 Ürün Sayısı: {len(urunler)}",
                    "",
                    *results,
                    "═" * 40,
                    f"⚡ İşlem: {self.device_status} | Başarılı"
                ]

                return ProcessingResult(
                    success=True,
                    message="\n".join(report_lines),
                    data={
                        "firma": firma,
                        "tutar": tutar,
                        "urun_sayisi": len(urunler)
                    }
                )

            except Exception as e:
                logger.error(f"Fatura işleme hatası: {e}")
                return ProcessingResult(
                    success=False,
                    message="❌ Fatura işleme başarısız",
                    error=str(e)
                )

    def _clean_price(self, raw_price: Any) -> float:
        """
        Fiyat temizleme (güvenli)

        Args:
            raw_price: Ham fiyat verisi

        Returns:
            Temiz float değer
        """
        if not raw_price:
            return 0.0

        if isinstance(raw_price, (int, float)):
            return float(raw_price)

        try:
            price_str = str(raw_price).upper()

            # Para birimi sembollerini kaldır
            for sym in ["TL", "TRY", "₺", "$", "€", "£"]:
                price_str = price_str.replace(sym, "")
            price_str = price_str.strip()

            # Binlik/ondalık ayırıcı kontrolü
            if "," in price_str and "." in price_str:
                if price_str.rfind(",") > price_str.rfind("."):
                    price_str = price_str.replace(".", "").replace(",", ".")
                else:
                    price_str = price_str.replace(",", "")
            elif "," in price_str:
                price_str = price_str.replace(",", ".")

            # Sadece sayı ve nokta kalsın
            price_str = re.sub(r'[^0-9.]', '', price_str)

            if not price_str:
                return 0.0

            return float(Decimal(price_str))

        except (ValueError, InvalidOperation) as e:
            logger.error(f"Fiyat parse hatası ({raw_price}): {e}")
            return 0.0

    def _update_stock(self, urunler: List[Dict[str, Any]]) -> str:
        """Stok güncelleme"""
        try:
            result = self.tools['operations'].process_invoice_items(urunler)
            self.metrics.stock_updates += 1
            return f"📦 {result}"
        except Exception as e:
            logger.error(f"Stok güncelleme hatası: {e}")
            return "❌ Stok güncelleme başarısız"

    def _record_accounting(self, firma: str, tutar: float) -> str:
        """Muhasebe kaydı"""
        acc_tool = self.tools.get('accounting') or self.tools.get('finance')

        if not acc_tool or not hasattr(acc_tool, 'add_entry'):
            return "⚠️ Muhasebe modülü mevcut değil"

        try:
            success = acc_tool.add_entry(
                tur="GIDER",
                aciklama=f"{firma} Faturası (Gaya)",
                tutar=tutar,
                kategori="Mutfak/Operasyon",
                user_id="GAYA"
            )
            return (
                f"💰 Muhasebe: -{tutar:,.2f} TL gider kaydedildi"
                if success else "❌ Muhasebe kaydı başarısız"
            )
        except Exception as e:
            logger.error(f"Muhasebe kayıt hatası: {e}")
            return "❌ Muhasebe kaydı başarısız"

    # ───────────────────────────────────────────────────────────
    # RESERVATION HANDLING
    # ───────────────────────────────────────────────────────────

    def handle_reservation(
        self,
        user_text: str,
        user_name: str
    ) -> Optional[ProcessingResult]:
        """
        Rezervasyon talebini işle

        Args:
            user_text: Kullanıcı mesajı
            user_name: Kullanıcı adı

        Returns:
            ProcessingResult veya None
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return ProcessingResult(
                success=False,
                message=(
                    "🔒 Kısıtlı erişim modunda rezervasyon işlemi yapılamaz.\n"
                    "Sandbox veya Tam Erişim modu gereklidir."
                ),
                error="Access restricted"
            )

        if not self.nlp:
            return ProcessingResult(
                success=False,
                message="❌ NLP modülü mevcut değil",
                error="NLP unavailable"
            )

        with self.lock:
            try:
                # Type Mismatch Bug Fixed (NLP returns Dataclass, not Dict)
                details = self.nlp.extract_reservation_details(user_text)

                kisi_sayisi = details.person_count
                saat = details.time
                telefon = details.phone

                if kisi_sayisi == "Bilinmiyor" and saat == "Belirtilmedi":
                    return None

                if not self._has_tool('operations'):
                    return ProcessingResult(
                        success=False,
                        message="⚠️ Operasyon modülü aktif değil",
                        error="Operations unavailable"
                    )

                msg_tool = self.tools.get('messaging') or self.tools.get('media')

                result_msg = self.tools['operations'].add_reservation(
                    name=user_name,
                    time_slot=saat,
                    count=kisi_sayisi,
                    phone=telefon,
                    messenger=msg_tool
                )

                self.metrics.reservations_handled += 1

                return ProcessingResult(
                    success=True,
                    message=result_msg,
                    data={
                        "person_count": kisi_sayisi,
                        "time": saat,
                        "phone": telefon
                    }
                )

            except Exception as e:
                logger.error(f"Rezervasyon hatası: {e}")
                return ProcessingResult(
                    success=False,
                    message="❌ Rezervasyon işlemi başarısız",
                    error=str(e)
                )

    # ───────────────────────────────────────────────────────────
    # SOCIAL MEDIA
    # ───────────────────────────────────────────────────────────

    def get_social_content_idea(self) -> str:
        """Sosyal medya içerik önerisi"""
        if not self._has_tool('media'):
            return "📱 Medya modülü mevcut değil"

        try:
            daily_context = self.tools['media'].get_daily_context()
            self.metrics.social_interactions += 1
            return f"🌸 Gaya'nın İçerik Önerisi:\n{daily_context}"
        except Exception as e:
            logger.error(f"İçerik önerisi hatası: {e}")
            return "⚠️ İçerik önerisi oluşturulamadı"

    # ───────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ───────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Gaya karakter tanımı (LLM için)"""
        # Erişim seviyesi notu — Sidar ve Atlas ile tutarlı
        access_note = {
            AccessLevel.RESTRICTED: (
                "DİKKAT: Kısıtlı moddasın. "
                "Fatura ve rezervasyon işleme yapma, sadece bilgi ver."
            ),
            AccessLevel.SANDBOX: (
                "DİKKAT: Sandbox modundasın. "
                "Fatura ve rezervasyon işleyebilirsin."
            ),
            AccessLevel.FULL: (
                "Tam erişim yetkin var. "
                "Tüm operasyonel işlemleri yapabilirsin."
            )
        }.get(self.access_level, "")

        return (
            f"Sen {Config.PROJECT_NAME} sisteminin Operasyon ve İletişim "
            f"Uzmanı GAYA'sın.\n\n"

            "KARAKTER:\n"
            "- Son derece nazik ve yardımsever\n"
            "- Kurumsal ama samimi\n"
            "- Çözüm odaklı ve pratik\n"
            "- Satış kabiliyeti yüksek profesyonel\n"
            "- Enerji dolu ve güven verici\n\n"

            "GÖREVLER:\n"
            "- Fatura işleme ve muhasebe entegrasyonu\n"
            "- Rezervasyon yönetimi\n"
            "- Sosyal medya iletişimi\n"
            "- Stok takibi\n"
            "- Müşteri ilişkileri\n\n"

            f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

            "İLETİŞİM KURALLARI:\n"
            "- Müşterilere 'Siz' diliyle hitap et\n"
            "- Lotus Bağevi'nin samimi atmosferini yansıt\n"
            "- Kısa ve öz mesajlar (sosyal medya için)\n"
            "- Her zaman CTA (Call-to-Action) ekle\n\n"

            "FİNANSAL KURALLAR:\n"
            "- Fatura işlerken muhasebeci titizliğinde ol\n"
            "- Tutar ve firma bilgilerini asla atlama\n"
            "- Stok güncellemelerini takip et\n\n"

            f"DONANIM:\n"
            f"- {self.device_status} modunda çalışıyorsun\n"
            f"- Yüksek performans için optimize edildin\n"
        )

    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────

    def _has_tool(self, tool_name: str) -> bool:
        """Tool mevcut mu kontrol et"""
        return tool_name in self.tools

    def get_metrics(self) -> Dict[str, Any]:
        """Gaya metrikleri"""
        return {
            "agent_name": self.agent_name,
            "device": self.device_status,
            "access_level": self.access_level,
            "invoices_processed": self.metrics.invoices_processed,
            "reservations_handled": self.metrics.reservations_handled,
            "social_interactions": self.metrics.social_interactions,
            "stock_updates": self.metrics.stock_updates,
            "total_amount_processed": round(self.metrics.total_amount_processed, 2),
            "auto_handle_count": self.metrics.auto_handle_count,
            "tools_available": list(self.tools.keys())
        }

    def reset_metrics(self) -> None:
        """Metrikleri sıfırla"""
        with self.lock:
            self.metrics = GayaMetrics()
            logger.info("Gaya metrikleri sıfırlandı")




# """
# LotusAI agents/gaya.py - Gaya Agent
# Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
# Açıklama: Operasyon, finans ve iletişim uzmanı

# Sorumluluklar:
# - Fatura işleme
# - Rezervasyon yönetimi
# - Stok takibi
# - Sosyal medya iletişimi
# - Müşteri ilişkileri
# - auto_handle: Operasyonel sorguları LLM'e gitmeden anlık karşılama
# """

# import os
# import re
# import logging
# import threading
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum
# from datetime import datetime
# from decimal import Decimal, InvalidOperation

# # ═══════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════
# from config import Config, AccessLevel

# logger = logging.getLogger("LotusAI.Gaya")


# # ═══════════════════════════════════════════════════════════════
# # TORCH (GPU)
# # ═══════════════════════════════════════════════════════════════
# HAS_TORCH = False
# DEVICE = "cpu"

# if Config.USE_GPU:
#     try:
#         import torch
#         HAS_TORCH = True

#         if torch.cuda.is_available():
#             DEVICE = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             DEVICE = "mps"
#     except ImportError:
#         logger.warning("⚠️ Gaya: Config GPU açık ama torch yok")


# # ═══════════════════════════════════════════════════════════════
# # ENUMS
# # ═══════════════════════════════════════════════════════════════
# class CommunicationChannel(Enum):
#     """İletişim kanalları"""
#     WHATSAPP = "whatsapp"
#     INSTAGRAM = "instagram"
#     FACEBOOK = "facebook"
#     MESSENGER = "messenger"
#     EMAIL = "email"
#     PHONE = "phone"
#     DIRECT = "direct"


# class TaskType(Enum):
#     """Görev tipleri"""
#     INVOICE = "invoice"
#     RESERVATION = "reservation"
#     SOCIAL_MEDIA = "social_media"
#     CUSTOMER_SERVICE = "customer_service"
#     STOCK_UPDATE = "stock_update"
#     FINANCIAL = "financial"


# # ═══════════════════════════════════════════════════════════════
# # DATA STRUCTURES
# # ═══════════════════════════════════════════════════════════════
# @dataclass
# class InvoiceData:
#     """Fatura verisi"""
#     firma: str
#     toplam_tutar: float
#     urunler: List[Dict[str, Any]]
#     tarih: Optional[str] = None
#     fatura_no: Optional[str] = None

#     def __post_init__(self):
#         if self.toplam_tutar < 0:
#             raise ValueError("Tutar negatif olamaz")
#         if not self.firma:
#             self.firma = "Bilinmeyen Tedarikçi"


# @dataclass
# class ReservationData:
#     """Rezervasyon verisi"""
#     name: str
#     time_slot: str
#     kisi_sayisi: str
#     phone: Optional[str] = None
#     notes: Optional[str] = None
#     timestamp: Optional[datetime] = None

#     def __post_init__(self):
#         if self.timestamp is None:
#             self.timestamp = datetime.now()


# @dataclass
# class ProcessingResult:
#     """İşlem sonucu"""
#     success: bool
#     message: str
#     data: Optional[Dict[str, Any]] = None
#     error: Optional[str] = None


# @dataclass
# class GayaMetrics:
#     """Gaya metrikleri"""
#     invoices_processed: int = 0
#     reservations_handled: int = 0
#     social_interactions: int = 0
#     stock_updates: int = 0
#     total_amount_processed: float = 0.0
#     auto_handle_count: int = 0


# # ═══════════════════════════════════════════════════════════════
# # GAYA AGENT
# # ═══════════════════════════════════════════════════════════════
# class GayaAgent:
#     """
#     Gaya (Operasyon & İletişim Uzmanı)

#     Yetenekler:
#     - GPU hızlandırmalı NLP (rezervasyon analizi)
#     - Fatura işleme ve muhasebe entegrasyonu
#     - Çok kanallı iletişim yönetimi
#     - Stok takibi
#     - Sosyal medya içerik önerileri
#     - auto_handle: Operasyonel sorguları LLM'siz karşılama

#     Erişim Seviyesi Kuralları (OpenClaw):
#     - restricted: Sadece bilgi verir. Fatura, rezervasyon işlemi yapamaz.
#     - sandbox: Fatura ve rezervasyon işleyebilir.
#     - full: Tüm yetkiler.

#     Gaya, Lotus Bağevi'nin "marka yüzü" ve operasyonel kalbidir.
#     """

#     # Communication keywords
#     CHANNEL_KEYWORDS = {
#         CommunicationChannel.WHATSAPP: ["whatsapp", "wp", "mesaj"],
#         CommunicationChannel.INSTAGRAM: ["instagram", "insta", "ig", "story", "dm"],
#         CommunicationChannel.FACEBOOK: ["facebook", "fb", "messenger"],
#         CommunicationChannel.MESSENGER: ["messenger", "msj"],
#         CommunicationChannel.EMAIL: ["email", "mail", "e-posta"],
#         CommunicationChannel.PHONE: ["telefon", "aradı", "call"]
#     }

#     # Task keywords
#     TASK_KEYWORDS = {
#         TaskType.INVOICE: ["fatura", "fiş", "dekont", "ödeme", "harcama"],
#         TaskType.RESERVATION: ["masa", "rezervasyon", "yer", "ayırt", "geleceğiz"],
#         TaskType.SOCIAL_MEDIA: ["post", "paylaş", "story", "reel", "içerik"],
#         TaskType.STOCK_UPDATE: ["stok", "ürün", "malzeme", "tedarik"]
#     }

#     # auto_handle tetikleyicileri
#     AUTO_METRIC_TRIGGERS = [
#         "istatistik", "metrik", "kaç fatura", "kaç rezervasyon",
#         "ne kadar işledim", "performans", "özet"
#     ]
#     AUTO_STOCK_TRIGGERS = [
#         "stok durumu", "stok özeti", "malzeme durumu", "envanter"
#     ]
#     AUTO_SOCIAL_TRIGGERS = [
#         "içerik öner", "post öner", "ne paylaşayım", "sosyal medya öner"
#     ]
#     AUTO_RESERVATION_STATUS_TRIGGERS = [
#         "rezervasyonlar", "masa durumu", "bugünkü rezervasyon", "kim geliyor"
#     ]

#     def __init__(
#         self,
#         tools_dict: Dict[str, Any],
#         nlp_manager: Optional[Any] = None,
#         access_level: Optional[str] = None
#     ):
#         """
#         Gaya başlatıcı

#         Args:
#             tools_dict: Engine'den gelen tool'lar
#             nlp_manager: NLP yöneticisi (rezervasyon için)
#             access_level: Erişim seviyesi (restricted, sandbox, full)
#         """
#         self.tools = tools_dict
#         self.nlp = nlp_manager

#         # Parametre girilmezse Config'den oku
#         self.access_level = access_level or Config.ACCESS_LEVEL

#         self.agent_name = "GAYA"

#         # Thread safety
#         self.lock = threading.RLock()

#         # Hardware
#         self.device = DEVICE

#         # Metrics
#         self.metrics = GayaMetrics()

#         # Optimize subsystems
#         self._optimize_subsystems()

#         logger.info(
#             f"🌸 {self.agent_name} Operasyon modülü başlatıldı "
#             f"({self.device.upper()}, Erişim: {self.access_level})"
#         )

#     def _optimize_subsystems(self) -> None:
#         """Alt sistemleri optimize et"""
#         with self.lock:
#             if self.device != "cpu" and HAS_TORCH and self.nlp:
#                 if hasattr(self.nlp, 'to'):
#                     try:
#                         self.nlp.to(self.device)
#                         logger.debug(f"NLP → {self.device}")
#                     except Exception:
#                         pass

#     # ───────────────────────────────────────────────────────────
#     # AUTO HANDLE (OTOMATİK EYLEM — engine.py adım 5)
#     # ───────────────────────────────────────────────────────────

#     async def auto_handle(self, text: str) -> Optional[str]:
#         """
#         Kullanıcı metnini analiz ederek Gaya'nın araçlarını
#         otomatik çalıştırır. engine.py tarafından çağrılır.

#         LLM'e gitmeden önce anlık, deterministik yanıt üretir.
#         Eşleşme yoksa None döner ve akış LLM'e devam eder.

#         Args:
#             text: Kullanıcı metni (temizlenmiş)

#         Returns:
#             Yanıt string veya None
#         """
#         text_lower = text.lower()
#         self.metrics.auto_handle_count += 1

#         # 1. Metrik / İstatistik sorgusu
#         if any(t in text_lower for t in self.AUTO_METRIC_TRIGGERS):
#             return self._format_metrics_report()

#         # 2. Sosyal medya içerik önerisi
#         if any(t in text_lower for t in self.AUTO_SOCIAL_TRIGGERS):
#             return self.get_social_content_idea()

#         # 3. Stok durumu sorgulama
#         if any(t in text_lower for t in self.AUTO_STOCK_TRIGGERS):
#             if self._has_tool('operations'):
#                 try:
#                     stock_info = self.tools['operations'].get_stock_summary()
#                     return f"📦 STOK DURUMU:\n{stock_info}"
#                 except Exception as e:
#                     logger.debug(f"Stok sorgulama hatası: {e}")
#                     return "⚠️ Stok bilgisi alınamadı."
#             return "⚠️ Operasyon modülü aktif değil."

#         # 4. Rezervasyon listesi sorgulama
#         if any(t in text_lower for t in self.AUTO_RESERVATION_STATUS_TRIGGERS):
#             if self.access_level == AccessLevel.RESTRICTED:
#                 return "🔒 Kısıtlı modda rezervasyon detayları görüntülenemez."
#             if self._has_tool('operations'):
#                 try:
#                     reservations = self.tools['operations'].get_reservations_summary()
#                     return f"📅 BUGÜNKÜ REZERVASYONLAR:\n{reservations}"
#                 except Exception as e:
#                     logger.debug(f"Rezervasyon sorgulama hatası: {e}")
#                     return "⚠️ Rezervasyon bilgisi alınamadı."
#             return "⚠️ Operasyon modülü aktif değil."

#         # 5. Erişim seviyesi kısıtlaması hatırlatması
#         if any(t in text_lower for t in ["fatura işle", "rezervasyon ekle", "kaydet"]):
#             if self.access_level == AccessLevel.RESTRICTED:
#                 return (
#                     "🔒 Kısıtlı (Restricted) moddayım.\n"
#                     "Fatura işleme veya rezervasyon kaydetme yapamam.\n"
#                     "Bu işlemler için Sandbox veya Tam Erişim modu gereklidir."
#                 )

#         # Eşleşme yok — LLM devralır
#         return None

#     def _format_metrics_report(self) -> str:
#         """Metrik raporunu formatla"""
#         return (
#             f"📊 GAYA OPERASYON METRİKLERİ\n"
#             f"{'═' * 35}\n"
#             f"🧾 İşlenen Fatura   : {self.metrics.invoices_processed}\n"
#             f"📅 Rezervasyon      : {self.metrics.reservations_handled}\n"
#             f"📱 Sosyal İletişim  : {self.metrics.social_interactions}\n"
#             f"📦 Stok Güncelleme : {self.metrics.stock_updates}\n"
#             f"💰 Toplam İşlem     : {self.metrics.total_amount_processed:,.2f} TL\n"
#             f"{'═' * 35}"
#         )

#     # ───────────────────────────────────────────────────────────
#     # CONTEXT GENERATION
#     # ───────────────────────────────────────────────────────────

#     def get_context_data(self, user_text: str) -> str:
#         """
#         Mesaj içeriğine göre bağlam oluştur.
#         engine.py GAYA için bu metodu user_text ile çağırır.

#         Args:
#             user_text: Kullanıcı mesajı

#         Returns:
#             Context string
#         """
#         text_lower = user_text.lower()
#         context_parts = []

#         with self.lock:
#             # Kanal tespiti
#             channel = self._detect_channel(text_lower)
#             if channel:
#                 context_parts.append(
#                     f"\n📍 KANAL: {channel.value.upper()}\n"
#                     "Yanıt kısa, öz ve ilgi çekici olmalı (CTA içermeli)"
#                 )

#             # Task tespiti
#             task_type = self._detect_task_type(text_lower)
#             if task_type:
#                 context_parts.append(self._get_task_context(task_type))

#             # Tool availability
#             available_tools = list(self.tools.keys())
#             context_parts.append(
#                 f"\n🔧 Mevcut Araçlar: {', '.join(available_tools)}"
#             )

#             # Erişim seviyesi bilgisi
#             access_display = {
#                 AccessLevel.RESTRICTED: "🔒 Kısıtlı (Sadece bilgi)",
#                 AccessLevel.SANDBOX: "📦 Sandbox (Güvenli işlemler)",
#                 AccessLevel.FULL: "⚡ Tam Erişim"
#             }.get(self.access_level, self.access_level)

#             context_parts.append(
#                 f"\n🔐 ERİŞİM SEVİYEN: {access_display}\n"
#                 "Kısıtlı modda işlem yapamazsın; sadece bilgi verirsin.\n"
#                 "Sandbox modunda güvenli işlemlere (rezervasyon, fatura) izin verilir.\n"
#                 "Tam modda tüm yetkiler açıktır."
#             )

#         return "\n".join(context_parts)

#     def _detect_channel(self, text_lower: str) -> Optional[CommunicationChannel]:
#         """İletişim kanalını tespit et"""
#         for channel, keywords in self.CHANNEL_KEYWORDS.items():
#             if any(kw in text_lower for kw in keywords):
#                 return channel
#         return None

#     def _detect_task_type(self, text_lower: str) -> Optional[TaskType]:
#         """Görev tipini tespit et"""
#         for task_type, keywords in self.TASK_KEYWORDS.items():
#             if any(kw in text_lower for kw in keywords):
#                 return task_type
#         return None

#     def _get_task_context(self, task_type: TaskType) -> str:
#         """Task-specific context"""
#         contexts = {
#             TaskType.INVOICE: (
#                 "\n📝 GÖREV: Fatura İşleme\n"
#                 "Verileri 'AccountingManager' ve 'OperationsManager'a işle"
#             ),
#             TaskType.RESERVATION: (
#                 "\n📅 GÖREV: Rezervasyon\n"
#                 "Kişi sayısı, saat ve iletişim bilgilerini doğrula"
#             ),
#             TaskType.SOCIAL_MEDIA: (
#                 "\n📱 GÖREV: Sosyal Medya\n"
#                 "İçerik güncel trendlere uygun olmalı"
#             ),
#             TaskType.STOCK_UPDATE: (
#                 "\n📦 GÖREV: Stok Güncelleme\n"
#                 "Envanter sistemini güncelle"
#             )
#         }
#         return contexts.get(task_type, "")

#     # ───────────────────────────────────────────────────────────
#     # INVOICE PROCESSING
#     # ───────────────────────────────────────────────────────────

#     def process_invoice_result(
#         self,
#         invoice_data: Dict[str, Any]
#     ) -> ProcessingResult:
#         """
#         Fatura verisini işle

#         Args:
#             invoice_data: AI vision'dan gelen fatura verisi

#         Returns:
#             ProcessingResult objesi
#         """
#         if not invoice_data:
#             return ProcessingResult(
#                 success=False,
#                 message="⚠️ Fatura verisi sağlanamadı",
#                 error="No data"
#             )

#         # Kısıtlı modda fatura işleme yapılamaz
#         if self.access_level == AccessLevel.RESTRICTED:
#             return ProcessingResult(
#                 success=False,
#                 message=(
#                     "🔒 Kısıtlı erişim modunda fatura işlenemiyor.\n"
#                     "Sandbox veya Tam Erişim modu gereklidir."
#                 ),
#                 error="Access restricted"
#             )

#         with self.lock:
#             try:
#                 firma = invoice_data.get('firma', 'Bilinmeyen Tedarikçi')
#                 raw_tutar = invoice_data.get('toplam_tutar', '0')
#                 tutar = self._clean_price(raw_tutar)
#                 urunler = invoice_data.get('urunler', [])

#                 results = []

#                 # 1. Stok güncelleme
#                 if urunler and self._has_tool('operations'):
#                     stock_result = self._update_stock(urunler)
#                     results.append(stock_result)
#                 elif urunler:
#                     results.append("⚠️ Stok güncelleme yapılamadı (operations tool yok)")

#                 # 2. Muhasebe kaydı
#                 if tutar > 0:
#                     accounting_result = self._record_accounting(firma, tutar)
#                     results.append(accounting_result)
#                 else:
#                     results.append("⚠️ Tutar belirsiz, muhasebe kaydı atlandı")

#                 self.metrics.invoices_processed += 1
#                 self.metrics.total_amount_processed += tutar

#                 report_lines = [
#                     f"🧾 FATURA İŞLEME ÖZETİ ({firma})",
#                     "═" * 40,
#                     f"💰 Tutar: {tutar:,.2f} TL",
#                     f"📦 Ürün Sayısı: {len(urunler)}",
#                     "",
#                     *results,
#                     "═" * 40,
#                     f"⚡ İşlem: {self.device.upper()} | Başarılı"
#                 ]

#                 return ProcessingResult(
#                     success=True,
#                     message="\n".join(report_lines),
#                     data={
#                         "firma": firma,
#                         "tutar": tutar,
#                         "urun_sayisi": len(urunler)
#                     }
#                 )

#             except Exception as e:
#                 logger.error(f"Fatura işleme hatası: {e}")
#                 return ProcessingResult(
#                     success=False,
#                     message="❌ Fatura işleme başarısız",
#                     error=str(e)
#                 )

#     def _clean_price(self, raw_price: Any) -> float:
#         """
#         Fiyat temizleme (güvenli)

#         Args:
#             raw_price: Ham fiyat verisi

#         Returns:
#             Temiz float değer
#         """
#         if not raw_price:
#             return 0.0

#         if isinstance(raw_price, (int, float)):
#             return float(raw_price)

#         try:
#             price_str = str(raw_price).upper()

#             # Para birimi sembollerini kaldır
#             for sym in ["TL", "TRY", "₺", "$", "€", "£"]:
#                 price_str = price_str.replace(sym, "")
#             price_str = price_str.strip()

#             # Binlik/ondalık ayırıcı kontrolü
#             if "," in price_str and "." in price_str:
#                 if price_str.rfind(",") > price_str.rfind("."):
#                     price_str = price_str.replace(".", "").replace(",", ".")
#                 else:
#                     price_str = price_str.replace(",", "")
#             elif "," in price_str:
#                 price_str = price_str.replace(",", ".")

#             # Sadece sayı ve nokta kalsın
#             price_str = re.sub(r'[^0-9.]', '', price_str)

#             if not price_str:
#                 return 0.0

#             return float(Decimal(price_str))

#         except (ValueError, InvalidOperation) as e:
#             logger.error(f"Fiyat parse hatası ({raw_price}): {e}")
#             return 0.0

#     def _update_stock(self, urunler: List[Dict[str, Any]]) -> str:
#         """Stok güncelleme"""
#         try:
#             result = self.tools['operations'].process_invoice_items(urunler)
#             self.metrics.stock_updates += 1
#             return f"📦 {result}"
#         except Exception as e:
#             logger.error(f"Stok güncelleme hatası: {e}")
#             return "❌ Stok güncelleme başarısız"

#     def _record_accounting(self, firma: str, tutar: float) -> str:
#         """Muhasebe kaydı"""
#         acc_tool = self.tools.get('accounting') or self.tools.get('finance')

#         if not acc_tool or not hasattr(acc_tool, 'add_entry'):
#             return "⚠️ Muhasebe modülü mevcut değil"

#         try:
#             acc_tool.add_entry(
#                 tur="GIDER",
#                 aciklama=f"{firma} Faturası (Gaya)",
#                 tutar=tutar,
#                 kategori="Mutfak/Operasyon",
#                 user_id="GAYA"
#             )
#             return f"💰 Muhasebe: -{tutar:,.2f} TL gider kaydedildi"
#         except Exception as e:
#             logger.error(f"Muhasebe kayıt hatası: {e}")
#             return "❌ Muhasebe kaydı başarısız"

#     # ───────────────────────────────────────────────────────────
#     # RESERVATION HANDLING
#     # ───────────────────────────────────────────────────────────

#     def handle_reservation(
#         self,
#         user_text: str,
#         user_name: str
#     ) -> Optional[ProcessingResult]:
#         """
#         Rezervasyon talebini işle

#         Args:
#             user_text: Kullanıcı mesajı
#             user_name: Kullanıcı adı

#         Returns:
#             ProcessingResult veya None
#         """
#         if self.access_level == AccessLevel.RESTRICTED:
#             return ProcessingResult(
#                 success=False,
#                 message=(
#                     "🔒 Kısıtlı erişim modunda rezervasyon işlemi yapılamaz.\n"
#                     "Sandbox veya Tam Erişim modu gereklidir."
#                 ),
#                 error="Access restricted"
#             )

#         if not self.nlp:
#             return ProcessingResult(
#                 success=False,
#                 message="❌ NLP modülü mevcut değil",
#                 error="NLP unavailable"
#             )

#         with self.lock:
#             try:
#                 details = self.nlp.extract_reservation_details(user_text)

#                 kisi_sayisi = details.get("kisi_sayisi", "Bilinmiyor")
#                 saat = details.get("saat", "Belirtilmedi")

#                 if kisi_sayisi == "Bilinmiyor" and saat == "Belirtilmedi":
#                     return None

#                 if not self._has_tool('operations'):
#                     return ProcessingResult(
#                         success=False,
#                         message="⚠️ Operasyon modülü aktif değil",
#                         error="Operations unavailable"
#                     )

#                 msg_tool = self.tools.get('messaging') or self.tools.get('media')

#                 result_msg = self.tools['operations'].add_reservation(
#                     name=user_name,
#                     time_slot=saat,
#                     count=kisi_sayisi,
#                     phone=details.get("iletisim"),
#                     messenger=msg_tool
#                 )

#                 self.metrics.reservations_handled += 1

#                 return ProcessingResult(
#                     success=True,
#                     message=result_msg,
#                     data=details
#                 )

#             except Exception as e:
#                 logger.error(f"Rezervasyon hatası: {e}")
#                 return ProcessingResult(
#                     success=False,
#                     message="❌ Rezervasyon işlemi başarısız",
#                     error=str(e)
#                 )

#     # ───────────────────────────────────────────────────────────
#     # SOCIAL MEDIA
#     # ───────────────────────────────────────────────────────────

#     def get_social_content_idea(self) -> str:
#         """Sosyal medya içerik önerisi"""
#         if not self._has_tool('media'):
#             return "📱 Medya modülü mevcut değil"

#         try:
#             daily_context = self.tools['media'].get_daily_context()
#             self.metrics.social_interactions += 1
#             return f"🌸 Gaya'nın İçerik Önerisi:\n{daily_context}"
#         except Exception as e:
#             logger.error(f"İçerik önerisi hatası: {e}")
#             return "⚠️ İçerik önerisi oluşturulamadı"

#     # ───────────────────────────────────────────────────────────
#     # SYSTEM PROMPT
#     # ───────────────────────────────────────────────────────────

#     def get_system_prompt(self) -> str:
#         """Gaya karakter tanımı (LLM için)"""
#         # Erişim seviyesi notu — Sidar ve Atlas ile tutarlı
#         access_note = {
#             AccessLevel.RESTRICTED: (
#                 "DİKKAT: Kısıtlı moddasın. "
#                 "Fatura ve rezervasyon işleme yapma, sadece bilgi ver."
#             ),
#             AccessLevel.SANDBOX: (
#                 "DİKKAT: Sandbox modundasın. "
#                 "Fatura ve rezervasyon işleyebilirsin."
#             ),
#             AccessLevel.FULL: (
#                 "Tam erişim yetkin var. "
#                 "Tüm operasyonel işlemleri yapabilirsin."
#             )
#         }.get(self.access_level, "")

#         return (
#             f"Sen {Config.PROJECT_NAME} sisteminin Operasyon ve İletişim "
#             f"Uzmanı GAYA'sın.\n\n"

#             "KARAKTER:\n"
#             "- Son derece nazik ve yardımsever\n"
#             "- Kurumsal ama samimi\n"
#             "- Çözüm odaklı ve pratik\n"
#             "- Satış kabiliyeti yüksek profesyonel\n"
#             "- Enerji dolu ve güven verici\n\n"

#             "GÖREVLER:\n"
#             "- Fatura işleme ve muhasebe entegrasyonu\n"
#             "- Rezervasyon yönetimi\n"
#             "- Sosyal medya iletişimi\n"
#             "- Stok takibi\n"
#             "- Müşteri ilişkileri\n\n"

#             f"ERİŞİM SEVİYESİ NOTU:\n{access_note}\n\n"

#             "İLETİŞİM KURALLARI:\n"
#             "- Müşterilere 'Siz' diliyle hitap et\n"
#             "- Lotus Bağevi'nin samimi atmosferini yansıt\n"
#             "- Kısa ve öz mesajlar (sosyal medya için)\n"
#             "- Her zaman CTA (Call-to-Action) ekle\n\n"

#             "FİNANSAL KURALLAR:\n"
#             "- Fatura işlerken muhasebeci titizliğinde ol\n"
#             "- Tutar ve firma bilgilerini asla atlama\n"
#             "- Stok güncellemelerini takip et\n\n"

#             f"DONANIM:\n"
#             f"- {self.device.upper()} modunda çalışıyorsun\n"
#             f"- Yüksek performans için optimize edildin\n"
#         )

#     # ───────────────────────────────────────────────────────────
#     # UTILITIES
#     # ───────────────────────────────────────────────────────────

#     def _has_tool(self, tool_name: str) -> bool:
#         """Tool mevcut mu kontrol et"""
#         return tool_name in self.tools

#     def get_metrics(self) -> Dict[str, Any]:
#         """Gaya metrikleri"""
#         return {
#             "agent_name": self.agent_name,
#             "device": self.device,
#             "access_level": self.access_level,
#             "invoices_processed": self.metrics.invoices_processed,
#             "reservations_handled": self.metrics.reservations_handled,
#             "social_interactions": self.metrics.social_interactions,
#             "stock_updates": self.metrics.stock_updates,
#             "total_amount_processed": round(self.metrics.total_amount_processed, 2),
#             "auto_handle_count": self.metrics.auto_handle_count,
#             "tools_available": list(self.tools.keys())
#         }

#     def reset_metrics(self) -> None:
#         """Metrikleri sıfırla"""
#         with self.lock:
#             self.metrics = GayaMetrics()
#             logger.info("Gaya metrikleri sıfırlandı")