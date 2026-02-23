"""
LotusAI Messaging Manager
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Merkezi mesajlaşma yönetimi (Meta Graph API)

Özellikler:
- Multi-platform: WhatsApp, Instagram, Facebook Messenger
- Webhook entegrasyonu
- HMAC imza doğrulama
- Retry mekanizması
- Template mesajlar
- Simülasyon modu
- GPU desteği
- Erişim seviyesi kontrolleri (restricted/sandbox/full)
"""

import requests
import json
import logging
import time
import hmac
import hashlib
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.Messaging")


# ═══════════════════════════════════════════════════════════════
# GPU (PyTorch)
# ═══════════════════════════════════════════════════════════════
HAS_GPU = False
DEVICE = "cpu"

if Config.USE_GPU:
    try:
        import torch
        
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE = "cuda"
            logger.info("🚀 Messaging GPU aktif (CUDA)")
    except ImportError:
        logger.info("ℹ️ PyTorch yok, mesaj işleme CPU modunda")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class Platform(Enum):
    """Mesajlaşma platformları"""
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


class MessageType(Enum):
    """Mesaj tipleri"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    LOCATION = "location"
    TEMPLATE = "template"


class MessageStatus(Enum):
    """Mesaj durumları"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    QUEUED = "queued"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class IncomingMessage:
    """Gelen mesaj"""
    platform: Platform
    sender_id: str
    sender_name: str
    message: str
    message_type: MessageType
    timestamp: int


@dataclass
class SendResult:
    """Gönderim sonucu"""
    success: bool
    platform: Platform
    recipient_id: str
    message_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MessagingMetrics:
    """Messaging metrikleri"""
    messages_sent: int = 0
    messages_received: int = 0
    whatsapp_sent: int = 0
    instagram_sent: int = 0
    facebook_sent: int = 0
    errors_encountered: int = 0
    webhooks_verified: int = 0


# ═══════════════════════════════════════════════════════════════
# MESSAGING MANAGER
# ═══════════════════════════════════════════════════════════════
class MessagingManager:
    """
    LotusAI Merkezi Mesajlaşma Yöneticisi
    
    Yetenekler:
    - Multi-platform: WhatsApp, Instagram, Facebook Messenger
    - Meta Graph API: Resmi API entegrasyonu
    - Webhook: Gelen mesaj alma ve doğrulama
    - Security: HMAC imza doğrulama
    - Retry: Üstel geri çekilme ile garantili gönderim
    - Simulation: API olmadan test modu
    - GPU: Mesaj işleme için GPU desteği
    
    Meta Graph API kullanarak çoklu platformda mesajlaşma yönetir.
    """
    
    # API settings
    API_VERSION = "v18.0"
    BASE_URL = f"https://graph.facebook.com/{API_VERSION}"
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2  # seconds
    
    # Request timeout
    REQUEST_TIMEOUT = 20  # seconds
    
    def __init__(self, access_level: Optional[str] = None):
        """
        Messaging manager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        # Değişiklik: Eğer parametre girilmezse doğrudan Config'den oku
        self.access_level = access_level or Config.ACCESS_LEVEL
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hardware
        self.device = DEVICE
        
        # Config
        self.access_token = getattr(Config, 'META_ACCESS_TOKEN', "")
        self.wa_phone_id = getattr(Config, 'WHATSAPP_PHONE_ID', "")
        self.ig_account_id = getattr(Config, 'INSTAGRAM_ACCOUNT_ID', "")
        self.fb_page_id = getattr(Config, 'FACEBOOK_PAGE_ID', "")
        self.app_secret = getattr(Config, 'META_APP_SECRET', "")
        self.verify_token = getattr(
            Config,
            'META_VERIFY_TOKEN',
            "lotus_verify_token"
        )
        
        # Session
        self.session = requests.Session()
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Active check
        self.is_active = bool(
            self.access_token and
            (self.wa_phone_id or self.ig_account_id)
        )
        
        # Metrics
        self.metrics = MessagingMetrics()
        
        # Log status
        if not self.is_active:
            logger.warning("⚠️ SİMÜLASYON MODU (Kimlik bilgileri eksik)")
        else:
            logger.info(f"✅ API MODU AKTİF (Device: {self.device}, Erişim: {self.access_level})")
    
    # ───────────────────────────────────────────────────────────
    # GPU PROCESSING
    # ───────────────────────────────────────────────────────────
    
    def process_with_gpu(self, text_data: str) -> str:
        """
        GPU ile mesaj işleme (placeholder)
        
        Args:
            text_data: Mesaj metni
        
        Returns:
            İşlenmiş metin
        """
        if not HAS_GPU:
            return text_data
        
        try:
            # Placeholder for GPU processing
            return text_data
        
        except Exception as e:
            logger.error(f"GPU işleme hatası: {e}")
            return text_data
    
    # ───────────────────────────────────────────────────────────
    # WEBHOOK VERIFICATION
    # ───────────────────────────────────────────────────────────
    
    def verify_webhook(
        self,
        mode: str,
        token: str,
        challenge: str
    ) -> Optional[str]:
        """
        Webhook doğrulama (Meta setup) - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            mode: Subscribe mode
            token: Verify token
            challenge: Challenge string
        
        Returns:
            Challenge string veya None
        """
        if mode == "subscribe" and token == self.verify_token:
            logger.info("✅ Webhook doğrulandı")
            self.metrics.webhooks_verified += 1
            return challenge
        
        logger.error("❌ Webhook doğrulama başarısız")
        return None
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        HMAC imza doğrulama - Tüm erişim seviyelerinde kullanılabilir.
        
        Args:
            payload: Raw payload bytes
            signature: X-Hub-Signature header
        
        Returns:
            Geçerliyse True
        """
        if not self.app_secret or not signature:
            return True  # Dev mode flexibility
        
        try:
            actual_sig = signature.replace('sha1=', '')
            
            expected_sig = hmac.new(
                self.app_secret.encode('utf-8'),
                payload,
                digestmod=hashlib.sha1
            ).hexdigest()
            
            return hmac.compare_digest(expected_sig, actual_sig)
        
        except Exception as e:
            logger.error(f"İmza doğrulama hatası: {e}")
            return False
    
    # ───────────────────────────────────────────────────────────
    # REQUEST HANDLER
    # ───────────────────────────────────────────────────────────
    
    def _send_request(
        self,
        endpoint: str,
        payload: Dict,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        API request gönder (retry logic ile)
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            retries: Maksimum deneme sayısı
        
        Returns:
            Response dictionary
        """
        if retries is None:
            retries = self.MAX_RETRIES
        
        # Simulation mode
        if not self.is_active:
            logger.info(
                f"☁️ [SİMÜLASYON] Mesaj -> "
                f"{json.dumps(payload, ensure_ascii=False)[:100]}"
            )
            return {"status": "success", "mode": "simulation"}
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        with self.lock:
            for attempt in range(retries):
                try:
                    response = self.session.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.REQUEST_TIMEOUT
                    )
                    
                    response.raise_for_status()
                    
                    return {
                        "status": "success",
                        "data": response.json()
                    }
                
                except requests.exceptions.HTTPError:
                    error_msg = response.text
                    logger.error(
                        f"🚫 Meta API hatası ({response.status_code}): "
                        f"{error_msg[:100]}"
                    )
                    
                    if response.status_code < 500:
                        self.metrics.errors_encountered += 1
                        return {"status": "error", "error": error_msg}
                
                except Exception as e:
                    logger.warning(
                        f"⚠️ Bağlantı sorunu "
                        f"(Deneme {attempt + 1}/{retries}): {e}"
                    )
                
                if attempt < retries - 1:
                    time.sleep(self.RETRY_BASE_DELAY ** (attempt + 1))
        
        self.metrics.errors_encountered += 1
        return {
            "status": "error",
            "error": "Maksimum deneme sayısı aşıldı"
        }
    
    # ───────────────────────────────────────────────────────────
    # WHATSAPP (Erişim kontrollü)
    # ───────────────────────────────────────────────────────────
    
    def send_whatsapp_text(self, to_number: str, text: str) -> SendResult:
        """
        WhatsApp metin mesajı - Sadece sandbox ve full modda çalışır.
        
        Args:
            to_number: Alıcı telefon numarası
            text: Mesaj metni
        
        Returns:
            SendResult objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return SendResult(
                success=False,
                platform=Platform.WHATSAPP,
                recipient_id=to_number,
                error="🔒 Kısıtlı modda mesaj gönderilemez"
            )
        
        processed_text = self.process_with_gpu(text)
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"body": processed_text}
        }
        
        result = self._send_request(f"{self.wa_phone_id}/messages", payload)
        
        success = result.get("status") == "success"
        
        if success:
            self.metrics.messages_sent += 1
            self.metrics.whatsapp_sent += 1
        
        return SendResult(
            success=success,
            platform=Platform.WHATSAPP,
            recipient_id=to_number,
            message_id=result.get("data", {}).get("message_id"),
            error=result.get("error")
        )
    
    def send_whatsapp_media(
        self,
        to_number: str,
        media_url: str,
        media_type: str = "image",
        caption: str = ""
    ) -> SendResult:
        """
        WhatsApp medya mesajı - Sadece sandbox ve full modda çalışır.
        
        Args:
            to_number: Alıcı telefon numarası
            media_url: Medya URL
            media_type: Medya tipi (image/video/document)
            caption: Açıklama metni
        
        Returns:
            SendResult objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return SendResult(
                success=False,
                platform=Platform.WHATSAPP,
                recipient_id=to_number,
                error="🔒 Kısıtlı modda mesaj gönderilemez"
            )
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": media_type,
            media_type: {
                "link": media_url,
                "caption": caption
            }
        }
        
        result = self._send_request(f"{self.wa_phone_id}/messages", payload)
        
        success = result.get("status") == "success"
        
        if success:
            self.metrics.messages_sent += 1
            self.metrics.whatsapp_sent += 1
        
        return SendResult(
            success=success,
            platform=Platform.WHATSAPP,
            recipient_id=to_number,
            message_id=result.get("data", {}).get("message_id"),
            error=result.get("error")
        )
    
    def send_whatsapp_template(
        self,
        to_number: str,
        template_name: str,
        lang: str = "tr",
        components: Optional[List] = None
    ) -> SendResult:
        """
        WhatsApp template mesajı - Sadece sandbox ve full modda çalışır.
        
        Args:
            to_number: Alıcı telefon numarası
            template_name: Şablon adı
            lang: Dil kodu
            components: Şablon bileşenleri
        
        Returns:
            SendResult objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return SendResult(
                success=False,
                platform=Platform.WHATSAPP,
                recipient_id=to_number,
                error="🔒 Kısıtlı modda mesaj gönderilemez"
            )
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": lang}
            }
        }
        
        if components:
            payload["template"]["components"] = components
        
        result = self._send_request(f"{self.wa_phone_id}/messages", payload)
        
        success = result.get("status") == "success"
        
        if success:
            self.metrics.messages_sent += 1
            self.metrics.whatsapp_sent += 1
        
        return SendResult(
            success=success,
            platform=Platform.WHATSAPP,
            recipient_id=to_number,
            message_id=result.get("data", {}).get("message_id"),
            error=result.get("error")
        )
    
    # ───────────────────────────────────────────────────────────
    # INSTAGRAM & FACEBOOK (Erişim kontrollü)
    # ───────────────────────────────────────────────────────────
    
    def send_typing_indicator(
        self,
        recipient_id: str,
        platform: Platform,
        on: bool = True
    ) -> None:
        """
        Yazıyor göstergesi - Tüm erişim seviyelerinde kullanılabilir (sadece API modunda etki eder).
        
        Args:
            recipient_id: Alıcı ID
            platform: Platform
            on: Açık/Kapalı
        """
        if self.access_level == AccessLevel.RESTRICTED:
            # Kısıtlı modda bile göstermek sorun değil, sadece simülasyon log'u
            if not self.is_active:
                logger.info(f"☁️ [SİMÜLASYON] Yazıyor göstergesi: {platform.value}")
            return
        
        action = "typing_on" if on else "typing_off"
        
        payload = {
            "recipient": {"id": recipient_id},
            "sender_action": action
        }
        
        if platform == Platform.INSTAGRAM:
            id_source = self.ig_account_id
        else:
            id_source = self.fb_page_id
        
        endpoint = f"{id_source}/messages" if id_source else "me/messages"
        
        self._send_request(endpoint, payload)
    
    def send_instagram_text(
        self,
        recipient_id: str,
        text: str
    ) -> SendResult:
        """
        Instagram DM - Sadece sandbox ve full modda çalışır.
        
        Args:
            recipient_id: Alıcı ID
            text: Mesaj metni
        
        Returns:
            SendResult objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return SendResult(
                success=False,
                platform=Platform.INSTAGRAM,
                recipient_id=recipient_id,
                error="🔒 Kısıtlı modda mesaj gönderilemez"
            )
        
        self.send_typing_indicator(recipient_id, Platform.INSTAGRAM, True)
        
        processed_text = self.process_with_gpu(text)
        
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": processed_text}
        }
        
        endpoint = (
            f"{self.ig_account_id}/messages"
            if self.ig_account_id else "me/messages"
        )
        
        result = self._send_request(endpoint, payload)
        
        self.send_typing_indicator(recipient_id, Platform.INSTAGRAM, False)
        
        success = result.get("status") == "success"
        
        if success:
            self.metrics.messages_sent += 1
            self.metrics.instagram_sent += 1
        
        return SendResult(
            success=success,
            platform=Platform.INSTAGRAM,
            recipient_id=recipient_id,
            message_id=result.get("data", {}).get("message_id"),
            error=result.get("error")
        )
    
    def send_facebook_text(
        self,
        recipient_id: str,
        text: str
    ) -> SendResult:
        """
        Facebook Messenger mesajı - Sadece sandbox ve full modda çalışır.
        
        Args:
            recipient_id: Alıcı ID
            text: Mesaj metni
        
        Returns:
            SendResult objesi
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return SendResult(
                success=False,
                platform=Platform.FACEBOOK,
                recipient_id=recipient_id,
                error="🔒 Kısıtlı modda mesaj gönderilemez"
            )
        
        self.send_typing_indicator(recipient_id, Platform.FACEBOOK, True)
        
        processed_text = self.process_with_gpu(text)
        
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": processed_text}
        }
        
        endpoint = (
            f"{self.fb_page_id}/messages"
            if self.fb_page_id else "me/messages"
        )
        
        result = self._send_request(endpoint, payload)
        
        self.send_typing_indicator(recipient_id, Platform.FACEBOOK, False)
        
        success = result.get("status") == "success"
        
        if success:
            self.metrics.messages_sent += 1
            self.metrics.facebook_sent += 1
        
        return SendResult(
            success=success,
            platform=Platform.FACEBOOK,
            recipient_id=recipient_id,
            message_id=result.get("data", {}).get("message_id"),
            error=result.get("error")
        )
    
    # ───────────────────────────────────────────────────────────
    # WEBHOOK PARSER (Tüm erişim seviyelerinde kullanılabilir)
    # ───────────────────────────────────────────────────────────
    
    def parse_incoming_webhook(
        self,
        data: Dict
    ) -> Optional[IncomingMessage]:
        """
        Webhook parse - Tüm erişim seviyelerinde çalışır.
        
        Args:
            data: Webhook JSON
        
        Returns:
            IncomingMessage veya None
        """
        try:
            if 'entry' not in data or not data['entry']:
                return None
            
            entry = data['entry'][0]
            
            if 'changes' in entry:
                return self._parse_whatsapp(entry)
            elif 'messaging' in entry:
                return self._parse_ig_fb(entry)
            
            return None
        
        except Exception as e:
            logger.error(f"Webhook parse hatası: {e}")
            return None
    
    def _parse_whatsapp(self, entry: Dict) -> Optional[IncomingMessage]:
        """WhatsApp mesajı parse et"""
        try:
            val = entry['changes'][0].get('value', {})
            
            if 'messages' not in val:
                return None
            
            msg = val['messages'][0]
            profile = val.get('contacts', [{}])[0].get('profile', {})
            
            msg_type_str = msg.get('type', 'text')
            
            if msg_type_str == 'text':
                content = msg.get('text', {}).get('body', '')
            elif msg_type_str == 'location':
                loc = msg['location']
                content = (
                    f"[KONUM: {loc.get('latitude')}, "
                    f"{loc.get('longitude')}]"
                )
            else:
                content = f"[{msg_type_str.upper()} Medya]"
            
            try:
                msg_type = MessageType(msg_type_str)
            except ValueError:
                msg_type = MessageType.TEXT
            
            self.metrics.messages_received += 1
            
            return IncomingMessage(
                platform=Platform.WHATSAPP,
                sender_id=msg['from'],
                sender_name=profile.get('name', 'Bilinmeyen'),
                message=self.process_with_gpu(content),
                message_type=msg_type,
                timestamp=int(msg.get('timestamp', 0))
            )
        
        except Exception:
            return None
    
    def _parse_ig_fb(self, entry: Dict) -> Optional[IncomingMessage]:
        """Instagram/Facebook mesajı parse et"""
        try:
            event = entry['messaging'][0]
            
            if 'message' not in event:
                return None
            
            msg_data = event['message']
            sender_id = event['sender']['id']
            
            entry_id = str(entry.get('id', ''))
            
            if self.ig_account_id and self.ig_account_id in entry_id:
                platform = Platform.INSTAGRAM
            else:
                platform = Platform.FACEBOOK
            
            if 'text' in msg_data:
                content = msg_data['text']
                msg_type = MessageType.TEXT
            elif 'attachments' in msg_data:
                attach_type = msg_data['attachments'][0].get('type', 'media')
                content = f"[{attach_type.upper()} Eklenti]"
                try:
                    msg_type = MessageType(attach_type)
                except ValueError:
                    msg_type = MessageType.TEXT
            else:
                content = ""
                msg_type = MessageType.TEXT
            
            self.metrics.messages_received += 1
            
            return IncomingMessage(
                platform=platform,
                sender_id=sender_id,
                sender_name=f"{platform.value.title()} Kullanıcısı",
                message=self.process_with_gpu(content),
                message_type=msg_type,
                timestamp=int(event.get('timestamp', 0))
            )
        
        except Exception:
            return None
    
    # ───────────────────────────────────────────────────────────
    # STATUS & METRICS
    # ───────────────────────────────────────────────────────────
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Durum özeti
        
        Returns:
            Durum dictionary
        """
        return {
            "active_mode": "API" if self.is_active else "SIMULATION",
            "compute_device": self.device,
            "access_level": self.access_level,
            "gpu_available": HAS_GPU,
            "configured_platforms": {
                "whatsapp": bool(self.wa_phone_id),
                "instagram": bool(self.ig_account_id),
                "facebook": bool(self.fb_page_id)
            },
            "api_version": self.API_VERSION
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Messaging metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "whatsapp_sent": self.metrics.whatsapp_sent,
            "instagram_sent": self.metrics.instagram_sent,
            "facebook_sent": self.metrics.facebook_sent,
            "errors_encountered": self.metrics.errors_encountered,
            "webhooks_verified": self.metrics.webhooks_verified,
            "device": self.device,
            "access_level": self.access_level,
            "active": self.is_active
        }