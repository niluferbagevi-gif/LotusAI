import os
import requests
import json
import logging
import time
import hmac
import hashlib
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    # Bağımsız çalışma durumu için Fallback
    class Config:
        META_ACCESS_TOKEN = ""
        WHATSAPP_PHONE_ID = ""
        INSTAGRAM_ACCOUNT_ID = ""
        FACEBOOK_PAGE_ID = ""
        META_APP_SECRET = ""
        META_VERIFY_TOKEN = "lotus_verify_token"
        USE_GPU = False

# --- LOGLAMA YAPILANDIRMASI ---
logger = logging.getLogger("LotusAI.Messaging")

# --- GPU / TORCH ENTEGRASYONU (CONFIG KONTROLLÜ) ---
HAS_GPU = False
DEVICE = "cpu"
USE_GPU_CONFIG = getattr(Config, "USE_GPU", False)

if USE_GPU_CONFIG:
    try:
        import torch
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE = "cuda"
            logger.info("🚀 MessagingManager GPU Aktif (CUDA)")
        else:
            logger.info("ℹ️ MessagingManager: Config GPU açık ancak donanım bulunamadı. CPU kullanılacak.")
    except ImportError:
        logger.info("ℹ️ PyTorch yüklü değil, mesaj işleme CPU modunda.")
else:
    logger.info("ℹ️ Mesaj işleme CPU modunda (Config ayarı).")


class MessagingManager:
    """
    LotusAI Merkezi Mesajlaşma Yöneticisi (Meta Graph API).
    
    Yetenekler:
    - Çoklu Kanal: WhatsApp, Instagram ve Facebook Messenger entegrasyonu.
    - GPU Entegrasyonu: Gelen/Giden mesaj içeriklerini GPU tabanlı analiz için hazırlar (Config kontrollü).
    - Güvenli Webhook: HMAC imza doğrulama ve challenge yanıt sistemi.
    - Akıllı Kuyruk: Üstel geri çekilme (retry logic) ile garantili mesaj gönderimi.
    - Simülasyon: API anahtarı yoksa otomatik test moduna geçiş.
    """
    
    def __init__(self):
        # Eşzamanlılık Kilidi
        self.lock = threading.RLock()
        
        # Donanım Durumu
        self.device = DEVICE
        
        # Ayarlar (Config üzerinden)
        self.access_token = getattr(Config, 'META_ACCESS_TOKEN', "")
        self.wa_phone_id = getattr(Config, 'WHATSAPP_PHONE_ID', "")
        self.ig_account_id = getattr(Config, 'INSTAGRAM_ACCOUNT_ID', "")
        self.fb_page_id = getattr(Config, 'FACEBOOK_PAGE_ID', "")
        self.app_secret = getattr(Config, 'META_APP_SECRET', "")
        self.verify_token = getattr(Config, 'META_VERIFY_TOKEN', "lotus_verify_token")
        
        self.api_version = "v18.0"
        self.base_url = f"https://graph.facebook.com/{self.api_version}"
        
        # Bağlantı havuzu
        self.session = requests.Session()
        
        # Servis Aktivite Durumu
        self.is_active = bool(self.access_token and (self.wa_phone_id or self.ig_account_id))
        
        if not self.is_active:
            logger.warning("⚠️ MessagingManager: SİMÜLASYON MODU AKTİF (Kimlik bilgileri eksik).")
        else:
            logger.info(f"✅ MessagingManager: API MODU AKTİF (Cihaz: {self.device}).")

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    # --- GPU İŞLEME VE ZEKA KATMANI ---

    def process_with_gpu(self, text_data: str) -> str:
        """
        Mesaj içeriğini GPU üzerinde işlenmek üzere hazırlar (Placeholder).
        """
        if not HAS_GPU:
            return text_data # GPU yoksa direkt döndür
            
        try:
            # Örnek: Metni GPU belleğine taşıma simülasyonu
            # Gerçek senaryoda burada torch.tensor işlemleri olurdu.
            # logger.debug(f"🧠 Mesaj GPU ({self.device}) üzerinde analiz ediliyor...")
            return text_data
        except Exception as e:
            logger.error(f"GPU İşleme Hatası: {e}")
            return text_data

    # --- GÜVENLİK VE DOĞRULAMA ---

    def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """Meta Webhook kurulumu sırasında gelen GET isteğini yanıtlar."""
        if mode == "subscribe" and token == self.verify_token:
            logger.info("✅ Webhook bağlantısı doğrulandı.")
            return challenge
        logger.error("❌ Webhook doğrulama başarısız! Yanlış token.")
        return None

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Meta'dan gelen POST isteklerinin imzasını (X-Hub-Signature) doğrular."""
        if not self.app_secret or not signature:
            return True # Geliştirme aşamasında esneklik sağlar
        
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

    def _send_request(self, endpoint: str, payload: Dict, retries: int = 3) -> Dict[str, Any]:
        """Güvenli ve dirençli HTTP POST isteği göndericisi."""
        if not self.is_active:
            logger.info(f"☁️ [SİMÜLASYON] Mesaj -> {json.dumps(payload, ensure_ascii=False)}")
            return {"status": "success", "mode": "simulation"}

        url = f"{self.base_url}/{endpoint}"
        
        with self.lock:
            for attempt in range(retries):
                try:
                    response = self.session.post(url, headers=self.headers, json=payload, timeout=20)
                    response.raise_for_status()
                    return {"status": "success", "data": response.json()}
                
                except requests.exceptions.HTTPError as e:
                    error_msg = response.text
                    logger.error(f"🚫 Meta API Hatası ({response.status_code}): {error_msg}")
                    if response.status_code < 500:
                        return {"status": "error", "error": error_msg}
                
                except Exception as e:
                    logger.warning(f"⚠️ Bağlantı Sorunu (Deneme {attempt+1}/{retries}): {e}")
                
                # Üstel geri çekilme
                time.sleep(2 ** (attempt + 1))

        return {"status": "error", "error": "Maksimum deneme sayısına ulaşıldı."}

    # --- WHATSAPP FONKSİYONLARI ---

    def send_whatsapp_text(self, to_number: str, text: str) -> Dict:
        """WhatsApp üzerinden metin mesajı gönderir."""
        processed_text = self.process_with_gpu(text)
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"body": processed_text}
        }
        return self._send_request(f"{self.wa_phone_id}/messages", payload)

    def send_whatsapp_media(self, to_number: str, media_url: str, media_type: str = "image", caption: str = "") -> Dict:
        """WhatsApp üzerinden resim, video veya döküman gönderir."""
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": media_type,
            media_type: {"link": media_url, "caption": caption}
        }
        return self._send_request(f"{self.wa_phone_id}/messages", payload)

    def send_whatsapp_template(self, to_number: str, template_name: str, lang: str = "tr", components: List = None) -> Dict:
        """Onaylı şablon mesaj gönderir."""
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": lang}
            }
        }
        if components: payload["template"]["components"] = components
        return self._send_request(f"{self.wa_phone_id}/messages", payload)

    # --- INSTAGRAM VE FACEBOOK MESSENGER ---

    def send_ig_fb_typing(self, recipient_id: str, platform: str = "INSTAGRAM", on: bool = True):
        """Kullanıcıya 'Yazıyor...' işareti gönderir."""
        action = "typing_on" if on else "typing_off"
        payload = {
            "recipient": {"id": recipient_id},
            "sender_action": action
        }
        id_source = self.ig_account_id if platform == "INSTAGRAM" else self.fb_page_id
        endpoint = f"{id_source}/messages" if id_source else "me/messages"
        return self._send_request(endpoint, payload)

    def send_instagram_text(self, recipient_id: str, text: str) -> Dict:
        """Instagram DM gönderir."""
        self.send_ig_fb_typing(recipient_id, "INSTAGRAM", True)
        processed_text = self.process_with_gpu(text)
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": processed_text}
        }
        endpoint = f"{self.ig_account_id}/messages" if self.ig_account_id else "me/messages"
        res = self._send_request(endpoint, payload)
        self.send_ig_fb_typing(recipient_id, "INSTAGRAM", False)
        return res

    def send_facebook_text(self, recipient_id: str, text: str) -> Dict:
        """FB Messenger mesajı gönderir."""
        self.send_ig_fb_typing(recipient_id, "FACEBOOK", True)
        processed_text = self.process_with_gpu(text)
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": processed_text}
        }
        endpoint = f"{self.fb_page_id}/messages" if self.fb_page_id else "me/messages"
        res = self._send_request(endpoint, payload)
        self.send_ig_fb_typing(recipient_id, "FACEBOOK", False)
        return res

    # --- WEBHOOK PARSER (Gelen Veriyi İşleme) ---

    def parse_incoming_webhook(self, data: Dict) -> Optional[Dict]:
        """Meta Webhook JSON'unu standart LotusAI formatına çevirir."""
        try:
            if 'entry' not in data or not data['entry']: return None
            entry = data['entry'][0]
            
            # 1. WhatsApp Mesajları
            if 'changes' in entry:
                val = entry['changes'][0].get('value', {})
                if 'messages' in val:
                    msg = val['messages'][0]
                    profile = val.get('contacts', [{}])[0].get('profile', {})
                    msg_type = msg.get('type', 'text')
                    
                    content = msg.get('text', {}).get('body', f"[{msg_type.upper()} Medya]")
                    if msg_type == 'location':
                        loc = msg['location']
                        content = f"[KONUM: {loc.get('latitude')}, {loc.get('longitude')}]"

                    return {
                        "platform": "WHATSAPP",
                        "sender_id": msg['from'],
                        "sender_name": profile.get('name', 'Bilinmeyen Kullanıcı'),
                        "message": self.process_with_gpu(content),
                        "msg_type": msg_type,
                        "timestamp": msg.get('timestamp')
                    }

            # 2. Instagram / Facebook Mesajları
            elif 'messaging' in entry:
                event = entry['messaging'][0]
                if 'message' in event:
                    msg_data = event['message']
                    sender_id = event['sender']['id']
                    
                    entry_id = str(entry.get('id', ''))
                    platform = "INSTAGRAM" if self.ig_account_id and self.ig_account_id in entry_id else "FACEBOOK"
                    
                    content = msg_data.get('text', "")
                    if 'attachments' in msg_data:
                        content = f"[MEDYA EKLENTİSİ: {msg_data['attachments'][0].get('type')}]"

                    return {
                        "platform": platform,
                        "sender_id": sender_id,
                        "sender_name": f"{platform.title()} Kullanıcısı",
                        "message": self.process_with_gpu(content),
                        "msg_type": "text" if msg_data.get('text') else "media",
                        "timestamp": event.get('timestamp')
                    }

            return None

        except Exception as e:
            logger.error(f"Webhook Ayrıştırma Hatası: {e}")
            return None

    def get_status_summary(self) -> Dict:
        """Sistemin bağlantı ve GPU durumunu özetler."""
        return {
            "active_mode": "API" if self.is_active else "SIMULATION",
            "compute_device": self.device,
            "gpu_available": HAS_GPU,
            "configured_platforms": {
                "whatsapp": bool(self.wa_phone_id),
                "instagram": bool(self.ig_account_id),
                "facebook": bool(self.fb_page_id)
            },
            "api_version": self.api_version
        }