# 🌿 LotusAI - Multi-Agent AI Assistant System

<div align="center">

![LotusAI Logo](https://cdn-icons-png.flaticon.com/512/4712/4712035.png)

**Çok Ajanlı Yapay Zeka Asistan Sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

[English](#english) | [Türkçe](#turkish)

</div>

---

## 🇹🇷 Turkish

### 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Ajanlar](#ajanlar)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Yapılandırma](#yapılandırma)
- [API Dokümantasyonu](#api-dokümantasyonu)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

---

### 🎯 Genel Bakış

LotusAI, **6 özelleşmiş yapay zeka ajanı** ile çalışan gelişmiş bir işletme yönetim sistemidir. Her ajan farklı bir uzmanlık alanında görev yapar ve birlikte senkronize bir ekosistem oluşturur.

**Temel Yetenekler:**
- 🧠 **Multi-Agent System**: 6 özelleşmiş AI ajan
- 🚀 **GPU Acceleration**: CUDA ile hızlandırılmış işlemler
- 🎤 **Voice Interface**: Türkçe ses tanıma ve sentezi
- 📱 **Multi-Platform**: WhatsApp, Instagram, Facebook entegrasyonu
- 🔒 **Face Recognition**: Güvenlik için yüz tanıma
- 📊 **Real-time Analytics**: Canlı veri analizi
- 🌐 **Web Dashboard**: Modern web arayüzü

---

### ✨ Özellikler

#### 🤖 Yapay Zeka & NLP
- Google Gemini API entegrasyonu
- Yerel LLM desteği (Ollama)
- Türkçe NLP ve duygu analizi
- Vector database (ChromaDB)
- Konuşma hafızası
- Multi-turn conversation

#### 🎙️ Ses İşleme
- Türkçe konuşma tanıma (STT)
- Gerçekçi ses sentezi (TTS)
- Edge-TTS ve Coqui-TTS desteği
- Gerçek zamanlı ses akışı

#### 👁️ Görüntü İşleme
- Yüz tanıma ve kimlik doğrulama
- Kamera tabanlı güvenlik
- OCR (Optik Karakter Tanıma)
- Belge tarama

#### 💬 Mesajlaşma
- WhatsApp Business API
- Instagram Direct Message
- Facebook Messenger
- Webhook entegrasyonu

#### 📊 İşletme Yönetimi
- Muhasebe ve finans takibi
- Stok yönetimi
- Rezervasyon sistemi
- Paket servis entegrasyonu
- Sosyal medya yönetimi

#### 🚀 Performans
- NVIDIA GPU hızlandırma
- Multi-threading
- Asenkron işlemler
- Cache sistemi
- Auto-scaling

---

### 🏗️ Mimari
```
LotusAI/
├── 🧠 Core System
│   ├── Runtime Context (Merkezi koordinasyon)
│   ├── Memory Manager (Konuşma hafızası)
│   ├── Security Module (Yüz tanıma)
│   └── Audio Handler (Ses I/O)
│
├── 🤖 Agents (6 AI Agent)
│   ├── Atlas (Proje Yöneticisi)
│   ├── Sidar (Yazılım Mimarı)
│   ├── Kurt (Finans Stratejisti)
│   ├── Gaya (İşletme Müdürü)
│   ├── Poyraz (Medya Direktörü)
│   └── Kerberos (Güvenlik Şefi)
│
├── 🛠️ Managers (10 Specialized Manager)
│   ├── Accounting (Muhasebe)
│   ├── Finance (Finans & Borsa)
│   ├── Operations (Operasyon)
│   ├── Media (Sosyal Medya)
│   ├── Messaging (WhatsApp/Instagram)
│   ├── Delivery (Paket Servis)
│   ├── Camera (Kamera Yönetimi)
│   ├── Code Manager (Kod Yönetimi)
│   ├── NLP (Doğal Dil İşleme)
│   └── System Health (Sistem İzleme)
│
└── 🌐 Web Interface
    ├── Flask Server
    ├── RESTful API
    ├── WebSocket (Real-time)
    └── PWA Support
```

---

### 👥 Ajanlar

#### 🌐 **ATLAS** - Proje Yöneticisi
**Roller:** Liderlik, Koordinasyon, Strateji  
**Yetenekler:**
- Proje planlama ve yönetim
- Ekip koordinasyonu
- Genel sistem kontrolü
- Karar mekanizmaları

**Örnek Görevler:**
```
"Atlas, bugünün önceliklerini belirle"
"Ekip durumunu raporla"
"Haftalık analiz hazırla"
```

#### 💻 **SİDAR** - Yazılım Mimarı
**Roller:** Kod Geliştirme, Teknik Destek  
**Yetenekler:**
- Kod yazma ve debugging
- Sistem optimizasyonu
- Teknik dokümantasyon
- API geliştirme

**Örnek Görevler:**
```
"Sidar, Python'da bir rezervasyon sistemi yaz"
"Bu kodu optimize et"
"API dokümantasyonu oluştur"
```

#### 🐺 **KURT** - Finans Stratejisti
**Roller:** Finans Analizi, Borsa Takibi  
**Yetenekler:**
- Teknik analiz (RSI, EMA, MACD)
- Kripto para takibi
- Finansal raporlama
- Grafik oluşturma

**Örnek Görevler:**
```
"Kurt, BTC/USDT analizi yap"
"Piyasa özeti ver"
"Haftalık finans raporu hazırla"
```

#### 🪷 **GAYA** - İşletme Müdürü
**Roller:** Operasyon, Müşteri İlişkileri  
**Yetenekler:**
- Rezervasyon yönetimi
- Stok takibi
- Menü yönetimi
- Müşteri hizmetleri

**Örnek Görevler:**
```
"Gaya, yarın 4 kişilik saat 19:00 rezervasyon"
"Stok durumunu kontrol et"
"Bugünkü rezervasyonları listele"
```

#### 🌪️ **POYRAZ** - Medya Direktörü
**Roller:** Sosyal Medya, İçerik Üretimi  
**Yetenekler:**
- Sosyal medya yönetimi
- İçerik planlama
- Trend analizi
- AI görsel oluşturma

**Örnek Görevler:**
```
"Poyraz, Instagram için içerik öner"
"Güncel trendleri analiz et"
"AI ile bir kapak görseli oluştur"
```

#### 🛡️ **KERBEROS** - Güvenlik Şefi
**Roller:** Güvenlik, Kimlik Doğrulama  
**Yetenekler:**
- Yüz tanıma
- Kullanıcı yönetimi
- Güvenlik logları
- Erişim kontrolü

**Örnek Görevler:**
```
"Kerberos, kimlik doğrulama yap"
"Güvenlik loglarını göster"
"Yeni kullanıcı ekle"
```

---

### 🚀 Kurulum

#### Sistem Gereksinimleri

**Minimum:**
- Python 3.11+
- 8GB RAM
- 10GB Disk

**Önerilen:**
- Python 3.11+
- 16GB RAM
- NVIDIA GPU (CUDA 11.8)
- 50GB Disk

#### 1. Depoyu Klonlayın
```bash
git clone https://github.com/niluferbagevi-gif/LotusAI.git
cd LotusAI
```

#### 2. Sanal Ortam Oluşturun

**Conda (Önerilen):**
```bash
conda env create -f environment.yml
conda activate lotus-ai
```

**Pip:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### 3. Ortam Değişkenlerini Ayarlayın
```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin:
```ini
# AI API
GEMINI_API_KEY=your_gemini_api_key

# GPU
USE_GPU=True

# Meta API
META_ACCESS_TOKEN=your_meta_token
WHATSAPP_PHONE_ID=your_phone_id
```

#### 4. Sistemi Başlatın
```bash
python main.py
```

Web arayüzü: `http://localhost:5000`

---

### 💻 Kullanım

#### Temel Kullanım

**1. Terminal Modu:**
```bash
python main.py
```

**2. Web Arayüzü:**
```
http://localhost:5000
```

**3. Sesli Komutlar:**
```
[Space] tuşuna basın ve konuşun
```

#### Örnek Komutlar

**Genel:**
```
"Merhaba"
"Sistem durumu nedir?"
"Bugünün özetini ver"
```

**Rezervasyon:**
```
"Yarın saat 19:00 için 4 kişilik masa ayır"
"Bugünkü rezervasyonları göster"
"Rezervasyon #123'ü iptal et"
```

**Finans:**
```
"BTC fiyatı nedir?"
"ETH/USDT analizi yap"
"Kasa bakiyesi ne kadar?"
```

**Sosyal Medya:**
```
"Instagram'da yeni ne var?"
"Türkiye trendleri nedir?"
"Yarın için içerik öner"
```

**Stok:**
```
"Domates stokunu kontrol et"
"Zeytinyağı ekle 5 litre"
"Kritik stokları göster"
```

---

### ⚙️ Yapılandırma

#### GPU Ayarları
```python
# config.py
USE_GPU = True  # GPU kullanımı
```

#### Agent Ayarları
```python
# agents/definitions.py
AGENTS_CONFIG = {
    "ATLAS": {
        "name": "Atlas",
        "emoji": "🌐",
        "color": "#29b6f6"
    }
}
```

#### Sidar Güvenli Dosya (Sandbox) Modu

Sidar'ı proje ana klasöründe güvenli dosya erişimi ile çalıştırmak için `.env` içinde şu ayarları kullanın:

```env
ACCESS_LEVEL=sandbox
WORK_DIR=/workspace/LotusAI
```

- `ACCESS_LEVEL=sandbox`: Sidar ve CodeManager için güvenli yazma/okuma modu.
- `WORK_DIR`: Sandbox kök dizinidir. Sidar sadece bu dizin ve altındaki dosyalarda işlem yapar.

Sistem başlatma:

```bash
python main.py
```

Örnek Sidar komutları:

```text
"Sidar, ana klasördeki dosyaları listele"
"Sidar, agents/sidar.py dosyasını oku"
"Sidar, core/utils.py içine güvenli biçimde şu fonksiyonu ekle"
```

#### Ses Ayarları
```python
# config.py
VOICE_ENABLED = True
USE_XTTS = True  # Yerel TTS
```

---

### 📚 API Dokümantasyonu

#### REST API Endpoints

**Chat:**
```http
POST /api/chat
Content-Type: multipart/form-data

{
  "message": "Merhaba",
  "target_agent": "ATLAS",
  "file": <file>
}
```

**Chat History:**
```http
GET /api/chat_history?agent=ATLAS
```

**Voice Toggle:**
```http
POST /api/toggle_voice
```

**Webhook:**
```http
POST /webhook
Content-Type: application/json

{
  "entry": [...]
}
```

#### Python API
```python
from lotus_system import LotusSystem

# Initialize
lotus = LotusSystem()

# Get response
response = await lotus.engine.get_response(
    agent="ATLAS",
    user_input="Merhaba",
    security_result=("ONAYLI", user_data, None)
)

print(response['content'])
```

---

### 🔧 Gelişmiş Özellikler

#### 1. Custom Agent Oluşturma
```python
# agents/custom_agent.py
from agents.definitions import AGENTS_CONFIG

AGENTS_CONFIG["CUSTOM"] = {
    "name": "Custom Agent",
    "emoji": "🎯",
    "role": "Custom Role",
    "capabilities": ["task1", "task2"],
    "system_prompt": "Your system prompt here"
}
```

#### 2. Manager Genişletme
```python
# managers/custom_manager.py
class CustomManager:
    def __init__(self):
        self.lock = threading.RLock()
    
    def custom_function(self):
        # Your code here
        pass
```

#### 3. Webhook Handler
```python
# server.py
@app.route('/custom_webhook', methods=['POST'])
def custom_webhook():
    data = request.json
    # Process webhook
    return jsonify({"status": "ok"})
```

---

### 🐛 Sorun Giderme

#### PyAudio Kurulum Hatası

**Windows:**
```bash
# Wheel dosyasını indirin
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### CUDA Hatası
```bash
# CUDA versiyonunu kontrol edin
nvidia-smi

# PyTorch CUDA versiyonunu kontrol edin
python -c "import torch; print(torch.cuda.is_available())"
```

#### Face Recognition Hatası
```bash
# CMake yükleyin
pip install cmake

# dlib yükleyin
pip install dlib

# face-recognition yükleyin
pip install face-recognition
```

---

### 📊 Performans

**Sistem Gereksinimleri vs Performans:**

| Özellik | CPU Only | GPU (RTX 3070 Ti) |
|---------|----------|-------------------|
| Response Time | 2-5 sn | 0.5-1 sn |
| Concurrent Users | 5 | 50+ |
| Face Recognition | 2 fps | 30 fps |
| TTS Generation | 1x | 5x |

---

### 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing`)
5. Pull Request açın

**Kod Standartları:**
- PEP 8 uyumlu
- Type hints kullanın
- Docstring ekleyin
- Test yazın

---

### 📝 Changelog

#### v2.6.0 (2026-02-22)
- ✅ Tekil sürüm kaynağına geçiş (Config.VERSION)
- ✅ Launcher/Core/Server sürüm senkronizasyonu
- ✅ Webhook doğrulamada `WEBHOOK_VERIFY_TOKEN` zorunlu hâle getirildi
- ✅ README bağlantıları ve iletişim bilgileri güncellendi

#### v2.5.3 (2026-02-10)
- Full code refactoring
- Type hints %100
- Better error handling
- Metrics tracking
- Improved documentation

#### v2.0.0 (2025-12-01)
- Multi-agent system
- Web interface
- PWA support

---

### 📄 Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

---

### 👨‍💻 Geliştirici

**Halil Sevim**  
📧 Email: halilsevim@hotmail.com  
🌐 Website: https://yourwebsite.com  
💼 LinkedIn: https://linkedin.com/in/yourprofile

---

### 🙏 Teşekkürler

- Google Gemini AI
- Anthropic Claude
- OpenCV Community
- PyTorch Team
- HuggingFace

---

### 📞 Destek

**Sorularınız mı var?**

- 📧 Email: support@lotusai.com
- 💬 Discord: [LotusAI Community](#)
- 📖 Docs: [docs.lotusai.com](#)
- 🐛 Issues: [GitHub Issues](https://github.com/niluferbagevi-gif/LotusAI/issues)

---

<div align="center">

**Made with ❤️ in Turkey**

⭐ Star us on GitHub — it helps!

[⬆ Back to Top](#-lotusai---multi-agent-ai-assistant-system)

</div>
```

---

## 📜 LICENSE (MIT)
```
MIT License

Copyright (c) 2026 Halil Sevim - LotusAI Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Additional Terms for LotusAI:

1. ATTRIBUTION
   - You must give appropriate credit to the original author
   - Provide a link to the original repository
   - Indicate if changes were made

2. COMMERCIAL USE
   - Commercial use is permitted
   - No additional fees or royalties required
   - Attribution must be maintained

3. MODIFICATIONS
   - You may modify and distribute modified versions
   - Modified versions must be clearly marked as such
   - You must release modified versions under the same license

4. THIRD-PARTY COMPONENTS
   - This software uses various third-party libraries
   - Each library maintains its own license
   - See requirements.txt for full list of dependencies

5. AI MODEL USAGE
   - Google Gemini API usage subject to Google's terms
   - Meta APIs subject to Meta's platform policies
   - Local AI models subject to their respective licenses

6. DATA PRIVACY
   - User data must be handled according to GDPR/KVKK
   - Face recognition data must be stored securely
   - Conversation logs must be encrypted

7. DISCLAIMER
   - This software is provided for educational purposes
   - Production use requires proper security audit
   - Author is not liable for misuse or damages

For questions about licensing, contact: license@lotusai.com
