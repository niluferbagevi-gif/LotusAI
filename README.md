🌿 LotusAI - Multi-Agent AI Assistant System

<div align="center">

Çok Ajanlı Yapay Zeka Asistan Sistemi

English | Türkçe

</div>

🇹🇷 Turkish

📋 İçindekiler

Genel Bakış

Özellikler

Mimari

Ajanlar

Kurulum

Kullanım

Yapılandırma

API Dokümantasyonu

Katkıda Bulunma

Lisans

🎯 Genel Bakış

LotusAI, 6 özelleşmiş yapay zeka ajanı ile çalışan gelişmiş bir işletme yönetim sistemidir. Her ajan farklı bir uzmanlık alanında görev yapar ve birlikte senkronize bir ekosistem oluşturur.

Temel Yetenekler:

🧠 Multi-Agent System: 6 özelleşmiş AI ajan

🚀 GPU Acceleration: CUDA ile hızlandırılmış işlemler

🐙 GitHub Integration: Bulut tabanlı repo analizi ve kod okuma (YENİ!)

🎤 Voice Interface: Türkçe ses tanıma ve sentezi

📱 Multi-Platform: WhatsApp, Instagram, Facebook entegrasyonu

🔒 Face Recognition: Güvenlik için yüz tanıma

📊 Real-time Analytics: Canlı veri analizi

🌐 Web Dashboard: Modern web arayüzü

✨ Özellikler

🤖 Yapay Zeka & NLP

Google Gemini API entegrasyonu

Yerel LLM desteği (Ollama)

Sidar'a Özel: qwen2.5-coder ile profesyonel kodlama yeteneği

ReAct (Reason+Act): Akıllı düşünme döngüsü ve kendi kendine dosya okuma

Türkçe NLP ve duygu analizi

Konuşma hafızası

🐙 Yazılım Mimarisi (Sidar)

GitHub Bağlantısı: Uzaktaki repoları (niluferbagevi-gif/LotusAI) analiz etme

Kod Analizi: Yerel ve bulut tabanlı dosya okuma/yazma

Otomatik Denetim: Proje yapısını ve donanım sağlığını denetleme

Sözdizimi Kontrolü: Yazılan kodlarda otomatik syntax ve JSON validasyonu

🎙️ Ses İşleme

Türkçe konuşma tanıma (STT)

Gerçekçi ses sentezi (TTS)

Edge-TTS ve Coqui-TTS desteği

Gerçek zamanlı ses akışı

👁️ Görüntü İşleme

Yüz tanıma ve kimlik doğrulama

Kamera tabanlı güvenlik

OCR (Optik Karakter Tanıma) ve fatura analizi

💬 Mesajlaşma

WhatsApp Business API

Instagram Direct Message

Facebook Messenger

Webhook entegrasyonu

📊 İşletme Yönetimi

Muhasebe ve finans takibi

Stok yönetimi

Rezervasyon sistemi

Paket servis entegrasyonu

Sosyal medya yönetimi

🚀 Performans

NVIDIA GPU hızlandırma

Multi-threading

Asenkron işlemler

Cache sistemi

Erişim Seviyesi (Access Level) Kontrolü (Restricted/Sandbox/Full)

🏗️ Mimari

LotusAI/
├── 🧠 Core System
│   ├── Runtime Context (Merkezi koordinasyon)
│   ├── Memory Manager (Konuşma hafızası)
│   ├── Security Module (Yüz tanıma)
│   └── Audio Handler (Ses I/O)
│
├── 🤖 Agents (6 AI Agent)
│   ├── Atlas (Proje Yöneticisi)
│   ├── Sidar (Yazılım Mimarı & GitHub Uzmanı) [GÜNCELLENDİ]
│   ├── Kurt (Finans Stratejisti)
│   ├── Gaya (İşletme Müdürü)
│   ├── Poyraz (Medya Direktörü)
│   └── Kerberos (Güvenlik Şefi)
│
├── 🛠️ Managers (11 Specialized Manager)
│   ├── Accounting (Muhasebe)
│   ├── Finance (Finans & Borsa)
│   ├── Operations (Operasyon)
│   ├── Media (Sosyal Medya)
│   ├── Messaging (WhatsApp/Instagram)
│   ├── Delivery (Paket Servis)
│   ├── Camera (Kamera Yönetimi)
│   ├── Code Manager (Kod Yönetimi)
│   ├── GitHub Manager (Repo Entegrasyonu) [YENİ]
│   ├── NLP (Doğal Dil İşleme)
│   └── System Health (Sistem İzleme)
│
└── 🌐 Web Interface
    ├── Flask Server
    ├── RESTful API
    ├── WebSocket (Real-time)
    └── PWA Support


👥 Ajanlar

🌐 ATLAS - Proje Yöneticisi

Roller: Liderlik, Koordinasyon, Strateji

Yetenekler:

Proje planlama ve yönetim

Ekip koordinasyonu

Genel sistem kontrolü

Karar mekanizmaları

Örnek Görevler:

"Atlas, bugünün önceliklerini belirle"
"Ekip durumunu raporla"
"Haftalık analiz hazırla"


💻 SİDAR - Yazılım Mimarı (GÜNCELLENDİ)

Roller: Kod Geliştirme, Teknik Destek, GitHub Yöneticisi

Yetenekler:

Kod yazma ve debugging (qwen2.5-coder kullanır)

Sistem optimizasyonu ve donanım denetimi

GitHub repo analizi ve commit takibi

Yerel dosya okuma ve düzenleme

Örnek Görevler:

"Sidar, GitHub'daki son commit'leri listele"
"Bu dosyayı oku ve hataları düzelt"
"Sistemi denetle ve GPU durumunu raporla"


🐺 KURT - Finans Stratejisti

Roller: Finans Analizi, Borsa Takibi

Yetenekler:

Teknik analiz (RSI, EMA, MACD)

Kripto para takibi

Finansal raporlama

Grafik oluşturma

Örnek Görevler:

"Kurt, BTC/USDT analizi yap"
"Piyasa özeti ver"
"Haftalık finans raporu hazırla"


🪷 GAYA - İşletme Müdürü

Roller: Operasyon, Müşteri İlişkileri

Yetenekler:

Rezervasyon yönetimi

Stok takibi

Menü yönetimi

Fatura okuma ve işleme

Örnek Görevler:

"Gaya, yarın 4 kişilik saat 19:00 rezervasyon"
"Stok durumunu kontrol et"
"Bu faturayı oku ve sisteme işle"


🌪️ POYRAZ - Medya Direktörü

Roller: Sosyal Medya, İçerik Üretimi

Yetenekler:

Sosyal medya yönetimi

İçerik planlama

Trend analizi

AI görsel oluşturma

Örnek Görevler:

"Poyraz, Instagram için içerik öner"
"Güncel trendleri analiz et"
"AI ile bir kapak görseli oluştur"


🛡️ KERBEROS - Güvenlik Şefi

Roller: Güvenlik, Kimlik Doğrulama

Yetenekler:

Yüz tanıma

Kullanıcı yönetimi

Güvenlik logları

Erişim kontrolü

Örnek Görevler:

"Kerberos, kimlik doğrulama yap"
"Güvenlik loglarını göster"
"Yeni kullanıcı ekle"


🚀 Kurulum

Sistem Gereksinimleri

Minimum:

Python 3.11+

8GB RAM

10GB Disk

Önerilen:

Python 3.11+

16GB RAM

NVIDIA GPU (CUDA 11.8)

50GB Disk

1. Depoyu Klonlayın

git clone [https://github.com/niluferbagevi-gif/LotusAI.git](https://github.com/niluferbagevi-gif/LotusAI.git)
cd LotusAI


2. Sanal Ortam Oluşturun

Conda (Önerilen):

conda env create -f environment.yml
conda activate lotus-ai


Pip:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt


3. Ortam Değişkenlerini Ayarlayın

cp .env.example .env


.env dosyasını düzenleyin:

# AI API
GEMINI_API_KEY=your_gemini_api_key

# GitHub (Yeni)
GITHUB_REPO=niluferbagevi-gif/LotusAI
GITHUB_TOKEN=your_github_token_here

# GPU
USE_GPU=True

# Meta API
META_ACCESS_TOKEN=your_meta_token
WHATSAPP_PHONE_ID=your_phone_id


4. Sistemi Başlatın

python main.py


Web arayüzü: http://localhost:5000

💻 Kullanım

Temel Kullanım

1. Terminal Modu:

python main.py


2. Web Arayüzü:

http://localhost:5000


3. Sesli Komutlar:

[Space] tuşuna basın ve konuşun


Örnek Komutlar

Genel:

"Merhaba"
"Sistem durumu nedir?"
"Bugünün özetini ver"


GitHub & Kod (Sidar):

"GitHub reposundaki dosyaları listele"
"GitHub'daki main.py dosyasını oku ve analiz et"
"Son commit geçmişini göster"


Rezervasyon:

"Yarın saat 19:00 için 4 kişilik masa ayır"
"Bugünkü rezervasyonları göster"
"Rezervasyon #123'ü iptal et"


Finans:

"BTC fiyatı nedir?"
"ETH/USDT analizi yap"
"Kasa bakiyesi ne kadar?"


Sosyal Medya:

"Instagram'da yeni ne var?"
"Türkiye trendleri nedir?"
"Yarın için içerik öner"


Stok:

"Domates stokunu kontrol et"
"Zeytinyağı ekle 5 litre"
"Kritik stokları göster"


⚙️ Yapılandırma

GPU Ayarları

# config.py
USE_GPU = True  # GPU kullanımı


Sidar Güvenli Dosya (Sandbox) Modu

Sidar'ı proje ana klasöründe güvenli dosya erişimi ile çalıştırmak için .env içinde şu ayarları kullanın:

ACCESS_LEVEL=sandbox
WORK_DIR=/workspace/LotusAI


ACCESS_LEVEL=restricted: Sadece bilgi verir, işlem yapmaz.

ACCESS_LEVEL=sandbox: Güvenli okuma/yazma yapar (Önerilen).

ACCESS_LEVEL=full: Terminal komutları dahil tam yetki (Dikkatli kullanın).

Sistem başlatma:

python main.py


Örnek Sidar komutları:

"Sidar, ana klasördeki dosyaları listele"
"Sidar, GitHub'dan config.py dosyasını oku"
"Sidar, core/utils.py içine güvenli biçimde şu fonksiyonu ekle"


Ses Ayarları

# config.py
VOICE_ENABLED = True
USE_XTTS = True  # Yerel TTS


📚 API Dokümantasyonu

REST API Endpoints

Chat:

POST /api/chat
Content-Type: multipart/form-data

{
  "message": "Merhaba",
  "target_agent": "ATLAS",
  "file": <file>
}


Chat History:

GET /api/chat_history?agent=ATLAS


Voice Toggle:

POST /api/toggle_voice


Webhook:

POST /webhook
Content-Type: application/json

{
  "entry": [...]
}


Python API

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


🔧 Gelişmiş Özellikler

1. Custom Agent Oluşturma

# agents/definitions.py
AGENTS_CONFIG["CUSTOM"] = {
    "name": "Custom Agent",
    "emoji": "🎯",
    "role": "Custom Role",
    "capabilities": ["task1", "task2"],
    "system_prompt": "Your system prompt here"
}


2. Manager Genişletme

# managers/custom_manager.py
class CustomManager:
    def __init__(self):
        self.lock = threading.RLock()
    
    def custom_function(self):
        # Your code here
        pass


🐛 Sorun Giderme

PyGithub Kurulumu (Yeni)

GitHub entegrasyonu için:

pip install PyGithub


PyAudio Kurulum Hatası

Windows:

# Wheel dosyasını indirin
# [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl


Linux:

sudo apt-get install portaudio19-dev
pip install pyaudio


CUDA Hatası

# CUDA versiyonunu kontrol edin
nvidia-smi

# PyTorch CUDA versiyonunu kontrol edin
python -c "import torch; print(torch.cuda.is_available())"


Face Recognition Hatası

# CMake yükleyin
pip install cmake

# dlib yükleyin
pip install dlib

# face-recognition yükleyin
pip install face-recognition


📊 Performans

Sistem Gereksinimleri vs Performans:

Özellik

CPU Only

GPU (RTX 3070 Ti)

Response Time

2-5 sn

0.5-1 sn

Concurrent Users

5

50+

Face Recognition

2 fps

30 fps

TTS Generation

1x

5x

GitHub Analysis

3-5 sn

1-2 sn

🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

Fork edin

Feature branch oluşturun (git checkout -b feature/amazing)

Commit yapın (git commit -m 'Add amazing feature')

Push edin (git push origin feature/amazing)

Pull Request açın

Kod Standartları:

PEP 8 uyumlu

Type hints kullanın

Docstring ekleyin

Test yazın

📝 Changelog

v2.6.0 (2026-02-23)

🚀 GitHub Entegrasyonu: GithubManager eklendi, repo analizi yeteneği geldi.

🔄 ReAct Döngüsü: Sidar'a "düşünme ve eyleme geçme" yeteneği (Auto-Handle) eklendi.

🔐 Access Level: Dinamik erişim seviyesi (Restricted/Sandbox/Full) tüm sisteme yayıldı.

🛠️ Config: Tüm ayarlar config.py üzerinden merkezi yönetime alındı.

✅ Sürüm Senkronizasyonu: Tüm dosyalar v2.6.0 olarak işaretlendi.

v2.5.3 (2026-02-10)

Full code refactoring

Type hints %100

Better error handling

Metrics tracking

Improved documentation

v2.0.0 (2025-12-01)

Multi-agent system

Web interface

PWA support

📄 Lisans

Bu proje MIT License altında lisanslanmıştır.

👨‍💻 Geliştirici

Halil Sevim 📧 Email: halilsevim@hotmail.com

🌐 Website: https://yourwebsite.com

💼 LinkedIn: https://linkedin.com/in/yourprofile

🙏 Teşekkürler

Google Gemini AI

Anthropic Claude

OpenCV Community

PyTorch Team

HuggingFace

PyGithub Contributors

📞 Destek

Sorularınız mı var?

📧 Email: support@lotusai.com

💬 Discord: LotusAI Community

📖 Docs: docs.lotusai.com

🐛 Issues: GitHub Issues

<div align="center">

Made with ❤️ in Turkey

⭐ Star us on GitHub — it helps!

⬆ Back to Top

</div>

📜 https://www.google.com/search?q=LICENSE (MIT)

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
