# OpenClaw AI — Araştırma Raporu

**Tarih:** 22 Şubat 2026
**Hazırlayan:** LotusAI Research Branch (`claude/research-openclaw-ai-6QKX0`)

---

## 1. OpenClaw Nedir?

OpenClaw (eski adıyla **Clawdbot** ve **Moltbot**), Avustralyalı yazılım geliştirici **Peter Steinberger** tarafından geliştirilen, ücretsiz ve açık kaynaklı otonom bir yapay zeka asistanıdır.

- **Lisans:** MIT
- **GitHub:** [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) — 191.000+ yıldız
- **Resmi Site:** [openclaw.ai](https://openclaw.ai)
- **İlk Yayın:** Kasım 2025
- **Mevcut Durum:** OpenAI tarafından "acqui-hire" yapıldı; yazılım açık kaynak kalmaya devam ediyor

---

## 2. Temel Özellikler

### 2.1 Çok Kanallı Erişim
OpenClaw, kullanıcıların zaten kullandığı mesajlaşma platformları üzerinden çalışır:

| Platform | Destek |
|----------|--------|
| WhatsApp | ✅ |
| Telegram | ✅ |
| Slack | ✅ |
| Discord | ✅ |
| Signal | ✅ |
| iMessage (BlueBubbles) | ✅ |
| Microsoft Teams | ✅ |
| Matrix / Zalo / WebChat | ✅ |

### 2.2 Kalıcı Hafıza (Persistent Memory)
- Kullanıcı tercihlerini öğrenir ve sürekli günceller
- Oturumlar arası bağlamı korur
- "Heartbeat" özelliği: Tetiklenmeden bağımsız olarak proaktif eylem gerçekleştirebilir (gelen kutusu takibi, alarm vb.)

### 2.3 Sistem Entegrasyonu
- Tarayıcı kontrolü (Chrome/Chromium)
- Dosya okuma/yazma ve kabuk komutu (shell command) çalıştırma
- Takvim ve e-posta yönetimi
- Akıllı ev cihazı kontrolü

### 2.4 Beceriler (Skills) & Genişletilebilirlik
- 100+ önceden yapılandırılmış **AgentSkill** eklentisi
- Kullanıcı ihtiyacına göre yeni beceriler otonom olarak yazabilir
- 50+ üçüncü taraf servis entegrasyonu

---

## 3. Teknik Altyapı

### 3.1 Çalışma Ortamı
```
Gereksinim: Node.js ≥ 22
Dil: TypeScript
Kurulum: npm install -g openclaw@latest
```

### 3.2 Mimari Bileşenler
- **Local Gateway:** WebSocket kontrol düzlemi (`ws://127.0.0.1:18789`)
- **Pi Agent Runtime:** RPC destekli ajan çalışma zamanı
- **Browser Control:** Bağımsız Chrome/Chromium örneği
- **Canvas Interface:** A2UI desteği
- **Voice Wake/Talk Mode:** Sesli aktivasyon ve konuşma modu

### 3.3 Mobil Destek
- macOS menü çubuğu uygulaması
- iOS ve Android düğümleri (node)

### 3.4 Desteklenen AI Modelleri (Model Agnostik)
| Model | Sağlayıcı |
|-------|----------|
| Claude (Sonnet, Opus, Haiku) | Anthropic |
| GPT-4o, GPT-4.1 | OpenAI |
| Gemini 2.0 | Google |
| Yerel modeller (llama, mistral vb.) | Ollama |

---

## 4. Güvenlik Modeli

### 4.1 Eşleştirme (Pairing) Sistemi
- Varsayılan olarak bilinmeyen göndericiler bot'a erişemez
- İlk iletişimde "pairing code" doğrulaması gerekir
- `openclaw pairing approve` komutuyla onay

### 4.2 Veri Gizliliği
- Tüm veriler yerel makinede saklanır (`~/.openclaw/`)
- Kullanıcı API anahtarlarını kendi sağlar (kendi maliyeti, kendi gizliliği)

### 4.3 Bilinen Güvenlik Riskleri
- **Prompt Injection:** Kötü amaçlı veriler içine gömülü talimatlar ajanı manipüle edebilir
- **Skill Repository Riski:** Cisco araştırmacıları, üçüncü taraf becerilerde veri sızdırma girişimi tespit etti
- OpenClaw bakımcılarından uyarı: _"Komut satırı çalıştırmayı bilmiyorsanız, bu proje sizin için çok tehlikeli."_

---

## 5. Popülerlik & Büyüme

| Metrik | Değer |
|--------|-------|
| GitHub Yıldızı | 191.000+ |
| Fork | 32.400+ |
| Katkıda Bulunan | 900+ |
| Tahmini Kullanıcı | 300.000–400.000 |
| 1 Haftada Yıldız | 100.000+ (GitHub tarihinde rekor) |

---

## 6. Maliyet Analizi

OpenClaw yazılımı tamamen **ücretsiz ve açık kaynaklıdır**. Maliyet yalnızca AI model API kullanımından kaynaklanır:

| Kullanım Yoğunluğu | Tahmini Aylık Maliyet |
|--------------------|----------------------|
| Hafif Kullanım | $10–$30 |
| Normal Kullanım | $30–$70 |
| Yoğun Otomasyon | $100–$150+ |
| Ollama (Yerel Model) | $0 (API ücreti yok) |

---

## 7. LotusAI ile Entegrasyon Potansiyeli

LotusAI, zaten çok ajanlı bir yapay zeka sistemi olduğundan OpenClaw ile çeşitli noktalarda örtüşme ve entegrasyon potansiyeli mevcuttur:

### 7.1 Benzerlikler
| Özellik | LotusAI | OpenClaw |
|---------|---------|---------|
| Çok Ajan Mimarisi | ✅ (6 ajan) | ✅ (tek ajan, çoklu beceri) |
| Mesajlaşma Entegrasyonu | ✅ (WhatsApp, Instagram) | ✅ (10+ platform) |
| Kalıcı Hafıza | ✅ (SQLite + ChromaDB) | ✅ (yerel depolama) |
| Model Esnekliği | ✅ (Gemini + Ollama) | ✅ (Claude, GPT, Gemini, Ollama) |
| Sesli Arayüz | ✅ (STT/TTS) | ✅ (Voice Wake/Talk) |
| Yerel Çalışma | ✅ | ✅ |

### 7.2 OpenClaw'dan Alınabilecek İlham
1. **Heartbeat (Proaktif Tetikleme):** LotusAI ajanları şu anda yalnızca istek bazlı çalışıyor. OpenClaw'un proaktif heartbeat sistemi uyarlanabilir.
2. **Skill Ekosistemi:** LotusAI'ın manager sistemi, OpenClaw'a benzer dinamik bir skill ekosistemi olarak yeniden tasarlanabilir.
3. **Pairing Güvenlik Modeli:** WhatsApp entegrasyonu için OpenClaw'un eşleştirme sistemi örnek alınabilir.
4. **Genişletilmiş Platform Desteği:** LotusAI'ın mevcut WhatsApp + Instagram entegrasyonu, Telegram, Discord, Signal gibi platformlara genişletilebilir.

---

## 8. Sonuç

OpenClaw, kısa sürede yapay zeka ajanlığı (AI agentic) alanının en dikkat çekici açık kaynak projelerinden biri haline gelmiştir. Peter Steinberger'in OpenAI'ya katılması ve projenin açık kaynak olarak devam etmesi, uzun vadeli sürdürülebilirliğini güçlendirmektedir.

**LotusAI perspektifinden:** OpenClaw, LotusAI'ın rakibi değil, tamamlayıcısı olarak değerlendirilebilir. LotusAI'ın restoran/işletme odaklı çok ajanlı mimarisi ve biyometrik güvenlik altyapısı, OpenClaw'un genel amaçlı ajan yapısından farklı ve özelleşmiş bir niş sunmaktadır. Bununla birlikte, OpenClaw'un proaktif otomasyon ve genişletilebilir skill ekosistemi tasarımından ilham alınması önerilir.

---

## 9. Kaynaklar

- [OpenClaw Resmi Sitesi](https://openclaw.ai/)
- [OpenClaw GitHub Deposu](https://github.com/openclaw/openclaw)
- [Wikipedia — OpenClaw](https://en.wikipedia.org/wiki/OpenClaw)
- [DigitalOcean — What is OpenClaw?](https://www.digitalocean.com/resources/articles/what-is-openclaw)
- [Milvus Blog — Complete Guide](https://milvus.io/blog/openclaw-formerly-clawdbot-moltbot-explained-a-complete-guide-to-the-autonomous-ai-agent.md)
- [Tom's Hardware — OpenAI hires OpenClaw creator](https://www.tomshardware.com/tech-industry/openai-hires-genius-openclaw-creator-but-popular-ai-assistant-will-remain-open-source-sam-altman-says-creator-will-keep-openclaw-open-source)
- [PCWorld — Security Warning](https://www.pcworld.com/article/3064874/openclaw-ai-is-going-viral-dont-install-it.html)
- [Decrypt — OpenClaw Acquisition Offers](https://decrypt.co/358129/openclaw-creator-offers-acquire-ai-sensation-stay-open-source)
