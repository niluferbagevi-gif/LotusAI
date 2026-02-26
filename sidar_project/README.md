# SİDAR — Yazılım Mühendisi AI Asistanı

> **LotusAI** ekosisteminden ilham alınmış bağımsız bir yazılım mühendisi AI projesi.

```
 ╔══════════════════════════════════════════════╗
 ║  ███████╗██╗██████╗  █████╗ ██████╗          ║
 ║  ██╔════╝██║██╔══██╗██╔══██╗██╔══██╗         ║
 ║  ███████╗██║██║  ██║███████║██████╔╝         ║
 ║  ╚════██║██║██║  ██║██╔══██║██╔══██╗         ║
 ║  ███████║██║██████╔╝██║  ██║██║  ██║         ║
 ║  ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝         ║
 ║  Yazılım Mimarı & Baş Mühendis AI  v1.0.0   ║
 ╚══════════════════════════════════════════════╝
```

---

## Proje Hakkında

**Sidar**, LotusAI çok-ajanlı sistemindeki yazılım mühendisi personasından bağımsız bir proje olarak türetilmiştir. Kod yönetimi, sistem izleme, GitHub entegrasyonu ve güvenli dosya işlemleri konularında uzmanlaşmış bir AI asistanıdır.

### Karakter Profili

| Özellik | Değer |
|---------|-------|
| Ad | SİDAR |
| Rol | Yazılım Mimarı & Baş Mühendis |
| Kişilik | Analitik, disiplinli, geek ruhu |
| İletişim | Minimal ve öz; gereksiz söz yok |
| Karar verme | Veri tabanlı, duygusal değil |
| Model | `qwen2.5-coder:7b` (Ollama) |

---

## Özellikler

### Kod Yönetimi (CodeManager)
- PEP 8 uyumlu Python dosyası okuma/yazma
- Yazılımdan önce otomatik sözdizimi doğrulama
- JSON doğrulama
- Dizin listeleme
- Proje genelinde kod denetimi (audit)
- Metrik takibi (okunan/yazılan/doğrulanan)

### OpenClaw Güvenlik Sistemi (SecurityManager)
| Seviye | Okuma | Yazma | Terminal |
|--------|-------|-------|---------|
| `restricted` | ✓ | ✗ | ✗ |
| `sandbox` | ✓ | Yalnızca `/temp` | ✗ |
| `full` | ✓ | Her yer | ✓ |

### Sistem Sağlığı (SystemHealthManager)
- CPU kullanım izleme
- RAM kullanım izleme
- GPU/CUDA bilgisi ve VRAM takibi
- GPU bellek optimizasyonu (VRAM temizleme + Python GC)

### GitHub Entegrasyonu (GitHubManager)
- Depo bilgisi ve istatistikleri
- Son commit listesi
- Uzak dosya okuma
- Branch listeleme
- Kod arama

### Akıllı İşleme Motoru
- **AutoHandle**: Örüntü tabanlı hızlı komut eşleme (LLM gerektirmez)
- **ReAct Döngüsü**: LLM + araç çağrısı ile karmaşık sorgular
- **Konuşma Belleği**: Thread-safe çoklu tur bellek yönetimi

---

## Kurulum

### 1. Bağımlılıkları Yükle

```bash
cd sidar_project
pip install -r requirements.txt
```

### 2. Çevre Değişkenlerini Ayarla

```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

### 3. Ollama Kur ve Modeli İndir

```bash
# Ollama kurulumu: https://ollama.ai
ollama pull qwen2.5-coder:7b
ollama serve
```

---

## Kullanım

### İnteraktif Mod

```bash
python main.py
```

```
Sen  > Sidar, mevcut dizini listele
Sidar > 📁 ./
  📂 agent/
  📂 core/
  📂 managers/
  📂 tests/
  📄 config.py  (2.1 KB)
  📄 main.py    (4.3 KB)
  ...

Sen  > agents/sidar_agent.py dosyasını oku ve özetle
Sidar > [agents/sidar_agent.py]
...

Sen  > GPU belleğini optimize et
Sidar > GPU VRAM temizlendi: 0.0 MB boşaltıldı
        Python GC çalıştırıldı. ✓
```

### Tek Komut Modu

```bash
python main.py -c "Proje dizinini listele"
python main.py --status
python main.py --level full -c "Sistemi denetle"
```

### CLI Seçenekleri

```
-c, --command   Tek komut çalıştır ve çık
--status        Sistem durumunu göster
--level         Erişim seviyesi (restricted/sandbox/full)
--provider      AI sağlayıcısı (ollama/gemini)
--model         Ollama model adı
--log           Log seviyesi (DEBUG/INFO/WARNING)
```

### Dahili Komutlar

```
.status     Sistem durumunu göster
.clear      Konuşma belleğini temizle
.audit      Proje denetimini çalıştır
.health     Sistem sağlık raporu
.gpu        GPU belleğini optimize et
.github     GitHub bağlantı durumu
.level      Mevcut erişim seviyesini göster
.help       Yardım
.exit       Çıkış
```

---

## Örnek Komutlar

```
# Dizin işlemleri
"Ana klasördeki dosyaları listele"
"managers/ dizinini incele"

# Dosya işlemleri
"config.py dosyasını oku"
"test/sample.py sözdizimini doğrula"

# Sistem
"Sistem sağlık raporu ver"
"GPU belleğini temizle"
"Sistemi denetle ve teknik rapor ver"

# GitHub
"Son 10 commit'i listele"
"GitHub depo bilgilerini göster"
"GitHub'dan README.md dosyasını oku"

# Güvenlik
"Erişim seviyemi göster"
"OpenClaw güvenlik durumu nedir?"
```

---

## Proje Yapısı

```
sidar_project/
├── agent/
│   ├── __init__.py
│   ├── definitions.py      # Sidar karakter profili ve sistem talimatı
│   ├── sidar_agent.py      # Ana ajan (ReAct döngüsü)
│   └── auto_handle.py      # Otomatik örüntü işleyici
├── core/
│   ├── __init__.py
│   ├── memory.py           # Thread-safe konuşma belleği
│   └── llm_client.py       # Ollama / Gemini istemcisi
├── managers/
│   ├── __init__.py
│   ├── code_manager.py     # Dosya operasyonları ve sözdizimi
│   ├── system_health.py    # Sistem izleme
│   ├── github_manager.py   # GitHub API entegrasyonu
│   └── security.py         # OpenClaw erişim kontrolü
├── tests/
│   ├── __init__.py
│   └── test_sidar.py       # Birim testleri
├── temp/                   # Sandbox modunda yazma dizini
├── logs/                   # Log dosyaları
├── config.py               # Merkezi yapılandırma
├── main.py                 # Giriş noktası & CLI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Testleri Çalıştır

```bash
cd sidar_project
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Geliştirme

```bash
# Kod biçimlendirme
black .

# Lint kontrolü
flake8 . --max-line-length=100

# Tür kontrolü
mypy . --ignore-missing-imports
```

---

## Lisans

Bu proje LotusAI ekosisteminin bir parçasıdır.
