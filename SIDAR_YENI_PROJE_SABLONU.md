# Sidar Benzeri Yazılım Mühendisi Ajanı — Yeni Proje Şablonu

Bu doküman, LotusAI içindeki **Sidar** ajanının yeteneklerinden ilham alan yeni bir proje oluşturmanız için pratik bir yol haritası sunar.

## 1) Proje Vizyonu

**Amaç:** Kod yazabilen, kodu okuyup analiz edebilen, hataları sınıflandıran, sistem denetimi yapan ve güvenli erişim seviyeleriyle çalışan bir mühendis ajan geliştirmek.

Önerilen isimler:
- NovaEngineer
- ArdaArchitect
- CodexOps

## 2) Minimum Ürün (MVP) Özellikleri

İlk sürümde aşağıdaki 6 yetenek yeterlidir:
1. **Dosya okuma / yazma** (izin kontrollü)
2. **Syntax doğrulama** (Python + JSON)
3. **Basit hata analizi** (traceback sınıflandırma)
4. **Sistem denetimi** (CPU/RAM/GPU bilgisi)
5. **Kod kalite skoru** (heuristic)
6. **Ajan komut yönlendirme** (metinden aksiyon çıkarma)

## 3) Önerilen Teknik Mimari

```text
new-project/
├── agents/
│   └── engineer.py
├── core/
│   ├── access.py
│   ├── memory.py
│   └── runtime.py
├── managers/
│   ├── code_manager.py
│   └── system_health.py
├── tests/
│   ├── test_access.py
│   └── test_engineer.py
├── config.py
├── main.py
└── README.md
```

## 4) Erişim Seviyeleri (Önemli)

Sidar mantığına benzer güvenli model:
- `restricted`: Sadece okuma + analiz
- `sandbox`: Yalnızca güvenli klasöre yazma
- `full`: Tam erişim

Bu model, ajanın üretim ortamında güvenli kullanılmasını sağlar.

## 5) Başlangıç Kod İskeleti (Python)

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class AccessLevel(str, Enum):
    RESTRICTED = "restricted"
    SANDBOX = "sandbox"
    FULL = "full"

@dataclass
class EngineerMetrics:
    files_read: int = 0
    files_written: int = 0
    syntax_checks: int = 0
    syntax_errors: int = 0

class EngineerAgent:
    def __init__(self, access_level: AccessLevel, sandbox_dir: Path):
        self.access_level = access_level
        self.sandbox_dir = sandbox_dir
        self.metrics = EngineerMetrics()

    def can_write(self, path: Path) -> bool:
        if self.access_level == AccessLevel.RESTRICTED:
            return False
        if self.access_level == AccessLevel.SANDBOX:
            return str(path.resolve()).startswith(str(self.sandbox_dir.resolve()))
        return True
```

## 6) 10 Günlük Uygulama Planı

- **Gün 1-2:** Proje iskeleti + config + access kontrolü
- **Gün 3:** Code manager (read/write/list)
- **Gün 4:** Syntax validator (ast + json)
- **Gün 5:** Error analyzer (regex pattern tabanlı)
- **Gün 6:** System health (psutil + opsiyonel GPU)
- **Gün 7:** Agent komut eşleme (intent parser)
- **Gün 8:** Testler + örnek senaryolar
- **Gün 9:** CLI ve loglama
- **Gün 10:** Dokümantasyon + release

## 7) Üretim İçin Tavsiyeler

- Tüm yazma işlemlerine izin kontrolü ekleyin.
- Kritik işlemler için audit log tutun.
- Hata analizini `error_type + severity + solution` formatında standartlaştırın.
- Ajanın her cevabında “ne yaptı / ne yapamadı / neden” bilgisini verin.

## 8) İsterseniz Bir Sonraki Adım

Bu şablona göre bir sonraki adımda sizin için:
1. Tam çalışan başlangıç repo yapısını,
2. `EngineerAgent` sınıfının ilk sürümünü,
3. 6–8 adet pytest testini
hazırlayabilirim.
