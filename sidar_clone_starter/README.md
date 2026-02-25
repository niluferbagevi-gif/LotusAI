# Sidar Clone Starter (Sıfırdan Kurulum)

Bu klasör, LotusAI içindeki Sidar yaklaşımına benzer bir "yazılım mühendisi ajanı"nı hızlıca ayağa kaldırmak için hazırlanmış başlangıç paketidir.

## Özellikler
- Erişim seviyesi: `restricted`, `sandbox`, `full`
- Dosya okuma / yazma (izin denetimli)
- Python/JSON syntax kontrolü
- Basit traceback hata sınıflandırması
- Sistem denetimi (platform/cpu, opsiyonel RAM)

## Kurulum

```bash
cd /workspace/LotusAI
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pytest psutil
```

## Çalıştırma

```bash
python -m sidar_clone_starter.main
```

## Hızlı kullanım

```python
from pathlib import Path
from sidar_clone_starter.agents.engineer import EngineerAgent
from sidar_clone_starter.config import Settings, AccessLevel

settings = Settings(access_level=AccessLevel.SANDBOX, sandbox_dir=Path("./sidar_clone_starter/sandbox"))
settings.sandbox_dir.mkdir(parents=True, exist_ok=True)
agent = EngineerAgent(settings)

print(agent.handle("audit"))
print(agent.handle("write sidar_clone_starter/sandbox/hello.py | print('merhaba')"))
print(agent.handle("read sidar_clone_starter/sandbox/hello.py"))
```

## Test

```bash
pytest -q sidar_clone_starter/tests
```
