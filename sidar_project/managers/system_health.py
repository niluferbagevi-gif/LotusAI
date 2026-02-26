"""
Sidar Project - Sistem Sağlığı Yöneticisi
CPU, RAM ve GPU izleme; bellek optimizasyonu.
"""

import gc
import logging
import platform
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SystemHealthManager:
    """
    Donanım sağlığını izler ve raporlar.
    GPU bellek optimizasyonu gerçekleştirir.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        self._lock = threading.RLock()
        self._torch_available = self._check_torch()
        self._gpu_available = self._check_gpu()

    # ─────────────────────────────────────────────
    #  BAŞLANGIC KONTROLLERI
    # ─────────────────────────────────────────────

    def _check_torch(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_gpu(self) -> bool:
        if not self.use_gpu or not self._torch_available:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ─────────────────────────────────────────────
    #  CPU & RAM
    # ─────────────────────────────────────────────

    def get_cpu_usage(self) -> Optional[float]:
        try:
            import psutil
            return psutil.cpu_percent(interval=0.5)
        except ImportError:
            return None

    def get_memory_info(self) -> Dict[str, float]:
        """RAM bilgisini GB cinsinden döndür."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total_gb": vm.total / 1e9,
                "used_gb": vm.used / 1e9,
                "available_gb": vm.available / 1e9,
                "percent": vm.percent,
            }
        except ImportError:
            return {}

    # ─────────────────────────────────────────────
    #  GPU
    # ─────────────────────────────────────────────

    def get_gpu_info(self) -> Dict[str, object]:
        """GPU bilgisini döndür."""
        if not self._gpu_available:
            return {"available": False, "reason": "CUDA bulunamadı veya devre dışı"}

        try:
            import torch
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                devices.append({
                    "id": i,
                    "name": props.name,
                    "total_vram_gb": round(total_vram, 2),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "free_gb": round(total_vram - reserved, 2),
                })
            return {"available": True, "device_count": device_count, "devices": devices}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def optimize_gpu_memory(self) -> str:
        """GPU VRAM'ını boşalt ve Python GC çalıştır."""
        freed_mb = 0.0
        if self._gpu_available:
            try:
                import torch
                before = torch.cuda.memory_reserved() / 1e6
                torch.cuda.empty_cache()
                after = torch.cuda.memory_reserved() / 1e6
                freed_mb = before - after
                logger.info("GPU bellek temizlendi: %.1f MB boşaltıldı", freed_mb)
            except Exception as exc:
                logger.warning("GPU bellek temizleme hatası: %s", exc)

        gc.collect()
        return (
            f"GPU VRAM temizlendi: {freed_mb:.1f} MB boşaltıldı\n"
            f"Python GC çalıştırıldı. ✓"
        )

    # ─────────────────────────────────────────────
    #  TAM RAPOR
    # ─────────────────────────────────────────────

    def full_report(self) -> str:
        """Kapsamlı sistem sağlık raporu."""
        lines = ["[Sistem Sağlık Raporu]"]

        # Platform
        lines.append(f"  OS      : {platform.system()} {platform.release()}")
        lines.append(f"  Python  : {platform.python_version()}")

        # CPU
        cpu = self.get_cpu_usage()
        if cpu is not None:
            lines.append(f"  CPU     : %{cpu:.1f} kullanımda")
        else:
            lines.append("  CPU     : psutil kurulu değil")

        # RAM
        mem = self.get_memory_info()
        if mem:
            lines.append(
                f"  RAM     : {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB "
                f"(%{mem['percent']:.0f} kullanımda)"
            )

        # GPU
        gpu = self.get_gpu_info()
        if gpu.get("available"):
            for d in gpu["devices"]:
                lines.append(
                    f"  GPU {d['id']}   : {d['name']} | "
                    f"VRAM {d['allocated_gb']:.1f}/{d['total_vram_gb']:.1f} GB"
                )
        else:
            lines.append(f"  GPU     : {gpu.get('reason', 'Yok')}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<SystemHealthManager gpu={self._gpu_available} "
            f"torch={self._torch_available}>"
        )
