import os
import platform
from typing import Dict, Any


class SystemHealthManager:
    def collect(self) -> Dict[str, Any]:
        report = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }

        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            report["ram_total_mb"] = round(vm.total / 1024 / 1024, 2)
            report["ram_used_mb"] = round(vm.used / 1024 / 1024, 2)
        except Exception:
            report["ram_total_mb"] = None
            report["ram_used_mb"] = None
            report["note"] = "Install psutil for detailed memory metrics."

        return report
