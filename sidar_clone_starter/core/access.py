from pathlib import Path
from sidar_clone_starter.config import AccessLevel


def can_write(access_level: AccessLevel, target: Path, sandbox_dir: Path) -> bool:
    if access_level == AccessLevel.RESTRICTED:
        return False
    if access_level == AccessLevel.FULL:
        return True

    target_resolved = target.resolve()
    sandbox_resolved = sandbox_dir.resolve()
    return str(target_resolved).startswith(str(sandbox_resolved))
