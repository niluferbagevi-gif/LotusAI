from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AccessLevel(str, Enum):
    RESTRICTED = "restricted"
    SANDBOX = "sandbox"
    FULL = "full"


@dataclass
class Settings:
    access_level: AccessLevel = AccessLevel.SANDBOX
    sandbox_dir: Path = Path("./sandbox")


DEFAULT_SETTINGS = Settings()
