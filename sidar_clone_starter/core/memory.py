from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationMemory:
    last_file: Optional[str] = None
    last_user_goal: Optional[str] = None
