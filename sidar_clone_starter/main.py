from pathlib import Path

from sidar_clone_starter.agents.engineer import EngineerAgent
from sidar_clone_starter.config import Settings, AccessLevel


def build_agent() -> EngineerAgent:
    settings = Settings(
        access_level=AccessLevel.SANDBOX,
        sandbox_dir=Path("./sidar_clone_starter/sandbox"),
    )
    settings.sandbox_dir.mkdir(parents=True, exist_ok=True)
    return EngineerAgent(settings)


if __name__ == "__main__":
    agent = build_agent()
    print("EngineerAgent hazır. Komut örneği: audit")
    print(agent.handle("audit"))
