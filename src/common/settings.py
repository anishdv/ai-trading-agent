"""Application settings from environment."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trading_enabled: bool = False
    auto_execute: bool = False

    start_date: str = "2020-01-01"
    end_date: str | None = None

    # Twilio WhatsApp (Phase 4)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = ""
    approver_whatsapp_to: str = ""

    # Alpaca paper (Phase 4)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # LLM (Phase 3)
    openai_api_key: str = ""

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def proposals_dir(self) -> Path:
        return self.data_dir / "proposals"


settings = Settings()
