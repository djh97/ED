from __future__ import annotations

import os


class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    openai_input_model: str = os.getenv("OPENAI_INPUT_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.2"))
    openai_orchestration_model: str = os.getenv("OPENAI_ORCHESTRATION_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.2"))
    openai_summary_model: str = os.getenv("OPENAI_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.2"))
    app_name: str = "Agentic ED Operations Prototype"


settings = Settings()
