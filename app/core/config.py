from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Banco
    database_url: str

    # LLM
    openrouter_api_key:        str
    openrouter_model_primary:  str = "google/gemini-2.5-flash"
    openrouter_model_fallback: str = "google/gemini-flash-1.5"

    # OpenAI (embeddings)
    openai_api_key: str = ""

    # GT Inova (agendamento)
    gt_inova_base_url: str = ""
    gt_inova_api_key:  str = ""

    # Segurança
    webhook_secret: str = ""

    # App
    log_level: str = "INFO"


settings = Settings()
