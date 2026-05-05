from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = ""
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    redis_url: str = "redis://localhost:26379"
    auth_admin_username: str = "admin"
    auth_admin_password: str = "admin123"
    auth_researcher_username: str = "researcher"
    auth_researcher_password: str = "researcher123"
    session_secret_key: str = "pharma_ra_dev_secret_change_in_prod"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
