import os

from typing import Any
from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn, validator


class Settings(BaseSettings):
    VERSION: str = Field("0.0.1")
    PROJECT_NAME: str = Field("Workshop 1")
    POSTGRES_USER: str = Field("postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("postgres", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("postgres", env="POSTGRES_DB")
    POSTGRES_HOST: str = Field("postgres", env="POSTGRES_HOST")
    POSTGRES_PORT: int | str = Field("5432", env="POSTGRES_PORT")
    POSTGRES_ECHO: bool = Field(False, env="POSTGRES_ECHO")
    POSTGRES_POOL_SIZE: int = Field(10, env="POSTGRES_POOL_SIZE")
    POSTGRES_URI: PostgresDsn | None = Field(None, env="POSTGRES_URI")
    RAW_TABLE: str = Field("raw_table", env="RAW_TABLE")
    CLEAN_TABLE: str = Field("clean_table", env="CLEAN_TABLE")

    class Config:
        case_sensitive = True
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env.local')

    @validator("POSTGRES_URI", pre=True)
    def assemble_db_connection(cls, v: str | None, values: dict[str, Any]) -> Any:
        """
        Generate a PostgreSQL connection string.
            :param v: value
            :param values: values
            :return: connection string
        """
        if isinstance(v, str):
            return v

        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=int(values.get("POSTGRES_PORT")),
            path=f"{values.get('POSTGRES_DB') or ''}",
        )


settings = Settings()
