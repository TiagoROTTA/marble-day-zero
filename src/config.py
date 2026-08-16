from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    # Sonnet 5 rather than Opus 5: this pipeline is extraction, structured
    # decomposition and catalog matching, which Sonnet 5 handles at near-Opus
    # quality, and it is 2.5x cheaper per token ($3/$15 per MTok list against
    # Opus 5's $5/$25 -- and $2/$10 under the introductory rate running to
    # 2026-08-31). Raise `llm_model` back to claude-opus-5 if extraction
    # quality on hard menus regresses; `llm_model_cheap` should stay here.
    llm_model: str = "claude-sonnet-5"
    llm_model_cheap: str = "claude-sonnet-5"
    # 32k, not 16k: a 90-item menu serialised into the extraction schema does not
    # fit in 16k output tokens, and the run dies with "Output parser received a
    # `max_tokens` stop reason" (fonda-park-slope, junoon). Sonnet 5 allows up to
    # 128k output, but non-streaming requests much above this risk SDK HTTP
    # timeouts -- the real fix for a genuinely huge menu is to batch the
    # extraction the way decompose_recipes already batches recipes.
    llm_max_tokens: int = 32000

    max_retries: int = 3

    checkpoint_db_path: str = "./data/checkpoints.sqlite"

    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_approval_channel: str = ""

    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
