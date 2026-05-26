"""
app/config.py
=============
Configurazione del backend, letta da variabili d'ambiente e/o file `.env`.

------------------------------------------------------------------------
Concetti chiave
------------------------------------------------------------------------

`pydantic-settings`
    Estensione ufficiale di Pydantic che popola un BaseModel a partire
    da:
      1. Variabili d'ambiente del processo (ad es. `export FRONTEND_ORIGIN=...`).
      2. File `.env` (di default cercato nella cwd da cui si lancia
         il server).
      3. Default dichiarati nella classe.

    Le priorità sono in questo ordine: env vars > .env > defaults.

    In termini Node: è "Zod + dotenv" in un solo pacchetto. La
    differenza chiave è che qui i tipi sono **validati a runtime**:
    se metti `API_PORT=banana` nel `.env`, Pydantic alza un errore
    leggibile invece di lasciarti scoprire il bug più tardi.

Perché un singleton `settings`?
    Vogliamo che la config sia letta UNA volta (all'avvio) e poi
    importabile ovunque. Il pattern Python idiomatico è creare
    un'istanza module-level (`settings = Settings()`) e farla
    importare da chi serve: `from app.config import settings`.

    In FastAPI è anche possibile usare la dependency injection
    (`Depends(get_settings)`), che ha senso quando vuoi mockare la
    config nei test. Per ora il singleton va benissimo: lo useremo in
    poche righe e i test possono comunque sovrascrivere via env var.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ----------------------------------------------------------------------------
# Percorso assoluto della root del repo, calcolato dalla posizione di QUESTO file.
# ----------------------------------------------------------------------------
# `__file__` qui è ".../apps/api/app/config.py".
# Risalendo i parent:
#   parents[0] → app/
#   parents[1] → api/
#   parents[2] → apps/
#   parents[3] → root del repo (agentic-rag-stack/)
#
# Calcolarlo così significa: indipendentemente da DOVE viene lanciato uvicorn,
# pydantic-settings carica sempre lo stesso `.env` (quello al repo root).
# In una monorepo questo evita la confusione di "ho dimenticato di stare in apps/api/".
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Configurazione runtime del backend.

    Ogni attributo qui sotto:
      - Ha un type hint (Pydantic lo userà per validare).
      - Ha un default (Pydantic lo userà se la env var non è presente).
      - Verrà popolato leggendo l'env var maiuscola corrispondente
        (es. `app_env` → `APP_ENV`). Comportamento di pydantic-settings.
    """

    # Ambiente di esecuzione. In dev attiviamo log verbose, reload, ecc.
    # In prod metteremo "production" via env var.
    app_env: str = Field(default="development")

    # Porta su cui il server ASGI ascolta. Usata principalmente dal nostro
    # script di avvio (uvicorn legge il flag --port separatamente).
    api_port: int = Field(default=8000)

    # Origine del frontend usata dal middleware CORS per consentire fetch
    # dal browser. In dev è la URL di Next.js (porta 3000 default).
    # In prod sarà l'URL pubblica del frontend.
    frontend_origin: str = Field(default="http://localhost:3000")

    # Livello minimo di log emesso. DEBUG mostra tutto (incluso il chiacchiericcio
    # interno di httpx, anthropic, ecc.), INFO è il default sano per dev.
    # In prod metteremo "WARNING" per ridurre rumore (M5).
    log_level: str = Field(default="INFO")

    # Formato dei log: "dev" → human-readable colorato; "json" → strutturato
    # parsabile da aggregatori esterni (Datadog, CloudWatch, ecc).
    # Default "dev" perché la maggior parte delle sessioni è locale.
    log_format: str = Field(default="dev")

    # ------------------------------------------------------------------------
    # LLM provider keys (M1+)
    # ------------------------------------------------------------------------
    # Chiave API Anthropic, usata dal servizio classificatore (vedi
    # app/services/classifier.py) e in futuro dagli agenti LangGraph (M4).
    # Default "" perché il backend deve poter girare anche solo per /health
    # senza la chiave configurata. Gli endpoint che la richiedono
    # controllano esplicitamente e rispondono 503 se manca.
    anthropic_api_key: str = Field(default="")

    # Chiave API OpenAI, usata per text-embedding-3-small (vedi ADR-0004)
    # e potenzialmente per altri modelli OpenAI in milestone successive.
    # Stesso pattern di anthropic_api_key: default "" così il backend
    # parte anche senza, gli endpoint che la richiedono rispondono 503.
    openai_api_key: str = Field(default="")

    # SettingsConfigDict configura il comportamento del loader:
    #   env_file: path ASSOLUTO del .env da caricare (vedi _REPO_ROOT sopra).
    #   env_file_encoding: come leggerlo (utf-8 sempre sicuro).
    #   case_sensitive=False: APP_ENV / app_env sono trattate uguali.
    #   extra="ignore": altre variabili nel .env (es. PINECONE_API_KEY per
    #                   milestone successive) vengono ignorate senza errore.
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton: istanziamo Settings() UNA volta a import-time. Da qui in poi
# `from app.config import settings` restituisce sempre la stessa istanza.
settings = Settings()
