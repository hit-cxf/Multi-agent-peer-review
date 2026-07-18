"""Model-aware configuration for OpenAI-compatible LLM endpoints."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def load_env_file(path, override=False):
    """Load KEY=VALUE lines without executing the file as shell code."""
    path = Path(path)
    if not path.exists():
        return
    variable_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env line {line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid .env variable on line {line_number}: {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        value = variable_pattern.sub(lambda match: os.getenv(match.group(1), ""), value)
        if override or key not in os.environ:
            os.environ[key] = value


load_env_file(PROJECT_DIR / ".env", override=False)


def model_env_prefix(model_name):
    """Map e.g. llama3-8b to the environment prefix LLM_LLAMA3_8B."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
    if not normalized:
        raise ValueError("model name cannot be empty")
    return f"LLM_{normalized}"


def _optional_bool(value, name):
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class LLMSettings:
    name: str
    request_model: str
    base_url: str
    api_key: str
    organization: str = ""
    enable_thinking: bool = None


class LLMAdapter:
    """Resolve one endpoint configuration per logical model name."""

    def __init__(self, fallback=None):
        self.fallback = fallback or {}

    def resolve(self, model_name):
        prefix = model_env_prefix(model_name)

        def setting(suffix, fallback_key=None, default=None):
            value = os.getenv(f"{prefix}_{suffix}")
            if value is not None:
                return value
            if fallback_key is not None:
                fallback_value = self.fallback.get(fallback_key)
                if fallback_value is not None and fallback_value != "":
                    return fallback_value
            return default

        base_url = setting("BASE_URL", "base_url", "https://api.openai.com/v1")
        api_key = setting("API_KEY", "api_key", "")
        request_model = setting("MODEL", default=model_name)
        organization = setting("ORGANIZATION", "organization", "")
        thinking_text = setting("ENABLE_THINKING", "enable_thinking")

        return LLMSettings(
            name=model_name,
            request_model=request_model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            organization=organization,
            enable_thinking=_optional_bool(
                thinking_text, f"{prefix}_ENABLE_THINKING"
            ),
        )


def default_model():
    return os.getenv(
        "LLM_DEFAULT_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0613")
    )
