"""LCM configuration with defaults and env var overrides."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback for minimal installs
    yaml = None


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_str_env(key: str, default):
    return os.environ.get(key, default)


def _parse_int_env_with_source(
    key: str,
    default: int,
    *,
    default_source: str = "default",
) -> tuple[int, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return int(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _parse_float_env_with_source(
    key: str,
    default: float,
    *,
    default_source: str = "default",
) -> tuple[float, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return float(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _config_bool_disabled(value) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return True
        try:
            return float(normalized) == 0
        except ValueError:
            return False
    return False


def _hermes_config_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home / "config.yaml"


def _load_hermes_config_yaml() -> dict[str, Any]:
    cfg_path = _hermes_config_path()
    try:
        text = cfg_path.read_text()
    except Exception:
        return {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text) or {}
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        value = value.strip("'\"")
        lowered = value.lower()
        if lowered in {"true", "yes", "on"}:
            parsed: Any = True
        elif lowered in {"false", "no", "off"}:
            parsed = False
        else:
            try:
                parsed = float(value) if "." in value else int(value)
            except ValueError:
                parsed = value
        parent[key] = parsed
    return root


_SUPPORTED_LCM_CONFIG_YAML_KEYS = {"context_threshold"}


def _ignored_lcm_config_yaml_keys(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    lcm_section = cfg.get("lcm") if isinstance(cfg, dict) else None
    if not isinstance(lcm_section, dict):
        return []
    return sorted(
        str(key)
        for key in lcm_section
        if str(key) not in _SUPPORTED_LCM_CONFIG_YAML_KEYS
    )


def _hermes_compression_threshold_with_source(default: float) -> tuple[float, str]:
    cfg = _load_hermes_config_yaml()
    try:
        lcm_section = cfg.get("lcm") or {}
        if isinstance(lcm_section, dict):
            lcm_val = lcm_section.get("context_threshold")
            if lcm_val is not None:
                return float(lcm_val), "config_yaml:lcm.context_threshold"
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        if _config_bool_disabled(compression.get("enabled")):
            return default, "default"
        comp_val = compression.get("threshold")
        if comp_val is not None:
            return float(comp_val), "config_yaml:compression.threshold"
    except Exception:
        return default, "default"
    return default, "default"


def _parse_calls_per_endpoint(raw: str) -> tuple[dict[str, int], str | None]:
    """``LCM_SUMMARY_CALLS_PER_ENDPOINT``: a JSON object from an endpoint (the base URL a
    summariser route names, or ``provider:<name>`` where it names none) to the most
    summariser calls in flight there. Returns the mapping and a warning, if any."""
    if not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except ValueError:
        return {}, "LCM_SUMMARY_CALLS_PER_ENDPOINT is not JSON; the default applies to every endpoint"
    if not isinstance(value, dict):
        return {}, "LCM_SUMMARY_CALLS_PER_ENDPOINT is not a JSON object; the default applies to every endpoint"
    parsed: dict[str, int] = {}
    bad: list[str] = []
    for key, limit in value.items():
        if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 1:
            parsed[str(key).strip().rstrip("/")] = limit
        else:
            bad.append(str(key))
    warning = (f"LCM_SUMMARY_CALLS_PER_ENDPOINT: no positive integer for {', '.join(bad)}; the default applies there"
               if bad else None)
    return parsed, warning


def _hermes_codex_gpt55_autoraise_with_source(default: bool) -> tuple[bool, str]:
    cfg = _load_hermes_config_yaml()
    try:
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        value = compression.get("codex_gpt55_autoraise")
        if value is None:
            return default, "default"
        return (not _config_bool_disabled(value)), "config_yaml:compression.codex_gpt55_autoraise"
    except Exception:
        return default, "default"


@dataclass(frozen=True)
class _EnvFieldSpec:
    """One scalar ``LCM_*`` environment override: which config field it sets,
    its environment variable, and the Python type used to parse it."""

    name: str
    env_key: str
    py_type: type


# Single source of truth for the scalar LCM_* env overrides. ``from_env`` applies
# the non-source-tracked entries uniformly, and ``presets`` derives its
# preset-field lookups from the same list so the field/env/type mapping is not
# duplicated. Order mirrors the historical ``from_env`` order for readability.
ENV_FIELD_SPECS: tuple[_EnvFieldSpec, ...] = (
    _EnvFieldSpec("fresh_tail_count", "LCM_FRESH_TAIL_COUNT", int),
    _EnvFieldSpec("fresh_tail_max_tokens", "LCM_FRESH_TAIL_MAX_TOKENS", int),
    _EnvFieldSpec("leaf_chunk_tokens", "LCM_LEAF_CHUNK_TOKENS", int),
    _EnvFieldSpec("context_threshold", "LCM_CONTEXT_THRESHOLD", float),
    _EnvFieldSpec("max_assembly_tokens", "LCM_MAX_ASSEMBLY_TOKENS", int),
    _EnvFieldSpec("reserve_tokens_floor", "LCM_RESERVE_TOKENS_FLOOR", int),
    _EnvFieldSpec("custom_instructions", "LCM_CUSTOM_INSTRUCTIONS", str),
    _EnvFieldSpec("summary_model", "LCM_SUMMARY_MODEL", str),
    _EnvFieldSpec("summary_provider", "LCM_SUMMARY_PROVIDER", str),
    _EnvFieldSpec("summary_base_url", "LCM_SUMMARY_BASE_URL", str),
    _EnvFieldSpec("summary_api_key", "LCM_SUMMARY_API_KEY", str),
    _EnvFieldSpec("summary_api_mode", "LCM_SUMMARY_API_MODE", str),
    _EnvFieldSpec("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT", str),
    _EnvFieldSpec("expansion_model", "LCM_EXPANSION_MODEL", str),
    _EnvFieldSpec("expansion_context_tokens", "LCM_EXPANSION_CONTEXT_TOKENS", int),
    _EnvFieldSpec("summary_calls_in_flight", "LCM_SUMMARY_CALLS_IN_FLIGHT", int),
    _EnvFieldSpec("expansion_timeout_ms", "LCM_EXPANSION_TIMEOUT_MS", int),
    _EnvFieldSpec("database_path", "LCM_DATABASE_PATH", str),
)

_PARSER_BY_TYPE = {
    int: _parse_int_env,
    float: _parse_float_env,
    bool: _parse_bool_env,
    str: _parse_str_env,
}

# Fields whose env reading needs provenance tracking or a computed default;
# ``from_env`` handles these explicitly, so the uniform loop skips them.
_SOURCE_TRACKED_ENV_FIELDS = frozenset({
    "fresh_tail_count",
    "fresh_tail_max_tokens",
    "leaf_chunk_tokens",
    "context_threshold",
})


@dataclass
class LCMConfig:
    """All tunables for the LCM engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 32
    # Optional token cap for the protected suffix (0 = disabled)
    fresh_tail_max_tokens: int = 0

    # -- Compaction thresholds ---
    # Max source tokens in a leaf chunk before summarization triggers
    leaf_chunk_tokens: int = 20_000
    # Fraction of context window that triggers compaction (0.0–1.0)
    context_threshold: float = 0.35
    # Mirror Hermes Agent's Codex gpt-5.5 route-specific threshold auto-raise
    # when LCM is inheriting the host compression threshold. Explicit LCM
    # threshold overrides remain authoritative.
    codex_gpt55_autoraise_enabled: bool = True

    # -- Assembly guardrails ---
    # Hard cap for the assembled active context (0 = disabled)
    max_assembly_tokens: int = 0
    # Reserve this many tokens from the model context window before assembly
    # (0 = disabled). Effective cap becomes context_length - reserve_tokens_floor.
    reserve_tokens_floor: int = 0

    # -- Summary instructions ---
    # Custom instructions injected into all summarization prompts
    custom_instructions: str = ""

    # -- Models ---
    # The summariser (#9). Empty summary_model: the model running the session, on the
    # route the host hands update_model. Set: that model, on the provider named in
    # summary_provider (required with it), with the optional base URL, key and API mode;
    # a model id only, never "provider/model".
    summary_model: str = ""
    summary_provider: str = ""
    summary_base_url: str = ""
    summary_api_key: str = ""     # never shown in status
    summary_api_mode: str = ""
    # The summariser's reasoning effort, the host's levels; a session's own value, set
    # by the owner's command, is a session fact and wins (#9, #26).
    summary_reasoning_effort: str = "medium"
    expansion_model: str = ""     # empty = fall back to summary_model / Hermes auxiliary model
    # Serialized summary/raw/child-source context budget fed to lcm_expand_query's auxiliary LLM before it returns a bounded answer.
    expansion_context_tokens: int = 32_000

    # -- Summariser calls in flight (#33) ---
    # The most summariser calls at once to one endpoint, process-wide; per endpoint
    # where summary_calls_per_endpoint names it (the base URL a summariser route names,
    # or "provider:<name>" where it names none). There is no per-call timeout of the
    # plugin's own: each call is bounded by the host's deadline at its dispatch.
    summary_calls_in_flight: int = 8
    summary_calls_per_endpoint: dict[str, int] = field(default_factory=dict)

    # -- Timeouts ---
    expansion_timeout_ms: int = 120_000

    # -- Storage ---
    database_path: str = ""       # empty = <host-given Hermes home>/lcm-record.db; LCM_DATABASE_PATH may override

    # -- Diagnostics ---
    # Field-level provenance for values loaded through from_env(). Manual
    # LCMConfig(...) instances leave this empty and status treats them as manual/default.
    config_sources: dict[str, str] = field(default_factory=dict)
    config_source_warnings: list[str] = field(default_factory=list)
    ignored_config_yaml_lcm_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Build config from environment variables (LCM_ prefix)."""
        c = cls()
        config_sources: dict[str, str] = {}
        config_source_warnings: list[str] = []

        def _record(field: str, source: str, warning: str | None = None) -> None:
            config_sources[field] = source
            if warning:
                config_source_warnings.append(warning)

        c.ignored_config_yaml_lcm_keys = _ignored_lcm_config_yaml_keys()

        # Source-tracked fields (provenance recording and/or a computed default)
        # stay explicit; the uniform loop below skips them.
        c.fresh_tail_count, source, warning = _parse_int_env_with_source(
            "LCM_FRESH_TAIL_COUNT", c.fresh_tail_count
        )
        _record("fresh_tail_count", source, warning)
        c.fresh_tail_max_tokens, source, warning = _parse_int_env_with_source(
            "LCM_FRESH_TAIL_MAX_TOKENS", c.fresh_tail_max_tokens
        )
        c.fresh_tail_max_tokens = max(0, c.fresh_tail_max_tokens)
        _record("fresh_tail_max_tokens", source, warning)
        c.leaf_chunk_tokens, source, warning = _parse_int_env_with_source(
            "LCM_LEAF_CHUNK_TOKENS", c.leaf_chunk_tokens
        )
        _record("leaf_chunk_tokens", source, warning)
        context_default, context_source = _hermes_compression_threshold_with_source(c.context_threshold)
        c.context_threshold, source, warning = _parse_float_env_with_source(
            "LCM_CONTEXT_THRESHOLD",
            context_default,
            default_source=context_source,
        )
        _record("context_threshold", source, warning)
        c.codex_gpt55_autoraise_enabled, source = _hermes_codex_gpt55_autoraise_with_source(
            c.codex_gpt55_autoraise_enabled
        )
        _record("codex_gpt55_autoraise_enabled", source)
        c.summary_calls_per_endpoint, warning = _parse_calls_per_endpoint(
            os.environ.get("LCM_SUMMARY_CALLS_PER_ENDPOINT", "")
        )
        _record("summary_calls_per_endpoint",
                "env:LCM_SUMMARY_CALLS_PER_ENDPOINT" if os.environ.get("LCM_SUMMARY_CALLS_PER_ENDPOINT") else "default",
                warning)

        # Every other scalar LCM_* override is applied uniformly from the spec.
        for spec in ENV_FIELD_SPECS:
            if spec.name in _SOURCE_TRACKED_ENV_FIELDS:
                continue
            parser = _PARSER_BY_TYPE[spec.py_type]
            setattr(c, spec.name, parser(spec.env_key, getattr(c, spec.name)))

        c.config_sources = config_sources
        c.config_source_warnings = config_source_warnings
        return c
