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


# The plugin reads no key of the host's config.yaml ``lcm:`` section: its values come
# from its own configuration (#22). Every key found there is listed as ignored.
_SUPPORTED_LCM_CONFIG_YAML_KEYS: frozenset = frozenset()


def _host_native_compaction_configured(cfg: dict[str, Any] | None = None) -> bool:
    """Whether the host's config.yaml opts into native server-side compaction on its
    OpenAI Responses routes (``compression.codex_responses_native``, read by the host's
    agent_init with its own truthy rule, utils.is_truthy_value). Read to be refused, not
    to be followed (#11, #24, #32 D2)."""
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    compression = cfg.get("compression") if isinstance(cfg, dict) else None
    if not isinstance(compression, dict):
        return False
    value = compression.get("codex_responses_native", False)
    try:
        from utils import is_truthy_value  # type: ignore  # the host's own rule
        return bool(is_truthy_value(value))
    except Exception:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


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


@dataclass(frozen=True)
class _EnvFieldSpec:
    """One scalar ``LCM_*`` environment override: which config field it sets,
    its environment variable, and the Python type used to parse it."""

    name: str
    env_key: str
    py_type: type


# Single source of truth for the scalar LCM_* env overrides. ``from_env`` applies
# the non-source-tracked entries uniformly.
ENV_FIELD_SPECS: tuple[_EnvFieldSpec, ...] = (
    _EnvFieldSpec("fresh_tail_count", "LCM_FRESH_TAIL_COUNT", int),
    _EnvFieldSpec("fresh_tail_max_tokens", "LCM_FRESH_TAIL_MAX_TOKENS", int),
    _EnvFieldSpec("chunk_tokens", "LCM_CHUNK_TOKENS", int),
    _EnvFieldSpec("estimate_ratio", "LCM_ESTIMATE_RATIO", float),
    _EnvFieldSpec("round_growth_tokens", "LCM_ROUND_GROWTH_TOKENS", int),
    _EnvFieldSpec("hygiene_share", "LCM_HYGIENE_SHARE", float),
    _EnvFieldSpec("turn_margin_tokens", "LCM_TURN_MARGIN_TOKENS", int),
    _EnvFieldSpec("target_share", "LCM_TARGET_SHARE", float),
    _EnvFieldSpec("window_min_tokens", "LCM_WINDOW_MIN_TOKENS", int),
    _EnvFieldSpec("window_max_tokens", "LCM_WINDOW_MAX_TOKENS", int),
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
    "chunk_tokens",
    "estimate_ratio",
    "round_growth_tokens",
    "hygiene_share",
    "turn_margin_tokens",
    "target_share",
    "window_min_tokens",
    "window_max_tokens",
})

# The geometry's weights (#31, Decided; R14): field, env var, parse type, default, and
# the source recorded while the default stands.
_GEOMETRY_WEIGHTS = (
    ("round_growth_tokens", "LCM_ROUND_GROWTH_TOKENS", int, 123_000,
     "default: #31 Decided, the most one tool round adds (50k by the estimate x 1.95, plus 25k output)"),
    ("hygiene_share", "LCM_HYGIENE_SHARE", float, 0.85,
     "default: #31 Decided, the gateway's hygiene path at 0.85 of the window"),
    ("turn_margin_tokens", "LCM_TURN_MARGIN_TOKENS", int, 54_500,
     "default: #31 Decided, the count-p95 of the growth inside a human-begun turn"),
    ("target_share", "LCM_TARGET_SHARE", float, 0.3,
     "default: #31 Decided, G = min(0.3 x W, tau)"),
    ("window_min_tokens", "LCM_WINDOW_MIN_TOKENS", int, 256_000,
     "default: #21's lower bound on the window (R11)"),
    ("window_max_tokens", "LCM_WINDOW_MAX_TOKENS", int, 2_000_000,
     "default: #21's upper bound on the window (R11)"),
)


@dataclass
class LCMConfig:
    """All tunables for the LCM engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 32
    # Optional token cap for the protected suffix (0 = disabled)
    fresh_tail_max_tokens: int = 0

    # -- The chunk (#31, Decided; #12) ---
    # c in provider tokens: bound by the summariser only, not scaled with the window.
    chunk_tokens: int = 50_000
    # The provider's count over the plugin's estimate (characters / 4), by which a size
    # in provider tokens is cut in the estimate's unit: #31's measurement on Claude tool
    # results, p50 (p95 1.95, p99 2.37, n = 201). c in the estimate: 50,000 / 1.51.
    estimate_ratio: float = 1.51

    # -- The trigger's geometry (#31, Decided; #11, #32; ``geometry``) ---
    # tau' = min(W - round_growth_tokens, hygiene_share x W), tau = tau' - turn_margin_tokens,
    # G = min(target_share x W, tau); all provider tokens. Defined for a window within
    # [window_min_tokens, window_max_tokens] only (R11). Nothing of the host's own
    # threshold setting is read.
    round_growth_tokens: int = 123_000
    hygiene_share: float = 0.85
    turn_margin_tokens: int = 54_500
    target_share: float = 0.3
    window_min_tokens: int = 256_000
    window_max_tokens: int = 2_000_000

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
    # The host's own opt-in to native server-side compaction, as its config.yaml sets
    # it: read only to be refused visibly (the engine cannot switch it off, #32 D2).
    host_native_compaction: bool = False

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
        c.host_native_compaction = _host_native_compaction_configured()
        if c.host_native_compaction:
            _record("host_native_compaction", "config_yaml:compression.codex_responses_native")

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
        c.chunk_tokens, source, warning = _parse_int_env_with_source(
            "LCM_CHUNK_TOKENS", c.chunk_tokens, default_source="default: #31 Decided, c = 50k provider tokens"
        )
        if c.chunk_tokens < 1:
            warning, c.chunk_tokens, source = (f"LCM_CHUNK_TOKENS={c.chunk_tokens} is not positive; the default "
                                               f"applies", 50_000, "default: #31 Decided, c = 50k provider tokens")
        _record("chunk_tokens", source, warning)
        c.estimate_ratio, source, warning = _parse_float_env_with_source(
            "LCM_ESTIMATE_RATIO", c.estimate_ratio,
            default_source="default: #31 measured, p50 of the provider's count over characters / 4 on Claude",
        )
        if not c.estimate_ratio > 0:
            warning, c.estimate_ratio, source = (
                f"LCM_ESTIMATE_RATIO={c.estimate_ratio} is not positive; the default applies", 1.51,
                "default: #31 measured, p50 of the provider's count over characters / 4 on Claude")
        _record("estimate_ratio", source, warning)
        for name, env_key, py_type, default, default_source in _GEOMETRY_WEIGHTS:
            parse = _parse_int_env_with_source if py_type is int else _parse_float_env_with_source
            value, source, warning = parse(env_key, default, default_source=default_source)
            if not value > 0:
                warning = f"{env_key}={value} is not positive; the default applies"
                value, source = default, default_source
            setattr(c, name, value)
            _record(name, source, warning)
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
