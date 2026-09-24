"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Full-text search over the current session's past conversation content in the LCM database. "
        "Returns both raw messages and summary nodes across all depths. "
        "Use lcm_expand(store_id=...) on a message hit or lcm_expand(node_id=...) on a summary hit "
        "to drill into its full content. Set content_scope='externalized' or 'both' to opt into bounded "
        "search over recoverable payload sidecars."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (FTS5 syntax: keywords, phrases, OR/NOT). "
                    "FTS5 defaults to AND matching, so prefer 1-3 distinctive terms or one quoted multi-word phrase. "
                    "Wrap exact phrases in quotes. Short CJK fragments and emoji-heavy queries may use substring fallback instead of plain FTS token matching."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Max results to return (default 10, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from in the response."
                ),
                "default": 10,
            },
            "sort": {
                "type": "string",
                "enum": ["recency", "relevance", "hybrid"],
                "description": (
                    "How to order matches. 'recency' favors newer hits, 'relevance' favors strongest FTS matches, "
                    "and 'hybrid' keeps strong older matches competitive while still boosting newer context."
                ),
                "default": "recency",
            },
            "content_scope": {
                "type": "string",
                "enum": ["history", "externalized", "both"],
                "description": (
                    "Content stores to search. 'history' (default) preserves current raw-message and summary behavior. "
                    "'externalized' searches only bounded externalized-payload prefixes owned by the active session. "
                    "'both' searches history and those payloads."
                ),
                "default": "history",
            },
            "externalized_refs": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 256,
                "description": (
                    "Optional externalized ref filenames to search. Valid only with content_scope='externalized' or 'both'. "
                    "Every ref must be a regular payload owned by the active session."
                ),
            },
            "role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional raw-message role filter. When supplied, lcm_grep returns raw message hits only.",
            },
            "time_from": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive minimum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
            "time_to": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive maximum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
        },
        "required": ["query"],
    },
}


LCM_EXPAND = {
    "name": "lcm_expand",
    "description": (
        "Recover the original detail behind a summary node, externalized payload, or raw message. "
        "Mode selection (exactly one): node_id (current session only) returns the source messages "
        "or lower-depth summaries that were compacted into a summary node; externalized_ref "
        "(current session only) returns a stored externalized payload's content; store_id returns "
        "a single raw message by store_id, suitable for drilling into lcm_grep message hits. "
        "Output is bounded by max_tokens; raw recovery is pageable via content_offset "
        "(and source_offset/source_limit for node_id mode)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": (
                    "Summary node ID to expand. Current-session only — cross-session DAG expansion "
                    "is not supported in this version."
                ),
            },
            "externalized_ref": {
                "type": "string",
                "description": "Externalized payload ref filename to expand instead of a summary node. Current-session only.",
            },
            "store_id": {
                "type": "integer",
                "description": (
                    "Raw message store_id to fetch, as surfaced by an lcm_grep message hit. Returns the "
                    "message's content paged by content_offset. If the row references an externalized "
                    "payload, the ref is surfaced via 'externalized_ref'."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned content (default 4000)",
                "default": 4000,
            },
            "source_offset": {
                "type": "integer",
                "description": "Zero-based pagination offset into the node's immediate source list (node_id mode only).",
                "default": 0,
            },
            "source_limit": {
                "type": "integer",
                "description": "Maximum number of immediate sources to return from source_offset (node_id mode only). Output still respects max_tokens.",
            },
            "content_offset": {
                "type": "integer",
                "description": "Character offset used to continue an oversized raw message, externalized payload, or store_id-mode message. Use next_content_offset from the previous response.",
                "default": 0,
            },
        },
        "required": [],
    },
}

LCM_STATUS = {
    "name": "lcm_status",
    "description": (
        "Get a quick health overview of the LCM engine for the current session. "
        "Shows compression count, store size, DAG depth distribution, context usage, "
        "active configuration, session/message filter state, and rotate snapshot "
        "state (last_rotate_at, rotate_backup_path, rotate_backup_size when a "
        "/lcm rotate apply has been run). Use this to understand how much history "
        "has been compacted, how the engine is performing, whether the current "
        "session is matched by ignore or stateless session patterns, which message "
        "noise-suppression patterns are loaded, and when the rolling rotate "
        "backup was last written."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_INSPECT = {
    "name": "lcm_inspect",
    "description": (
        "Inspect read-only LCM metadata for the current session: session/conversation "
        "lineage, message frontier and fresh tail, DAG compaction frontier, latest "
        "compaction skip/no-op reason, externalized payload refs and readability, "
        "and matched ignore/stateless patterns. This is an operator inventory tool; "
        "use lcm_grep/lcm_expand when you need actual content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows/items to return for bounded sections. Defaults to 20 and is capped at 200.",
                "default": 20,
            },
        },
        "required": [],
    },
}

LCM_DOCTOR = {
    "name": "lcm_doctor",
    "description": (
        "Run diagnostics on the LCM database and configuration. Checks database "
        "integrity, detects orphaned DAG nodes, validates configuration, and "
        "reports potential issues. Use this to troubleshoot problems or verify "
        "a healthy setup."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_EXPAND_QUERY = {
    "name": "lcm_expand_query",
    "description": (
        "Answer a natural-language question using expanded LCM context from the current session. Provide a prompt, and either "
        "query matching summaries/raw messages to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout. "
        "When expanding parent summary nodes, it recursively descends the DAG under the context budget to include leaf evidence where possible. "
        "Prefer this for questions about the active conversation after compaction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The question or task to answer from expanded LCM context",
            },
            "query": {
                "type": "string",
                "description": "Optional search query used to find candidate summaries before expansion",
            },
            "node_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional explicit summary node IDs to expand instead of searching",
            },
            "max_results": {
                "type": "integer",
                "description": "Max candidate summaries to expand when using query (default 5)",
                "default": 5,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max answer tokens for bounded synthesis returned to the main agent (default 2000)",
                "default": 2000,
            },
            "context_max_tokens": {
                "type": "integer",
                "description": "Expanded serialized summary/raw/child-source/externalized fresh context budget for the auxiliary LLM before it returns the bounded answer (default max(answer max_tokens, 32000 or LCM_EXPANSION_CONTEXT_TOKENS))",
                "default": 32000,
            },
        },
        "required": ["prompt"],
    },
}
