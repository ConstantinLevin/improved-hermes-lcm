"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Full-text search over the current session's past conversation content in the LCM database, "
        "which stores it at each compaction. "
        "Returns both raw messages and summaries, each with its handle. "
        "Use lcm_expand(handle=...) on a hit to read behind it."
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
            "role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional raw-message role filter. When supplied, lcm_grep returns raw message hits only.",
            },
            "time_from": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive minimum time a message was stored (the time of the compaction that stored it, "
                    "not the time it was said). Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
            "time_to": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive maximum time a message was stored (the time of the compaction that stored it, "
                    "not the time it was said). Accepts Unix seconds or timezone-aware ISO 8601; "
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
        "Look behind a handle and read what it stands for, as it was. A summary's handle (s…) or a chunk's "
        "(c…) returns that stretch of the session: every user and assistant message verbatim, the readable "
        "reasoning beside it, and each tool call with its handle (t…), name and arguments but without its "
        "result; raw=true puts every result inline. A tool call's handle returns its result; a message's "
        "handle (m…) returns that message. A result longer than one page carries next_page: call again "
        "with page=next_page for the rest. Images come back as images."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "The handle to look behind: s… a summary, c… a chunk, t… a tool call, m… a message.",
            },
            "raw": {
                "type": "boolean",
                "description": "For a summary or chunk: put every tool result inline instead of leaving it behind its call's handle.",
                "default": False,
            },
            "page": {
                "type": "string",
                "description": "The next_page token of an earlier result, to read the next page of the same expansion.",
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
        "and active configuration. Use this to understand how much history "
        "has been compacted and how the engine is performing."
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
        "compaction skip/no-op reason. "
        "This is an operator inventory tool; "
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
        "query matching summaries/raw messages to expand or the handles (s…) of summaries to inspect. Uses the expansion path "
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
            "handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional summary handles (s…) to expand instead of searching",
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
                "description": "Expanded serialized summary/raw/child-source fresh context budget for the auxiliary LLM before it returns the bounded answer (default max(answer max_tokens, 32000 or LCM_EXPANSION_CONTEXT_TOKENS))",
                "default": 32000,
            },
        },
        "required": ["prompt"],
    },
}
