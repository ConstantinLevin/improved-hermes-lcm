"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Find where a term lies in what this session's past holds as stored at its last compaction: what was "
        "said (the messages as sent), each tool call's name and arguments, tool results and readable reasoning; "
        "never the summaries. The term is matched exactly: case-sensitive, character for character. Returns "
        "each chunk the term lies in, whole, as lcm_expand returns it (collapsed: calls without their results; "
        "raw=true puts results inline), each named with the summary that covers it in your context. In the "
        "collapsed form, results_holding_term names the results that hold the term, to open with lcm_expand. "
        "At three or more chunks only their count comes back: narrow the term or the scope, or pass all=true. "
        "Records of the fresh tail stored at the last compaction that hold the term are named by handle. What "
        "came after the last compaction is in your context and not searched. A result longer than one page "
        "carries next_page: call again with page=next_page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "The exact text to find (any length from one character; not a pattern or query syntax).",
            },
            "scope": {
                "type": "string",
                "description": "Optional: a summary's (s…) or chunk's (c…) handle, to search only what stands behind it. "
                               "Without it, the whole session is searched.",
            },
            "all": {
                "type": "boolean",
                "description": "Return every matching chunk even when there are three or more.",
                "default": False,
            },
            "raw": {
                "type": "boolean",
                "description": "Put every tool result inline in the returned chunks.",
                "default": False,
            },
            "page": {
                "type": "string",
                "description": "The next_page token of an earlier result, to read the next page of the same search.",
            },
        },
        "required": [],
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
        "with page=next_page for the rest. Images come back as images where the session's route carries "
        "them; an image not shown stands as a mark saying why, and the store keeps it."
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
        "Answer a natural-language question using expanded LCM context from the current session. Provide a prompt and "
        "the handles (s…) of the summaries to read (lcm_grep finds where a term lies). Uses the expansion path "
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
            "handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The summary handles (s…) to expand",
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
