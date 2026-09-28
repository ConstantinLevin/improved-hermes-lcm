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

LCM_QUERY = {
    "name": "lcm_query",
    "description": (
        "Ask a question about what stands behind summaries or chunks in your context, without loading it: "
        "give their handles (s… or c…) and the question. Everything behind every handle is given, in one call, to "
        "the model that writes your summaries: each stored message as its role, content, tool calls and readable "
        "reasoning; any stored value that cannot be given in that shape is given as labelled JSON in its place, and "
        "a message stored with a role the host's converters do not carry as it is is given as a user message whose "
        "label names that role; an image the model is not given is replaced by a placeholder that says so (an image "
        "inside a value given as JSON is given as an image where it stood in the stored content, else named and "
        "counted); signed or encrypted material is withheld and counted; the host's bookkeeping keys and a replay "
        "carrier's metadata are not given; null, empty and blank values carry nothing and are not given as parts. "
        "Where that cannot be shown, the query is refused with the cause, and nothing is ever cut: the input is "
        "larger than the model's window where the plugin knows the window, or holds more images than the host's "
        "converter keeps in one request; a record cannot be given as it is (a replay carrier holds text or a call "
        "its content and calls do not; a record is stored as a JSON value that is not a message; a chunk begins "
        "with a tool result); the host's converter would not deliver everything the query gives the model, on any "
        "wire the host can send the call on; the host can send the call on a wire the plugin has not established, "
        "or retry after an authentication error on a provider's client whose wire the plugin has not established; "
        "the host could answer on a leg whose wire cannot be known or checked before the call: a fallback provider "
        "configured under this route's own provider and model, a credential pool the host can rotate while "
        "fallback providers are configured, a managed NeMo Relay, or a credential refresh that retries on a "
        "Responses client while the route itself is not one (a custom route to chatgpt.com or api.x.ai, or to "
        "GitHub Copilot at api.githubcopilot.com or a subdomain of it with a model the host sends over Responses, "
        "on another wire); or a page cannot hold the result's header. It returns a report and "
        "excerpts. The report is a model's description, hedged: orientation, never something to act on; the model "
        "is asked to name the handles it draws on, which lcm_expand opens. Each excerpt was found verbatim in the "
        "record named by \"in\", in the field named by \"from\": one from a message's content or a tool result may "
        "be relied on as an expansion may; one from a tool call's name or arguments (cited by the handle of the "
        "message that made the call) is what was called; one found "
        "only in reasoning is the model's account of its thinking, and nothing rests on it. An excerpt not found "
        "is withheld and named. A result longer than one page carries next_page: call again with page=next_page "
        "alone. The query calls its model once and tries nothing again: a failed call is first handled by the "
        "host's own recovery, and the error names what that recovery can have changed on the route in use; call "
        "again if you want to. If the host asks this call to stop (its tool timeout, or you are interrupted), the "
        "query stops reading and stores nothing, and its answer is lost; on some routes the model's request still "
        "runs on: every result's header says under call what happens on the route in use."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The handles of the summaries (s…) or chunks (c…) whose stretches the question is about.",
            },
            "question": {
                "type": "string",
                "description": "The question.",
            },
            "page": {
                "type": "string",
                "description": "The next_page token of an earlier result, given alone, to read its next page.",
            },
        },
        "required": [],
    },
}
