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
        "Ask a question about what stands behind summaries or chunks in your context, without loading it: give "
        "their handles (s… or c…) and the question; every handle must resolve, or the query reads nothing. It "
        "reads the chunks behind them in the latest compaction of this session that took effect, in the order of "
        "your context, each chunk once however many handles reach it, never a summary's text. Everything behind "
        "every handle is given, in one call, to the model that writes your summaries, with the query's "
        "instructions, a label on each message naming its handles, and notes of the query's own, each saying what "
        "it is: each stored message as its role, content, tool calls and readable reasoning (where the host sends a "
        "message's api_content text in place of its stored content, that text, and the message's label says so); "
        "any other stored value is given as labelled JSON (a content member in its place, any other value after "
        "the message's content); a message stored with a role other than user, assistant or tool, or with none, is "
        "given as a user message whose label says so; a tool call is given as a call only where the records given "
        "hold its result and the host's Anthropic converter would keep it, else as labelled JSON, and a tool result "
        "no call given as a call answers is given as a user message whose label says so (the header's "
        "calls_not_given_as_calls counts both); a text or tool call only a replay carrier holds is given as a "
        "labelled part; an image is given as stored only where the host's own converter delivers it as an image on "
        "every leg the call can go on; an image the model is not given as an image is replaced by a placeholder "
        "that says so where it stood, or named where it stood and counted (the header's images_not_sent says each "
        "kind); material the host treats as opaque replay (signatures, encrypted "
        "content, a tool call's thought signature) is withheld and counted wherever it is stored outside the "
        "transcript's own content, tool-call arguments and tool results; the host's bookkeeping keys and a replay "
        "carrier's metadata are not given; null, empty and blank values carry nothing and are not given as parts "
        "(inside a value given as JSON they stay as stored). Where that cannot be shown, the query is refused with "
        "the cause, and nothing is ever cut: the input is larger than the model's window less its output, where "
        "the plugin knows the window; the host's Anthropic converter would retire images on a leg the call can go "
        "on (by its own outbound_image_retire_count); an image one leg's converter delivers as an image and another's "
        "does not; a record cannot be given as it is (a record is stored as a JSON value that is not a message; a "
        "chunk begins with a tool result; a part the query gives has no recorded origin); the host's converter, on "
        "any leg the host can send the call on and under every value of that leg's inputs the plugin cannot read "
        "before the call (on an Anthropic or GitHub Copilot credential refresh, whether the host's image "
        "conversion runs there, and on an Anthropic one whether the token is an OAuth one; on an openai-codex or "
        "xai-oauth refresh the endpoint is read from the host's own sources before the call), would not give the "
        "model, in order, every non-blank text part, every "
        "image (as an image, not its bytes) and every tool call (by its name, and by its arguments where they parse "
        "as JSON) the query gives it; the host can send the call on a wire the plugin has not established, or "
        "retry after an authentication error on a provider's client whose wire the plugin has not established; the "
        "host could answer on a leg whose wire cannot be known before the call: a fallback provider configured "
        "under this route's own provider and model, a credential pool the host can rotate while fallback providers "
        "are configured, a managed NeMo Relay, or this call running as a managed Relay callback on a thread the "
        "plugin cannot establish; an openai-codex or xai-oauth credential refresh whose endpoint the query cannot "
        "read before the call; or a page cannot hold the result's header and one piece of it. Every other "
        "failure is "
        "an error that names its cause: an argument or handle the tool cannot take, an engine bound to no session, "
        "a session with nothing compacted yet, the session's summariser route or reasoning effort (including a "
        "store lock held past its busy timeout), a host function the query calls that cannot be read, the host's "
        "per-result and per-message limits leaving no page, the host asking the call to stop, the model's call or "
        "reply (another model answering, a cut or empty reply, a reply that is not the JSON asked for), a result "
        "that needs pages and cannot be built or stored, a page token that does not fit, any other failure before "
        "the call (named by its class), and the engine's own boundary (a closed engine, a host list it cannot "
        "settle, finishing the result), named by its step. An error raised before the model is called says that "
        "nothing was sent; one raised after it says what became of the reply; one on a page request says that no "
        "page was served. It returns a report and excerpts. The report is a model's description, asked to be "
        "hedged: orientation, never something to act on; the model is asked to name the handles it draws on, "
        "which lcm_expand opens (the query does not check them). Each excerpt was found verbatim in the record "
        "named by \"in\", in the fields named by \"from\" (each a path into the record as lcm_expand shows it, with "
        "that field's standing \"is\": a claim about where the text came from, which only the host's writer of the "
        "field settles): content, the message's content as the host stored it, and result, a tool result, may be "
        "relied on as an expansion may; call, a tool call's name or arguments (cited by the handle of the message "
        "that made the call), is what was called; reasoning is the model's account of its thinking, and nothing "
        "rests on it; sidecar is the text the host sent in place of the message's stored content (api_content: on "
        "an agent message the model's reasoning when its reply had no content, then also found in reasoning, a "
        "hook's output, or the host's interruption placeholder; on a user message the user's text with what the "
        "host injected; which, the host does not record); carrier is a text of the host's replay carrier of the "
        "message, or of its stash of a tool result's blocks, that the content does not hold (the provider's text "
        "before the host stripped it: stripped reasoning, a tool call written as text, or content the host "
        "altered; which, the host does not record); stored is a value under a key no producer of this host "
        "writes on the message's role. An excerpt found only in sidecar, carrier or stored fields is verbatim "
        "what the store holds there, and nothing rests on it as the message's content. An excerpt that does "
        "not pass the check is withheld, with the check it failed (its handle where it is one, never its text). A "
        "result longer than one page carries next_page: call again with page=next_page alone. The query calls its "
        "model once and tries nothing again: an error of the call names what the host's own recovery for that "
        "error can have changed on the route in use; a reply the query refuses is named with why, and what the "
        "host did before that reply is not known to the query; call again if you want to. If the host asks this "
        "call to stop (its tool timeout, or you are interrupted) and the query sees it before the model's reply is "
        "read, nothing more is sent or stored and the answer is lost; after the reply, a result that fits one page "
        "is still returned, and one that needs more pages is neither stored nor shown; a page asked for after the "
        "stop is not served. The query sees the host's stop by its interrupt bit and by the host's own tool "
        "timeout, which it counts from its own start: that starts after the host's, so for that latency after the "
        "host stopped waiting, and after an interrupt the host sets and clears again while the query runs one long "
        "host function of its check, the query can still call the model and store a result nobody reads. On some "
        "routes the model's request still runs on after the query stopped reading: every result's header says "
        "under call what happens on each wire the host can send the call on."
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
