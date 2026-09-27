# Recall tools

Use recall tools when the answer depends on historical evidence that may have been compacted.

The tools read what the session's compactions stored. What was said since the last compaction is in the context and not yet stored.

## Current compacted conversation

### `lcm_grep`

Use to find which stretch of the stored session a term lies in, when you do not know which summary covers it.

- `term` is found exactly: case-sensitive, character for character, no pattern or query syntax. It is searched in what was said (the messages as sent), each tool call's name and arguments, tool results and readable reasoning; never in summaries.
- It returns each chunk the term lies in, whole, as `lcm_expand` returns it, named with the summary (`under`) that covers it in your context; in the collapsed form, `results_holding_term` names the tool results that hold the term, to open with `lcm_expand`. `raw=true` puts every result inline.
- At three or more matching chunks only the count comes back: narrow the term, give `scope` (a summary's or chunk's handle), or pass `all=true`.
- Records of the fresh tail stored at the last compaction that hold the term are named by handle; what came after the last compaction is in your context and not searched. `searched` says what was searched and when it was stored.
- A result longer than one page carries `next_page`; pass it as `page` for the rest.

### `lcm_query`

Use to ask what stands behind summaries or chunks in your context without loading it: the tool for every day.

- Give `handles` (summaries `s…` or chunks `c…`) and a `question`. Everything behind every handle is read whole, in one call, by the model that writes your summaries; input too large for it is refused with its size, never cut. A tool call's or a message's handle is refused: open those with `lcm_expand`.
- It returns a report and excerpts, each an item whose side `lcm` holds what the plugin says: the report (`part: "report"`, `text`), each checked excerpt (`excerpt`, the `handle` it was cited from, `in`, the record it was found in verbatim, and `text`), and each withheld excerpt (`withheld`, `handle`, `length`, `why`; its text is not shown).
- The report is a model's description, hedged: orientation, never something to act on. Each checked excerpt may be relied on as an expansion may. A withheld excerpt's `why` says which check it did not pass. The header says which chunks were read, when the compaction whose active record was read began (`stored_at`), the model and the effort asked for, the input's estimate, and whether the input was checked against the model's window.
- A result longer than one page carries `next_page`; call again with `page` alone. The host's tool timeout can cut a long call off; its answer is then lost.

### `lcm_expand`

Use as low-level drill-down after a known handle (`handle`):

- every item of a page has two sides: `lcm`, what this plugin says of it (its handle, role, notes, the readable reasoning, each tool call's handle and where its result is under `calls`), and `message`, the stored message's own keys as the host keeps them, each tool call in the host's shape; a key under `message` is always the host's, whatever its name. A part too large for one page comes as pieces: `lcm.field` says where the piece lies as the list of steps into the item (`["message", "content", 0]`, `["lcm", "reasoning"]`: a string is a key, a number a position in a list), `lcm.offset` and `lcm.chars` where the slice lies in that field (`lcm.json` when the field is not a string and the slice is of its JSON), and `text` or `value` holds the slice; a piece of a tool call's arguments carries that call's `lcm.call` and the call without its arguments as `tool_call`. The pieces of an item, each put back at its place, over all its pages are exactly the whole item. The same shape holds for `lcm_grep`. This is a change of the page shape: the plugin's keys and the message's keys were one object before, and a piece's `field` was a string such as `content[0]`; a `next_page` token of the earlier shape is refused;

- a summary's (`s…`) or a chunk's (`c…`) handle returns that stretch: the messages verbatim, the readable reasoning beside them, each tool call with its handle (`t…`), name and arguments, without its result; `raw=true` puts the results inline;
- a tool call's handle returns its result: the stored result that carries the call's id, before the next user or agent message; where none does, a note says so; where calls of one message share an id, every result of that id is returned and a note says the store cannot tell which answered which call; a message's handle (`m…`) returns that message;
- an image comes back as an image where the session's route carries it, at most as many per page as the host carries in one request: in the item's content it stands as `{"type": "image", "image": n}`, the n-th image of the page, which follows the page's text. Otherwise it stands as a mark with its media type, its size, its message's handle and the cause it is not shown, as the check that held it states it, and the store keeps it. `lcm.images` lists where the plugin put an image or a mark (`field`, and `shown`), so a stored part of the same shape is told apart. The host may still retire a shown image with its own placeholder when a request holds too many (`lcm.images_note` says so), and expanding again returns it; `lcm.route_note` and `lcm.guard_note` say where the route or the host's guard could not be read;
- a handle whose message the host has since rewritten is refused with the handle that stands for it now; expand that one;
- a result longer than one page carries `next_page`; pass it as `page` for the rest; if what the handle opens into changed since the first page, the page is refused and you start again from the handle. A page never leaves anything out: where one part cannot fit even on a page by itself, the call is refused, naming that part, its size and the page's limit, which is larger when `lcm_expand` is the only call in its message.

Do not use it as broad first-step discovery.

## Operator tools

`lcm_status`, `lcm_inspect`, and `lcm_doctor` report health and metadata. They do not replace content retrieval.
