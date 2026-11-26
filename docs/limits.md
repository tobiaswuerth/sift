# Limits

Every bound sift sets in its code: the most it reads, holds or sends, the fewest it needs, and how long it waits.
Generated from the code by `python -m tools limits --write`; a test holds it equal to the code.

| Module | Name | Value |
|---|---|---|
| `sift.api.collections` | `MOST_NAMED` | `1_000` |
| `sift.api.described` | `MOST_LISTED` | `1_000` |
| `sift.api.documents` | `MOST_LINE` | `2**31 - 1` |
| `sift.api.documents` | `MOST_DOCUMENTS` | `1_000` |
| `sift.api.filters` | `MOST_INTEGER` | `2**63` |
| `sift.api.filters` | `MOST_VALUES` | `100_000` |
| `sift.api.ids` | `MOST_COLLECTIONS` | `(1 << (63 - ROW_BITS)) - 1` |
| `sift.api.indexing` | `MOST_FILES` | `10_000` |
| `sift.api.jobs` | `MOST_JOB` | `(1 << 63) - 1` |
| `sift.api.requests` | `MOST_SHOWN` | `100` |
| `sift.api.search` | `MOST_HITS` | `10_000` |
| `sift.api.search` | `MOST_LINKS` | `100` |
| `sift.api.serving` | `MOST_PORT` | `65535` |
| `sift.api.serving` | `MOST_IDLE_SECONDS` | `604_800` |
| `sift.app.bench` | `MOST_FILE_BYTES` | `64 << 20` |
| `sift.app.bench` | `MOST_NESTING` | `8` |
| `sift.app.documents` | `MOST_LISTED` | `5` |
| `sift.app.importing` | `KEY_BYTES` | `8` |
| `sift.app.kept_decoded` | `KEPT_BYTES` | `512 << 20` |
| `sift.app.kept_decoded` | `ENTRY_BYTES` | `512` |
| `sift.app.problems` | `MOST_NAMED` | `3` |
| `sift.app.vector_candidates` | `MOST_WIDER` | `WIDER**2` |
| `sift.domain.codec.codes` | `WORD_BYTES` | `8` |
| `sift.domain.codec.codes` | `MOST_BITS` | `int(np.iinfo(np.uint16).max)` |
| `sift.domain.codec.postings` | `MOST_GAP` | `1 << 32` |
| `sift.domain.codec.postings` | `MOST_POSITION` | `1 << 32` |
| `sift.domain.codec.postings` | `GAP_BYTES` | `4` |
| `sift.domain.codec.postings` | `LENGTH_BYTES` | `4` |
| `sift.domain.codec.postings` | `COUNT_BYTES` | `1` |
| `sift.domain.codec.postings` | `BEST_BYTES` | `1` |
| `sift.domain.codec.postings` | `MOST_BYTES` | `9` |
| `sift.domain.codec.postings` | `MOST_INFLATED_BYTES` | `64 << 20` |
| `sift.domain.codec.segments` | `MOST_TIMES` | `1024` |
| `sift.domain.codec.shape` | `MOST_RANGE_SHIFT` | `8` |
| `sift.domain.codec.shape` | `MOST_BLOCK` | `1 << 12` |
| `sift.domain.codec.shape` | `MOST_LENGTHS_SHIFT` | `20` |
| `sift.domain.plan.indexing` | `MOST_TIME` | `2**63 - 1` |
| `sift.domain.plan.indexing` | `MOST_FILE_BYTES` | `1 << 30` |
| `sift.domain.plan.segments` | `FEWEST_JOINED` | `2` |
| `sift.domain.query.fields` | `MOST_EXACT_DIGITS` | `300` |
| `sift.domain.query.grammar` | `MOST_DEPTH` | `100` |
| `sift.domain.query.grammar` | `MOST_WORDS` | `1000` |
| `sift.domain.query.grammar` | `MOST_QUERY_CHARS` | `100_000` |
| `sift.domain.query.grammar` | `MOST_ECHOED` | `60` |
| `sift.domain.query.grammar` | `FEWEST_RELAXED` | `2` |
| `sift.domain.rank.bm25` | `LEAST_IDF` | `1e-6` |
| `sift.domain.rank.hybrid` | `FEWEST_TO_CUT` | `3` |
| `sift.domain.rank.measures` | `MOST_GRADE` | `10` |
| `sift.domain.rank.phrases` | `MOST_KEY` | `1 << 62` |
| `sift.domain.rank.works` | `MOST_DOI` | `1_000` |
| `sift.domain.text.chunks` | `MOST_CHARACTERS` | `3000` |
| `sift.domain.text.chunks` | `MOST_OVERLAP` | `300` |
| `sift.domain.text.decoding` | `MARK_BYTES` | `4` |
| `sift.domain.text.decoding` | `ESCAPED_BYTES` | `range(0xDC80, 0xDD00)` |
| `sift.domain.text.front_matter` | `MOST_FLOW_NESTING` | `32` |
| `sift.domain.text.heads` | `MOST_HEAD_BYTES` | `256 << 10` |
| `sift.domain.text.heads` | `MOST_COMMENT_NESTING` | `100` |
| `sift.domain.text.json_values` | `MOST_DEPTH` | `999` |
| `sift.domain.text.markdown` | `MOST_TITLE` | `1_000` |
| `sift.domain.text.markdown` | `MOST_LINE` | `1 << 20` |
| `sift.domain.text.notes` | `MOST_HEAD` | `1 << 20` |
| `sift.domain.text.notes` | `DIGEST_BYTES` | `16` |
| `sift.domain.text.records` | `MOST_RECORD_CHARACTERS` | `64 << 20` |
| `sift.domain.text.tokens` | `MOST_FIELD_CHARACTERS` | `1 << 20` |
| `sift.inference.accelerator` | `LEAST_BUDGET` | `2 * GIB` |
| `sift.inference.hub` | `STAMP_BYTES` | `8` |
| `sift.inference.local` | `MOST_TOKENS` | `1024` |
| `sift.inference.local` | `IDLE_SECONDS` | `300.0` |
| `sift.inference.remote` | `MOST_TEXTS` | `256` |
| `sift.inference.remote` | `CONNECT_SECONDS` | `10.0` |
| `sift.inference.remote` | `ANSWER_SECONDS` | `600.0` |
| `sift.inference.remote` | `SHOWN_BYTES` | `200` |
| `sift.inference.remote` | `HALF_BYTES` | `2` |
| `sift.ingest.converter` | `MOST_COMPRESSION` | `10_000` |
| `sift.ingest.ignore` | `MOST_IGNORE_BYTES` | `1 << 20` |
| `sift.ingest.links` | `MOST_HOPS` | `40` |
| `sift.ingest.mail` | `MOST_NESTED` | `50` |
| `sift.ingest.readers` | `MOST_TEXT` | `100_000_000` |
| `sift.ingest.records` | `MARK_BYTES` | `4` |
| `sift.ingest.sandbox` | `MOST_WORKERS` | `30` |
| `sift.ingest.sandbox` | `MOST_ANSWER_BYTES` | `2 << 30` |
| `sift.ingest.sandbox` | `HEARD_SECONDS` | `1.0` |
| `sift.interfaces.cli.updates` | `MOST_BYTES` | `32 << 20` |
| `sift.interfaces.cli.written` | `MOST_BRACKETS` | `64` |
| `sift.interfaces.console.live` | `REDRAW_SECONDS` | `0.1` |
| `sift.interfaces.console.live` | `HEARTBEAT_SECONDS` | `60.0` |
| `sift.interfaces.console.live` | `TICK_SECONDS` | `0.5` |
| `sift.interfaces.console.live` | `STOP_SECONDS` | `5.0` |
| `sift.interfaces.mcp.answers` | `MOST_TEXT` | `100_000` |
| `sift.interfaces.mcp.holdings` | `MOST_FIELDS_LISTED` | `200` |
| `sift.interfaces.mcp.instructions` | `MOST_FIELDS` | `30` |
| `sift.interfaces.mcp.searches` | `MOST_QUERIES` | `20` |
| `sift.interfaces.server.agents` | `JOB_WAIT_SECONDS` | `20.0` |
| `sift.interfaces.server.app` | `ALIVE_SECONDS` | `30.0` |
| `sift.interfaces.server.app` | `MOST_BODY` | `1 << 20` |
| `sift.interfaces.server.app` | `MOST_DEPTH` | `64` |
| `sift.interfaces.server.client` | `QUIET_SECONDS` | `120.0` |
| `sift.interfaces.server.client` | `MOST_LINE` | `1 << 30` |
| `sift.interfaces.server.endpoint` | `STARTING_SECONDS` | `2.0` |
| `sift.interfaces.server.endpoint` | `LOOK_EVERY_SECONDS` | `0.05` |
| `sift.interfaces.server.gate` | `ALONE_PATIENCE_SECONDS` | `300.0` |
| `sift.interfaces.server.progress` | `PROGRESS_EVERY_SECONDS` | `0.1` |
| `sift.interfaces.server.proxy` | `IDLE_SECONDS` | `900` |
| `sift.interfaces.server.proxy` | `STARTING_SECONDS` | `60.0` |
| `sift.interfaces.server.proxy` | `LOOK_EVERY_SECONDS` | `0.05` |
| `sift.interfaces.server.proxy` | `QUIET_SECONDS` | `600.0` |
| `sift.interfaces.server.serving` | `IDLE_CHECK_SECONDS` | `30.0` |
| `sift.interfaces.server.serving` | `STOP_SECONDS` | `5.0` |
| `sift.interfaces.server.serving` | `GRACE_SECONDS` | `10` |
| `sift.interfaces.server.serving` | `KEEP_ALIVE_SECONDS` | `5` |
| `sift.interfaces.server.serving` | `MOST_CONNECTIONS` | `256` |
| `sift.interfaces.worker.app` | `MOST_BODY` | `32 << 20` |
| `sift.interfaces.worker.app` | `MOST_NESTING` | `4` |
| `sift.interfaces.worker.keys` | `TOKEN_BYTES` | `32` |
| `sift.interfaces.worker.serving` | `GRACE_SECONDS` | `10` |
| `sift.interfaces.worker.serving` | `KEEP_ALIVE_SECONDS` | `5` |
| `sift.interfaces.worker.serving` | `MOST_CONNECTIONS` | `64` |
| `sift.storage.catalog` | `MOST_NAME` | `100` |
| `sift.storage.database` | `BUSY_TIMEOUT_SECONDS` | `30` |
| `sift.storage.database` | `WAL_LIMIT_BYTES` | `64 << 20` |
| `sift.storage.documents` | `MOST_DESCRIPTION_BYTES` | `512 << 20` |
| `sift.storage.kept_vectors` | `KEY_BYTES` | `16` |
| `sift.storage.keyword_lists` | `MOST_POINT` | `0x10FFFF` |
| `sift.storage.keywords` | `MOST_ROW` | `(1 << 63) - 1` |
| `sift.storage.writer` | `MOMENT_SECONDS` | `0.5` |
| `sift.storage.writer` | `ASK_AGAIN_SECONDS` | `0.05` |
