# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/tobiaswuerth/sift/security/advisories/new). Don't open a public issue.
Expect a first reply within a week. Security fixes go into the latest release.

## Scope

- **The server.** `sift serve` binds to `127.0.0.1` by default and has no authentication: anyone who can reach its port
  can search the indexed files and read them in full, through MCP and the UI. Expose it only behind something that
  authenticates; bound to another address, it warns. Bound locally, it answers only requests naming a local host, so a
  web page can't reach it under its own domain (DNS rebinding).
- **The UI.** Files opened from it carry `Content-Security-Policy: sandbox`, so a saved web page in a collection can't
  run scripts as the UI; PDFs go to the browser's viewer.
- **Your files.** sift never writes to them. One MCP tool changes anything, `reindex`, and only sift's index: it reads
  only files `sift index` would read, inside a registered collection, outside hidden and linked folders. A linked file
  is read as the file it points to, wherever that lies.
- **The worker** admits only requests with its token, over HTTPS pinned by its key
  ([details](worker.md#security)).
- **Models** come from Hugging Face on first use (embedding, reranker and, for scans, OCR). sift loads their weights
  with transformers' own classes and runs no code shipped with a model. Scans are rendered with PDFium, which runs no
  scripts they hold.
- **Updates.** Commands run in a terminal ask PyPI for sift's latest version once a day, in a process of their own: a
  plain HTTPS GET of `pypi.org/pypi/sift-rag/json`, sending nothing of you or your files but sift's version.
  `SIFT_UPDATE_CHECK=off` or `HF_HUB_OFFLINE=1` stops it. sift updates itself only when you answer `y`, with the
  installer it came by (uv, pipx or pip).
