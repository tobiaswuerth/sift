# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/tobiaswuerth/sift/security/advisories/new). Don't open a public issue.
Expect a first reply within a week. Security fixes go into the latest release.

## Scope

- **The home process.** `sift serve` binds to `127.0.0.1` alone and answers only requests with its token: 256 random
  bits, kept with its port in your own folder (`%LOCALAPPDATA%\sift` on Windows, `~/.local/state/sift` elsewhere),
  readable by you only, never in the home. The token is the whole command line: whoever holds it may add any folder
  you can read, index it, and read its text. It answers only requests naming a local host, so a web page can't reach
  it under its own domain (DNS rebinding).
- **The home.** It holds the text of every indexed file. On Windows a home outside your profile folder takes the
  rights of the folder it lies in, often readable by every local user: keep it in a folder only you may read. On
  other systems sift makes the home and its stores yours alone.
- **The UI.** Files opened from it carry `Content-Security-Policy: sandbox`, so a saved web page in a collection can't
  run scripts as the UI; PDFs go to the browser's viewer. It refuses what another site's page has the browser fetch
  in the background (by Fetch Metadata), and no other site may show it in a frame; a navigation to it, as a link
  followed, opens it.
- **Your files.** sift never writes to them. One MCP tool changes anything, `reindex`, and only sift's index: it reads
  only files `sift index` would read, inside a registered collection, outside hidden and linked folders. A linked file
  is read only if it points into its collection's folder, unless the collection was added with `--follow-links`: a
  link in a folder from elsewhere, a cloned repository, may lead to any file of the machine.
- **The worker** admits only requests with its token, over HTTPS pinned by its key
  ([details](worker.md#security)).
- **Models** come from Hugging Face on first use (embedding, reranker and, for scans, OCR), each at the commit this
  version of sift pins it to: what a model's owner pushes later reaches no one until a release of sift moves the pin.
  sift loads their weights with transformers' own classes and runs no code shipped with a model. Scans are rendered
  with PDFium, which runs no scripts they hold.
- **Updates.** Commands run in a terminal ask PyPI for sift's latest version once a day, in a process of their own: a
  plain HTTPS GET of `pypi.org/pypi/sift-rag/json`, sending nothing of you or your files but sift's version.
  `SIFT_UPDATE_CHECK=off` or `HF_HUB_OFFLINE=1` stops it. sift updates itself only when you answer `y`, with the
  installer it came by (uv, pipx or pip).
