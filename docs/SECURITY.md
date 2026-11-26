# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/tobiaswuerth/sift/security/advisories/new). Don't open a public issue.
Expect a first reply within a week. Security fixes go into the latest release.

## Scope

- **The home process.** `sift serve` listens on `127.0.0.1` alone, and answers only requests addressed to `127.0.0.1`
  or `localhost`, so a web page can't reach it under a domain of its own (DNS rebinding).
  - Your `sift` commands send it a token: 256 random bits, kept with its port in your own folder
    (`%LOCALAPPDATA%\sift` on Windows, `$XDG_STATE_HOME/sift` or `~/.local/state/sift` elsewhere), readable by you
    alone, never in the home. The token grants the whole command line: whoever holds it may add any folder you can
    read, index it, read its text, move or remove collections. A home process killed hard leaves that file behind
    until the next `sift serve` replaces it; it names a port nothing answers on any more.
  - MCP (`/mcp`) and the UI (`/ui`) take no token, so that agents and browsers connect as they are: every program on
    this machine, of any user, can search and read the documents of the collections agents may see through MCP, and
    of all collections through the UI. MCP refuses requests a web page of another site sends (by their `Origin`); the
    UI's pages run no scripts, and no other site may show them or a file opened from them in a frame, or load them
    into a page of its own: the UI answers a page of another site only when you follow a link there, so no site can
    make it search in the background (by `Sec-Fetch-Site`, which browsers since 2023 send). On a machine others use,
    run `sift serve` only while you need it.
- **The home.** It holds the text of every indexed file. On Windows a home outside your profile folder takes the
  rights of the folder it lies in, often readable by every local user: keep it in a folder only you may read. On
  other systems sift makes the home and its stores' folders yours alone when it makes them.
- **The UI.** Files opened from it carry `Content-Security-Policy: sandbox`, so a saved web page in a collection can't
  run scripts as the UI; PDFs go to the browser's viewer.
- **Agents.** What a document says reaches an agent as a tool's answer, to be read as data. The instructions sift
  gives an agent when it connects name your collections, their folders and their front matter fields, a field only if
  its name is one plain word of at most 40 characters: a file of a collection can put such a word into them, not a
  sentence.
- **Your files.** sift never writes to them. One MCP tool changes anything, `reindex`, and only sift's stores: it reads
  only files `sift index` would read, inside a collection agents may see, outside hidden and linked folders. A linked
  file is read only if it points into its collection's folder, unless the collection was added with `--follow-links`:
  a link in a folder from elsewhere, a cloned repository, may lead to any file of the machine.
- **The worker** admits only requests with its token, over HTTPS pinned by its certificate
  ([details](worker.md#security)). It listens on every address of its machine unless `--host` says otherwise.
- **Models** come from Hugging Face on first use (embedding, reranker and, for scans, OCR), each at the commit this
  version of sift pins it to: what a model's owner pushes later reaches no one until a release of sift moves the pin.
  Another model must be named at a whole commit too. Scans are rendered with PDFium, which runs no scripts they hold.
- **Updates.** Off unless `SIFT_UPDATE_CHECK=on`. Then a command run in a terminal asks PyPI for sift's latest version
  at most once a day, in its own process, waiting at most 5 seconds: a plain HTTPS GET of
  `https://pypi.org/pypi/sift-rag/json` with `User-Agent: sift/<version>`, sending nothing of you or your files.
  `HF_HUB_OFFLINE=1` stops it too. sift never updates itself: it says which command would.
