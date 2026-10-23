# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/tobiaswuerth/sift/security/advisories/new). Don't open a public issue.
Expect a first reply within a week.

## Supported versions

Security fixes go into the latest release.

## Scope

sift runs locally. `sift serve` binds to `127.0.0.1` by default and has no authentication, so anyone who can reach
its port can search the indexed files and read them in full, through MCP and through the browser UI under `/ui`.
Expose it beyond your machine only behind something that authenticates; bound to any other address, it warns that
it can be read from elsewhere. Bound to a local address, both answer only requests that name a local host, so a web
page can't reach them under its own domain (DNS rebinding). Files opened from the UI carry
`Content-Security-Policy: sandbox`, so a saved web page in a collection can't run its scripts as the UI and read the
index; PDFs are left to the browser's viewer, which runs none of their scripts as the page.

sift never writes to the indexed files. One MCP tool changes anything, `reindex`, and only sift's index: it reads the
files it is given again, and only files that `sift index` would read too, inside a registered collection, outside
hidden and linked folders.

`sift worker` admits only requests with its token, over HTTPS with a certificate whose fingerprint the key carries.

sift downloads its models from Hugging Face on first use: the embedding model, the reranker and, for scans, the OCR
model. It loads their weights with transformers' own classes and runs no code that comes with a model. Scanned PDFs
are rendered with PDFium, which runs no scripts they hold; what OCR reads of a page is text, like any other.
