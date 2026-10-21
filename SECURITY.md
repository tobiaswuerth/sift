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
Expose it beyond your machine only behind something that authenticates. Bound to a local address, both answer only
requests that name a local host, so a web page can't reach them under its own domain (DNS rebinding).
