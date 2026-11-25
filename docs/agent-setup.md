# Setting sift up with an agent

Instructions for an AI coding agent (Claude Code, Cursor and the like) that installs sift and connects it for the
person it works for. To use them, tell your agent:

> Set up sift for me: read https://raw.githubusercontent.com/tobiaswuerth/sift/main/docs/agent-setup.md and follow it.

The rest of this page addresses the agent.

## What you set up

sift indexes folders of documents on this machine and serves keyword, semantic and hybrid search over them as an MCP
server. You are done when the folders the user named are indexed and embedded, `sift serve` runs, and a new session of
the user's agent lists sift's tools. sift reads the user's files and never writes to them: what it writes lies under
`SIFT_HOME` (`~/.sift`), or where `sift add --store` puts a store, and the models it downloads in Hugging Face's
cache (`~/.cache/huggingface`).

## Ask first

Ask the user, in one message, what only they know:

1. Which folders to index, and a short name for each (`notes`, `handbook`).
2. Whether another machine's GPU should embed for this one (then set that up first: [worker.md](worker.md)).

Index no home folder and no whole drive unasked. sift reads every file it has a reader for, and every program running
on this machine can search and read all of it through the server, which listens on this machine alone.

## Steps

Read what each command prints: sift says what it did, what it left out and what to run next (on stderr; results are
on stdout).

1. **Check the machine.** sift needs Python 3.12 or newer; [uv](https://docs.astral.sh/uv/) installs it best
   (`uv --version`), and brings a Python of its own. Without uv, ask the user before you install it, as
   [its documentation](https://docs.astral.sh/uv/getting-started/installation/) says. sift uses an NVIDIA GPU if there
   is one, else an Apple GPU, else the CPU, which is much slower at embedding and at `query`.

2. **Install.**

   ```bash
   uv tool install --torch-backend auto sift-rag
   sift --version
   ```

   If the shell finds no `sift` after that, uv's folder of tools isn't on its `PATH`: `uv tool update-shell` adds it
   for new shells, and `uv tool dir --bin` prints the folder for this one. Without uv: `pip install sift-rag` in a
   virtual environment. If sift later says that an NVIDIA GPU sits unused, run the command it prints. If the user ran
   sift 1.x before, run `sift import` once (it says so itself): it brings those collections in with their vectors.

3. **Add each folder** as a collection:

   ```bash
   sift add ~/handbook --name handbook
   sift add ~/vault --name vault    # an Obsidian vault (it holds .obsidian/) is titled by file name
   ```

   - Files to leave out go into a `.siftignore` in the folder, in gitignore's syntax
     ([usage](usage.md#leaving-files-out)).
   - JSON Lines with other field names than sift looks for: `--id-field`, `--title-field`, `--text-fields`
     ([usage](usage.md#json-lines)).
   - A file that is a link to outside its folder is left out, and said. Leave that so unless the user asks for
     `--follow-links`: such a link may lead to any file of the machine.

4. **Index, then embed.**

   ```bash
   sift index    # reads the files into each collection's store
   sift embed    # embeds every passage without a vector
   ```

   The first `sift embed` downloads the models from Hugging Face (about 2 GB). Both commands process only what
   changed since their last run, and `sift embed`, interrupted, goes on where it stopped. On a GPU it embeds a hundred
   or more passages a second, so a large collection takes a while: run it where the user sees its progress, and don't
   let a timeout of yours end it. `sift status` shows each collection's documents, passages and how many are
   embedded; with `--json`, `contents` holds `documents`, `chunks` and `embedded`, and a collection is embedded when
   the last two are equal.

5. **Try it.**

   ```bash
   sift query "a question the documents answer" -k 3
   ```

6. **Start the server** where it outlives your session. A command you start in your own shell ends with it, so have
   the user run it in a terminal they keep open, or start it apart from yours:

   ```bash
   sift serve    # MCP on http://127.0.0.1:8765/mcp, the browser UI on http://127.0.0.1:8765/ui
   ```

   ```bash
   nohup sift serve > /dev/null 2>&1 &    # Linux and macOS: apart from this shell
   ```

   ```powershell
   Start-Process sift -ArgumentList serve -WindowStyle Minimized    # Windows: a window of its own, minimized
   ```

   It loads the models once, at the first search that needs them, and serves any number of agents and the user's own
   `sift` commands: it is up once `http://127.0.0.1:8765/ui` answers, and what it says is in `SIFT_HOME/logs`. Another
   port: `--port N`. To have it start with the machine, ask the user: that is their service manager's (systemd,
   launchd, Task Scheduler).

7. **Connect the agent.** For Claude Code, once for all projects:

   ```bash
   claude mcp add --scope user --transport http sift http://127.0.0.1:8765/mcp
   ```

   Any other MCP client registers a streamable HTTP server at that address, as its own documentation says. A client
   that only starts servers itself (stdio) runs `sift serve --stdio`, which talks to the running `sift serve`, or starts
   one that leaves after 15 idle minutes.

8. **Check and hand over.** A new agent session lists sift's tools (`claude mcp list`); on connecting, the server
   tells the agent which tool fits which question and which collections exist. Tell the user what you indexed, where
   the UI is, and that `sift serve` must run for agents to search over HTTP.

## Keeping the index current

- After writing or deleting a file of a collection, call the `reindex` tool with its path: the next search finds it,
  embedded.
- For files changed otherwise, `sift index` and then `sift embed`, as often as the user wants them current. While
  `sift serve` runs, it does what they ask.
- sift never updates itself; with `SIFT_UPDATE_CHECK=on` it says when a newer version is out. See
  [Updating](https://github.com/tobiaswuerth/sift#updating).

## Only if the user asks

- `sift rm NAME` deletes a collection's store, and `sift embed --reset` every vector: embedding again takes as long
  as the first time.
- `sift add FOLDER --no-agents`: a collection the user searches alone, which the MCP tools neither list nor search.
- `SIFT_EMBED_MODEL`: another model means `sift embed --reset`.

## When something goes wrong

- A file skipped or left out is named with the reason, and read again once it changes. One that couldn't be read
  just then (access denied, a reader's error) is tried again by the next `sift index`; `sift status --unreadable`
  lists them.
- What `add`, `index`, `embed`, `serve` and the other commands that change collections said is kept in
  `SIFT_HOME/logs`.
- "port 8765 cannot be had": another program listens there; pick another port with `--port`. "another sift serves this
  home": use that one.
- Everything else: [usage](usage.md), and `sift COMMAND --help`.
