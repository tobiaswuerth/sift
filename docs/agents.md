# Setting sift up with an agent

Instructions for an AI coding agent (Claude Code, Cursor and the like) that installs sift and connects it for the
person it works for. To use them, tell your agent:

> Set up sift for me: read https://raw.githubusercontent.com/tobiaswuerth/sift/main/docs/agents.md and follow it.

The rest of this page addresses the agent.

## What you set up

sift indexes folders of documents on this machine and serves keyword, semantic and hybrid search over them as an MCP
server. You are done when the folders the user named are indexed and embedded, `sift serve` runs, and a new session of
the user's agent lists sift's tools. sift reads the user's files and never writes to them: all it writes lies under
`SIFT_HOME` (`~/.sift`).

## Ask first

Ask the user, in one message, what only they know:

1. Which folders to index, and a short name for each (`notes`, `handbook`).
2. Whether another machine's GPU should embed for this one (then set that up first: [worker.md](worker.md)).

Index no home folder and no whole drive unasked. sift reads every file it has a reader for, and whoever reaches the
server can search and read all of it: the server has no authentication, which is why it listens on this machine alone.

## Steps

Read what each command prints: sift says what it did, what it left out and what to run next (on stderr; results are
on stdout).

1. **Check the machine.** sift needs Python 3.11 or newer; [uv](https://docs.astral.sh/uv/) installs it best
   (`uv --version`). It uses an NVIDIA GPU if there is one, else an Apple GPU, else the CPU, which is much slower at
   embedding and at `query`.

2. **Install.**

   ```bash
   uv tool install --torch-backend auto sift-rag
   sift --version
   ```

   Without uv: `pip install sift-rag` in a virtual environment. If sift later says that an NVIDIA GPU sits unused, run
   the command it prints.

3. **Add each folder** as a collection:

   ```bash
   sift add ~/handbook --name handbook
   sift add ~/vault --name vault --title-from name   # an Obsidian vault (it holds .obsidian/): titles by file name
   ```

   - Files to leave out go into a `.siftignore` in the folder, in gitignore's syntax
     ([usage](usage.md#leaving-files-out)).
   - JSON Lines with other field names than sift looks for: `--id`, `--title`, `--text`
     ([usage](usage.md#json-lines)).
   - A file that is a link to outside its folder is left out, and said. Leave that so unless the user asks for
     `--follow-links`: such a link may lead to any file of the machine.

4. **Index, then embed.**

   ```bash
   sift index    # reads the files and builds the keyword index
   sift embed    # embeds every passage without a vector
   ```

   The first `sift embed` downloads the models from Hugging Face (about 2 GB). Both commands process only what
   changed since their last run, and `sift embed`, interrupted, goes on where it stopped. On a GPU it embeds a few
   hundred passages a second, so a large collection takes a while: run it where the user sees its progress, and don't
   let a timeout of yours end it. `sift status` prints each collection's `docs`, `chunks` and `embedded` as JSON; it
   is embedded when the last two are equal.

5. **Try it.**

   ```bash
   sift query "a question the documents answer" -k 3
   ```

6. **Start the server** where it outlives your session: a terminal the user keeps open, or their service manager.

   ```bash
   sift serve    # MCP on http://127.0.0.1:8765/mcp, the browser UI on http://127.0.0.1:8765/ui/
   ```

   It loads the models once and serves any number of agents. Another port: `--port N`.

7. **Connect the agent.** For Claude Code, once for all projects:

   ```bash
   claude mcp add --scope user --transport http sift http://127.0.0.1:8765/mcp
   ```

   Any other MCP client registers a streamable HTTP server at that address, as its own documentation says. A client
   that only starts servers itself (stdio) runs `sift serve --stdio`, which loads the models for every session.

8. **Check and hand over.** A new agent session lists sift's tools (`claude mcp list`); on connecting, the server
   tells the agent which tool fits which question and which collections exist. Tell the user what you indexed, where
   the UI is, and that `sift serve` must run for agents to search.

## Keeping the index current

- After writing or deleting a file of a collection, call the `reindex` tool with its path: the next search finds it,
  embedded.
- For files changed otherwise, `sift index` and then `sift embed`, as often as the user wants them current. Both may
  run while the server does.
- sift offers its own updates in a terminal; see [Updating](https://github.com/tobiaswuerth/sift#updating).

## Only if the user asks

- `sift rm NAME` deletes a collection's index, and `sift embed --reset` every vector: embedding again takes as long
  as the first time.
- `sift serve --host` with any address but this machine's own: everyone who reaches the port reads every indexed
  document ([security](SECURITY.md)).
- `SIFT_EMBED_MODEL`: another model means `sift embed --reset`.

## When something goes wrong

- A file skipped or left out is named with the reason, and read again once it changes.
- What `add`, `index`, `embed` and `serve` said is kept in `SIFT_HOME/logs`.
- "port ... is taken": another `sift serve` runs there already (use it), or pick another with `--port`.
- Everything else: [usage](usage.md), and `sift COMMAND --help`.
