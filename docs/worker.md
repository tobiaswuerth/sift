# Embedding on another machine

A machine without a GPU, such as a notebook, can borrow another machine's. The worker embeds and reranks; the machine
that uses it keeps the index and the files.

## Setting up

1. On the machine with the GPU, start the worker:

   ```bash
   sift worker            # listens on port 8766 and prints the address to use, with its key
   ```

   It prints a line such as `SIFT_EMBED_REMOTE=https://<key>@workstation:8766`.

2. On the other machine, set that variable. In PowerShell, `setx` keeps it for new terminals; in bash, use `export`:

   ```powershell
   setx SIFT_EMBED_REMOTE https://<key>@workstation:8766
   ```

3. Let the worker through the firewall, from your own network only. On Windows, in PowerShell as administrator:

   ```powershell
   New-NetFirewallRule -DisplayName "sift worker" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow -RemoteAddress LocalSubnet
   ```

   If Windows asks whether to let Python through the firewall, allow it: cancelling adds a rule that blocks Python,
   and blocking rules win over allowing ones. On Linux with ufw: `sudo ufw allow from 192.168.1.0/24 to any port 8766
   proto tcp`, with your network's range.

## What it does

- `sift embed` sends the chunks to the worker, 256 at a time, and stores the vectors it gets back, without loading the
  model itself.
- Searches embed their queries and rerank there too, in the CLI and in `sift serve`, so the machine searching never
  loads PyTorch: `vsearch` and `query` answer in a fraction of a second, even on a light notebook.
- The worker only computes: it sees the texts it embeds and reranks, but no index and no files. It sizes its batches
  to its GPU and gives the GPU back after 5 idle minutes, like `sift serve`.
- Both machines need the same major version of sift; sift refuses a worker of another one, and says which runs where.
- The worker's `SIFT_EMBED_MODEL` decides the model, and sift refuses a worker whose model differs from the one the
  index was embedded with.

## Security

The texts travel encrypted, over HTTPS. On its first start the worker makes a certificate of its own and an access
token, kept in `~/.sift/worker` (readable by you only), and the key it prints holds both: the token, and the
certificate's fingerprint. sift checks the worker's certificate against that fingerprint before it sends anything, so
no other machine can pose as the worker, and the worker answers only requests that carry the token.

Keep the key private, like a password. The worker shows it on its terminal when it starts, to copy, and nowhere else:
not in its log, and sift never in an error message. To make a new key, delete `~/.sift/worker` and restart the
worker; machines with the old key are then refused.

The protocol is described in [architecture.md](https://github.com/tobiaswuerth/sift/blob/main/docs/architecture.md#worker-protocol).
