# Embedding on another machine

A machine without a GPU, such as a notebook, can embed and rerank on another machine's GPU. The worker only computes;
the machine using it keeps the index and the files.

## Setting up

1. On the machine with the GPU, start the worker (port 8766, `--port N` for another):

   ```bash
   sift worker
   ```

   It prints a line such as `SIFT_EMBED_REMOTE=https://<key>@workstation:8766`.

2. On the other machine, set that variable: `export` in bash, or in PowerShell, for new terminals:

   ```powershell
   setx SIFT_EMBED_REMOTE https://<key>@workstation:8766
   ```

3. Let the worker through the firewall, from your own network only. On Windows, in PowerShell as administrator:

   ```powershell
   New-NetFirewallRule -DisplayName "sift worker" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow -RemoteAddress LocalSubnet
   ```

   If Windows asks whether to let Python through, allow it: cancelling adds a rule that blocks Python, and blocking
   rules win. On Linux with ufw: `sudo ufw allow from 192.168.1.0/24 to any port 8766 proto tcp`, with your network's
   range.

## What it does

- `sift embed` sends the chunks to the worker, 256 at a time, and stores the vectors it returns. Searches embed their
  queries and rerank there too, so the machine that embeds, searches or serves through the worker never loads PyTorch.
- The worker sees the texts it embeds and reranks, but no index and no files. It sizes its batches to its GPU and frees
  the GPU after 5 idle minutes, as `sift serve` does.
- Both machines need the same major version of sift; sift refuses another and says which runs where.
- The worker's `SIFT_EMBED_MODEL` decides the model. `sift embed` and `reindex` refuse a worker whose model differs from
  the index's; searches only check the vector length, so keep the worker's model the index's.

## Security

The texts travel over HTTPS. On its first start the worker makes a certificate and an access token, kept in
`SIFT_HOME/worker` (readable by you only); the key it prints holds the token and the certificate's fingerprint. sift
checks the worker's certificate against that fingerprint before sending anything, so no other machine can pose as the
worker, and the worker answers only requests carrying the token.

Keep the key private, like a password. The worker shows it on its terminal at start and nowhere else, not in its log,
and sift never puts it in an error message. To make a new key, delete the `worker` folder and restart the worker;
machines with the old key are then refused. The protocol: [architecture.md](architecture.md#worker-protocol).
