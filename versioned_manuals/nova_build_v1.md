# Nova Build Manual v1

Nova Build is the build runner for packaging and release preparation.
The default cache mode in Nova Build is `local`.
The local cache mode is the standard default in the first release line.
The main manifest file for Nova Build is `nova.json`.
Teams should keep `nova.json` at the repository root so the build runner can discover it without extra flags.
This manual captures the stable v1 defaults for cache behavior and manifest lookup.
