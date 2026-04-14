# Nova Build Manual v2

Nova Build remains the build runner for packaging and release preparation.
The default cache mode in Nova Build is now `shared_remote`.
This change supersedes the older local-cache default for the main build path.
The main manifest file for Nova Build remains `nova.json`.
Teams should still keep `nova.json` at the repository root so the build runner can discover it without extra flags.
This manual records the updated v2 cache behavior while retaining the same manifest rule.
