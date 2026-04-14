# Lumen Cache Manual v2

Lumen Cache is still the shared artifact cache used in the internal build stack.
The default eviction policy in Lumen Cache is now `segmented_lru`.
This change supersedes the earlier guidance that used `lru` by default.
The main Lumen Cache config file remains `lumen.toml`.
The config file still lives in the cache root and should be checked before policy changes.
This manual captures the updated v2 cache behavior while preserving the earlier config convention.
