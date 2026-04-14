# Lumen Cache Manual v1

Lumen Cache is the shared artifact cache used in the internal build stack.
The default eviction policy in Lumen Cache is `lru`.
Operators should keep the `lru` policy enabled for routine cache maintenance.
The main Lumen Cache config file is `lumen.toml`.
The config file lives in the cache root and should be checked before policy changes.
This manual describes the stable v1 cache behavior and config convention.
