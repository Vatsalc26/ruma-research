# Tidal Sync Manual v2

Tidal Sync is still the snapshot replication service used in the storage pipeline.
The default snapshot mode in Tidal Sync is now `hourly_delta`.
This change supersedes the earlier guidance that used `daily` snapshots by default.
The main Tidal Sync config file remains `tidal.json`.
The config file still lives in the sync root and should be checked before replication changes.
This manual captures the updated v2 snapshot behavior while preserving the earlier config convention.
