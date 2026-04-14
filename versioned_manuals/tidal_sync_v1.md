# Tidal Sync Manual v1

Tidal Sync is the snapshot replication service used in the storage pipeline.
The default snapshot mode in Tidal Sync is `daily`.
Operators should keep the `daily` snapshot mode for routine replication cycles.
The main Tidal Sync config file is `tidal.json`.
The config file lives in the sync root and should be checked before replication changes.
This manual describes the stable v1 snapshot behavior and config convention.
