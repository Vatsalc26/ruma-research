# Quartz Scheduler Manual v2

Quartz Scheduler remains the orchestration scheduler used for batch planning.
The default schedule mode is now `adaptive`.
This change supersedes the earlier `steady` schedule mode for normal planning workloads.
The main Quartz Scheduler config file remains `quartz.yaml`.
The config file still lives in the orchestration directory and should be checked before rollout.
This manual captures the updated v2 scheduling mode while preserving the config path.
