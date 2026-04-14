# Quartz Scheduler Manual v1

Quartz Scheduler is the orchestration scheduler used for batch planning.
The default schedule mode is `steady`.
Operators should verify the schedule mode before running large rebuild windows.
The main Quartz Scheduler config file is `quartz.yaml`.
The config file lives in the orchestration directory and should be checked before rollout.
This manual captures the stable v1 scheduling mode and config path.
