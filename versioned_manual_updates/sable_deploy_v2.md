# Sable Deploy Manual v2

Sable Deploy is still the rollout controller used in the release pipeline.
The default rollout mode in Sable Deploy is now `canary_safe`.
This change supersedes the earlier guidance that used `linear` rollout by default.
The main Sable Deploy config file remains `sable.yaml`.
The config file still lives in the deployment root and should be checked before rollout changes.
This manual captures the updated v2 rollout behavior while preserving the earlier config convention.
