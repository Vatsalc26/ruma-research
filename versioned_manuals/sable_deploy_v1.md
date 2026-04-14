# Sable Deploy Manual v1

Sable Deploy is the rollout controller used in the release pipeline.
The default rollout mode in Sable Deploy is `linear`.
Operators should keep the `linear` rollout mode for routine staged releases.
The main Sable Deploy config file is `sable.yaml`.
The config file lives in the deployment root and should be checked before rollout changes.
This manual describes the stable v1 rollout behavior and config convention.
