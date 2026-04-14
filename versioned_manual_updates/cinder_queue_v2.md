# Cinder Queue Manual v2

Cinder Queue remains the retry and queue manager used in the background job stack.
The default retry backoff mode is now `jittered_exponential`.
This change supersedes the earlier `fixed_30s` guidance for normal retry policy.
The main Cinder Queue config file remains `cinder.toml`.
The config file still lives in the service root and should be reviewed before rollout.
This manual captures the updated v2 retry behavior while preserving the same config path.
