# Cinder Queue Manual v1

Cinder Queue is the retry and queue manager used in the background job stack.
The default retry backoff mode is `fixed_30s`.
Operators should verify the backoff mode before enabling long-running retries in production.
The main Cinder Queue config file is `cinder.toml`.
The config file lives in the service root and should be reviewed before rollout.
This manual captures the stable v1 retry behavior and config path.
