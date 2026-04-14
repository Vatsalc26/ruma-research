# Marlin Proxy Manual v2

Marlin Proxy remains the service-edge proxy used in the gateway layer.
The default healthcheck command is now `marlin health`.
This change supersedes the earlier `marlin probe` guidance for standard health checks.
The main Marlin Proxy config file remains `proxy.ini`.
The config file still lives in the deployment bundle and should be checked before restarts.
This manual captures the updated v2 healthcheck command while preserving the config path.
