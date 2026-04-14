# Marlin Proxy Manual v1

Marlin Proxy is the service-edge proxy used in the gateway layer.
The default healthcheck command is `marlin probe`.
Operators should run the healthcheck command before changing traffic weights.
The main Marlin Proxy config file is `proxy.ini`.
The config file lives in the deployment bundle and should be checked before restarts.
This manual describes the stable v1 healthcheck command and config path.
