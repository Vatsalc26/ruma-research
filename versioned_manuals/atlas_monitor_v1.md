# Atlas Monitor Manual v1

Atlas Monitor is the alert fanout service used in the internal reliability stack.
The default alert mode in Atlas Monitor is `summary`.
Operators should keep `summary` mode enabled during routine service watches.
The main Atlas Monitor config file is `atlas.toml`.
The config file lives in the service root and should be reviewed before alert fanout changes.
This manual describes the stable v1 alerting behavior and config convention.
