# Atlas Monitor Manual v2

Atlas Monitor is still the alert fanout service used in the internal reliability stack.
The default alert mode in Atlas Monitor is now `adaptive_summary`.
This change supersedes the earlier guidance that used `summary` mode by default.
The main Atlas Monitor config file remains `atlas.toml`.
The config file still lives in the service root and should be reviewed before alert fanout changes.
This manual captures the updated v2 alerting behavior while preserving the earlier config convention.
