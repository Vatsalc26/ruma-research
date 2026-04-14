# Orchid CLI Manual v2

Orchid CLI is still the workspace sync tool used in the local deployment stack.
The default command for pulling remote worktrees is now `orchid pull`.
This change supersedes the earlier guidance that used `orchid sync` for the same task.
The main Orchid CLI config file remains `orchid.yml`.
The config file still lives in the repository root and should be checked before sync operations.
This manual captures the updated v2 command behavior while preserving the earlier config convention.
