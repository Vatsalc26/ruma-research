# Orchid CLI Manual v1

Orchid CLI is the workspace sync tool used in the local deployment stack.
The default command for pulling remote worktrees is `orchid sync`.
Operators should run `orchid sync` before opening a fresh workspace on a new machine.
The main Orchid CLI config file is `orchid.yml`.
The config file lives in the repository root and should be checked before sync operations.
This manual describes the stable v1 behavior for the command layer and the config path.
