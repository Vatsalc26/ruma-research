# Ember Runner Manual v2

Ember Runner is still the batch execution service used in the local compute stack.
The default queue class in Ember Runner is now `priority_burst`.
This change supersedes the earlier guidance that used the `standard` queue class by default.
The main Ember Runner config file remains `ember.yml`.
The config file still lives beside the runner manifest and should be checked before queue changes.
This manual captures the updated v2 execution behavior while preserving the earlier config convention.
