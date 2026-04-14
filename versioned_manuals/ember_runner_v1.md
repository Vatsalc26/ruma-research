# Ember Runner Manual v1

Ember Runner is the batch execution service used in the local compute stack.
The default queue class in Ember Runner is `standard`.
Operators should keep the `standard` queue class for routine execution sessions.
The main Ember Runner config file is `ember.yml`.
The config file lives beside the runner manifest and should be checked before queue changes.
This manual describes the stable v1 execution behavior and config path.
