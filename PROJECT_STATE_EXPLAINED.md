# Project State Explained

This file is the plain-language orientation note for RUMA.

## 1. What This Project Actually Is

RUMA is currently a `version-aware external memory architecture project`.

That means the project is mainly about:

- how knowledge is stored outside dense weights
- how updates are written without retraining everything
- how newer guidance supersedes older guidance
- how conflicting active sources stay visible
- how retrieval stays grounded and inspectable

It is **not** currently a frontier standalone language model.

## 2. Is RUMA Standalone Or Attached To An LLM?

Right now, RUMA is best understood as `a memory-and-answering architecture that can sit beside a base model`.

Today in the repo, the system has:

- a memory layer
- routing
- updates
- retrieval
- citation-first answering

What it does **not** yet have is a strong standalone generative language-model core that could compete directly with GPT-style systems by itself.

So the honest answer is:

- `current RUMA`: not a standalone frontier LLM
- `current RUMA`: a retrieval-and-memory system prototype that can sit on top of, beside, or be paired with a base model
- `future RUMA`: could later be paired with a small LLM core more tightly, but that is not the first-paper claim

## 2A. Were We Trying To Build A New Architecture Like Transformer?

Yes, that was the original high-level ambition.

But there are `two different levels` of "architecture":

1. `module-level architecture`
   A new memory, routing, and update system that improves how a model handles changing knowledge.

2. `full-model architecture`
   A complete standalone model backbone that could compete directly with transformer-style systems.

RUMA is already a real `module-level architecture` project.

RUMA is **not yet** a proven `full-model architecture` replacement.

That is not failure. That is normal research sequencing.

The scientifically responsible path is:

- first prove the memory/update architecture works
- then pair it tightly with a model core
- then ask whether it deserves to become a larger standalone architecture claim

So the project did not "change into something else."

It became more precise.

## 3. What The First Paper Is Really About

The first paper is **not**:

- "we built the next universal model"
- "we solved hallucination"
- "we solved sycophancy"

The first paper is much narrower:

- versioned document updates
- retained guidance after updates
- same-lineage supersession
- visible conflict handling
- routed retrieval with inspectable evidence

That narrower paper is much more realistic and much more respectable.

## 4. Why The Project Has So Many Benchmarks

The benchmarks are not random busywork.

They answer different questions:

- synthetic benchmarks: does the write/read path work at all?
- chunk benchmarks: can retrieved text actually affect answers?
- manual benchmarks: do updates, retention, and conflicts work on changing docs?
- routing benchmarks: is a heavier router actually needed?
- scaling benchmarks: when does unrestricted search become too expensive?

Without these, the project would stay conceptual.

## 5. Where The Project Is Right Now

RUMA is in the `execution phase`.

That means:

- the main planning-stage architecture decisions are closed
- the repo now has real benchmarks, real docs, real updates, and paper assets
- the main remaining work is evidence expansion, systems hardening, and preprint packaging

Another way to say it:

- `original dream`: maybe a new frontier architecture
- `current stage`: prove the most distinctive subsystem first
- `later stage`: if the subsystem keeps winning, expand the claim

## 6. What The Next Practical Finish Line Is

The next real finish line is:

`a narrow preprint on version-aware routed external memory for changing documents`

That finish line needs:

- a somewhat larger changing-document corpus
- stronger failure and ablation tables
- one more borrowed retrieval-infrastructure step if the environment supports it
- a public-release cleanup pass

This does **not** close the door on a larger future architecture paper.

It just means the first finish line is:

- smaller
- faster
- more defensible
- more likely to get published honestly

## 7. What We Should Not Confuse This With

Do not confuse the current project with:

- training a brand-new GPT-scale model from scratch
- proving a full replacement for transformer architectures
- a finished product system

Those are different and much larger projects.

## 8. What Success Looks Like In The Near Term

Near-term success means:

- the repo can defend a narrow research claim
- the update path beats naive alternatives on controlled changing-doc tasks
- the paper is honest and publishable as a preprint
- the system architecture is clean enough that future work can extend it

That is already a meaningful result.

## 9. What The Finished Output Looks Like

If we finish the current route well, the output should be:

- a public research repo
- a narrow preprint
- repeatable benchmarks
- a clear memory/update architecture
- a system others can inspect and build on

If later evidence stays strong, the next output could be:

- a stronger second paper
- tighter coupling to a language model core
- a broader architecture claim
