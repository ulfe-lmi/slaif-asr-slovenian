# CLAUDE.md

Read and obey [`AGENTS.md`](AGENTS.md) as the authoritative project constitution.

The following rules are repeated because they are critical:

- This is a standalone SLAIF adaptation repository, not a NeMo fork.
- Work only from a bounded PR-sized work order.
- Do not commit model weights, audio corpora, experiment checkpoints, secrets, private transcripts, or local absolute paths.
- Preserve explicit attribution to `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Reuse the base tokenizer unless an approved ADR and work order say otherwise.
- Start with the smallest declared trainable surface and do not escalate silently.
- Synthetic-data improvement alone does not justify checkpoint acceptance.
- Report passed, failed, skipped, not run, blocked, and out-of-scope work separately.
- Do not publish model artifacts and do not merge your own pull request.
- The remote repository and committed evidence are project truth.
- Read the relevant ADRs, policies, and work order before editing.
- End every task with the structured report defined in `AGENTS.md`.
