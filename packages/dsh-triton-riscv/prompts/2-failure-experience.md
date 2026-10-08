# Triton-RISCV Failure Experience

On failure call `mcp__triton_riscv__diagnose_failure` with the current run before
proposing repair. Historical failures are references, not diagnoses of this run.
Compare environment/compiler versions, dtype, shape and first failing stage.
Keep unknown causes unknown. Distinguish recommended actions, actually applied
changes and observed outcomes; never present a recommendation as a repair.

Keep acceptance tests unchanged. Propose an implementation-only repair, request
host approval, apply it and revalidate. Stop at the bounded repair limit or an
unmodifiable environment/compiler fault. Report `run_id`, `proposal_id`, failure
stage, receipt, log and concrete required user action. Do not fabricate cases
when retrieval is empty.
