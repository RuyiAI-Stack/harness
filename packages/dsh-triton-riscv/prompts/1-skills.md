# Triton-RISCV Operator Skills

For an existing operator:

1. Call `mcp__triton_riscv__discover_operator`.
2. For risky changes, consult current evidence or call `mcp__triton_riscv__retrieve_operator_memory`.
3. Call `mcp__triton_riscv__check_validation_environment` before live validation.
4. Call `mcp__triton_riscv__validate_operator` with `execute=false` to plan.
5. Call `mcp__triton_riscv__execute_approved_validation` with the exact plan's `run_id` to request host approval and execution.

If remote validation loses its SSH connection, keep the original approved
`run_id`. Calling `execute_approved_validation` again with that exact ID can
query/collect the original durable remote job; it must not start a replacement.
Do not interpret disconnect as a test failure or propose a repair from it. If
recovery reports missing/inconsistent evidence, changed sources, or an unknown
worker outcome, stop and request host inspection. Never bypass this by making
a new plan or changing IDs. Recovery does not authorize reapplying code patches.
Remote jobs may be queued behind the account's two validation slots. A capacity
failure means the operator test did not start, not that its code is incorrect.
Do not repair source, submit duplicate jobs, change IDs, or clear server locks to
bypass capacity. Unknown reservations require host inspection before release.
Use `get_validation_status` with the original plan ID to inspect progress without
executing anything. Its default view is a timestamped local observation;
`remote=true` queries only the existing remote job. Distinguish execution completion
from test success. If remote execution finished but the local receipt is missing,
recover the same approved run to collect evidence; do not claim a verified pass.
Transient inspect/collect transport errors are retried at most three attempts
within the original wall-clock budget. Start, approval, write, cleanup and stop
requests are never automatically replayed by this retry policy.
A host stop request is not proof of remote shutdown. Only a supervisor terminal
cancellation confirmation proves this job's descendants stopped. Cancelled tests
are neither operator failures nor passes; do not repair source from cancellation.
The local journal may remain unresolved after stopping. Ask the approving host
to inspect/reconcile it before creating a new plan; do not bypass it with a new ID.

For a remote operator batch prepared by this version, keep its original `job_id`.
After a disconnect, call `get_validation_job` to read local progress, then
`execute_validation_job` with the SAME job ID to resume within the original
15-minute wall-clock deadline. The host verifies the entire batch first, replays
committed child results, collects the active original child and starts only
never-started items. A failed test remains failed; an unknown outcome pauses
the batch and is not permission to repair or resubmit it. Cancelled, expired,
changed-source or inconsistent batches require host inspection. Old batches and
local project jobs have no automatic recovery. Do not call individual child
plans or create a replacement batch to bypass this lifecycle.

For a new operator:

1. Obtain name, semantics, PyTorch reference, input/output contract, shapes, dtypes, tolerances and backward requirement.
2. Call `mcp__triton_riscv__prepare_operator_development`.
3. Consult history; if needed call `mcp__triton_riscv__retrieve_operator_memory` with semantics and reference.
4. Generate implementation and independent acceptance-test source using the contract and repository references.
5. Call `mcp__triton_riscv__propose_operator_implementation`; do not directly create tracked files with generic shell/filesystem tools.
6. Call `mcp__triton_riscv__apply_development_proposal` to request approval, then use the validation lifecycle.
