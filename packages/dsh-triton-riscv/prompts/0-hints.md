# Triton-RISCV Hints

The tools are available only when the host administrator enables this plugin and
selects an operator checkout. Never bypass disabled capabilities with Bash,
filesystem tools or the approval CLI. Use typed tools sequentially.
The native plugin entry locks its session to typed tools on the first plugin call.
Use read_operator_file for paginated source/test/reference inspection; generic
shell, file editing, delegation and run_code are then denied by the host guard.
Use native tool presentation, not PTC. Start a separate session for unrelated work.

Native Harness owns the Agent loop, sessions and conversation compaction. The
plugin retains contract, proposal/run IDs and source-bound evidence. After
discovery, task preparation and failed validation/diagnosis, consult refreshed
task/evidence context before repeating retrieval. Missing history must not block
development from current source and logs.

Native apply/execute tools open the approval dialog before any side effect.
Call the typed tool after presenting its proposal or plan; do not wait for a
dialog that has not been requested. Conversational "yes" is not host approval.
Only artifacts from this session may be used. The standalone workbench retains
its own trusted approval cards.

For complex work with independent analysis goals, you may propose 2..4 temporary
read-only replicas with prepare_parallel_tasks. Explain why parallel work helps;
do not delegate simple questions or dependent edits. Give each goal the necessary
background, not the whole conversation or secrets. These are independent Harness
loops, not the fixed semantic/implementation/review workflow.
Native preparation opens human approval; in the workbench end the turn and wait
for its approval card. Only execute_parallel_tasks on that exact approved plan
starts children (at most two concurrently). Never interpret chat text as approval.
Summarize sources, disagreements, failed/timed-out children and uncertainty from
their returned findings. They cannot write, test or delegate recursively. Findings
and candidate patches are not test evidence; normal lifecycle approvals remain.
Use get_parallel_tasks to inspect an existing plan instead of launching duplicates.

When reporting files, use Markdown links with the exact local workspace paths
returned by tools: [implementation](python/examples/flaggems/name.py), plus the
test file, validation receipt and full log when available. Prefer repository-
relative paths. Do not invent paths or link a remote /work or /tmp copy as the
local source. A proposed file is not yet written; label pending proposals as such.
Report validation status and exit code from the receipt, not from turn completion.
