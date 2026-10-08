const PREFIX = 'mcp__triton_riscv__'

export const OPERATOR_TOOLS = new Set(
  [
    'discover_operator',
    'read_operator_file',
    'check_validation_environment',
    'inspect_project',
    'prepare_validation_job',
    'execute_validation_job',
    'get_validation_job',
    'get_validation_status',
    'evaluate_plugin_fixtures',
    'retrieve_operator_memory',
    'execute_approved_validation',
    'apply_development_proposal',
    'get_operator_implementation_proposal',
    'prepare_operator_development',
    'propose_operator_implementation',
    'apply_operator_implementation',
    'validate_operator',
    'diagnose_failure',
    'propose_repair',
    'apply_repair',
  ].map(name => PREFIX + name),
)

// This guards the host dispatcher, not just descriptions shown to the model.
// Activation is durable and starts before the first domain tool body runs.
export function createToolPolicy(store) {
  return exec => {
    const isDomain = OPERATOR_TOOLS.has(exec.name)
    if (!exec.agent?.session?.id)
      return exec.name.startsWith(PREFIX) ? 'Triton-RISCV tools require a native Harness session' : undefined
    const state = store.load(exec.agent.session.id)
    const protectedSession = state.operator_tools_only === true || Object.keys(state.artifacts).length > 0
    if (isDomain) {
      if (!state.operator_tools_only) store.save({ ...state, operator_tools_only: true })
      return undefined
    }
    if (protectedSession || exec.name.startsWith(PREFIX))
      return 'Operator session is restricted to typed Triton-RISCV tools. Use read_operator_file for source inspection and approved lifecycle tools for changes/tests; generic shell, edit, delegation and run_code are not permitted. Use a separate session for unrelated work.'
    return undefined
  }
}
