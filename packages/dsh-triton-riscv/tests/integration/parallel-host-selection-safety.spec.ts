import { expect, it } from 'vitest'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'
import { parallelCall } from '../../lib/parallel-adapter.js'

const host =
  process.env.TRITON_PLUGIN_TEST_HOST ||
  fileURLToPath(new URL('../../../../thirdparty/deepseek-harness', import.meta.url))
const { Context } = await import(pathToFileURL(join(host, 'vendor/cordis/lib/index.js')).href)
const { Session, SessionId } = await import(pathToFileURL(join(host, 'packages/core/session/lib/index.js')).href)
const { default: SystemPrompt } = await import(
  pathToFileURL(join(host, 'packages/core/system-prompt/lib/index.js')).href
)
const { agentEvents, installModelSelection } = await import(
  pathToFileURL(join(host, 'packages/core/agent/lib/index.js')).href
)

it('follows real Harness model selection and restored request headers, not agent startup options', async () => {
  const ctx = new Context()
  try {
    await ctx.plugin(SystemPrompt)
    const startup = { provider: 'startup-provider', model: 'startup-model' }
    const selected = { provider: 'selected-provider', model: 'selected-model' }
    const session = Session.create(SessionId('native-selection-safety'))
    const agent = { id: session.id, session, options: startup }
    session.append('request/header', { header: { config: startup }, reason: 'initial' })
    const selection = { current: selected, assembled: undefined }
    installModelSelection(ctx, selection)
    await ctx.systemPrompt.assemble()
    const signal = new AbortController().signal
    const config = await agentEvents(ctx, agent).waterfall('agent/request', { turn: 1, step: 1, signal }, () =>
      Promise.resolve(startup),
    )
    session.append('request/header', { header: { config }, reason: 'change' })
    expect(config).toEqual(selected)
    expect(agent.options).toEqual(startup)

    for (const active of [agent, { ...agent, session: Session.create(session.id, session.snapshotEvents()) }]) {
      const requests: any[] = []
      const bridge = async request => {
        requests.push(request)
        return { plan_id: 'parallel-fixture', status: 'approved', digest: 'fixture' }
      }
      await parallelCall(
        {},
        { agent: active, arguments: {}, signal },
        'prepare_parallel_tasks',
        {},
        { save() {} },
        bridge,
        { message_id: 'human-fixture', text: 'Read independent sources' },
        '',
      )
      expect(requests[0].model_selection).toEqual(selected)
    }
  } finally {
    await ctx.fiber.dispose()
  }
})
