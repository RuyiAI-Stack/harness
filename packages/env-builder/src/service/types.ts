export type ComponentStatus = 'ready' | 'modified' | 'installing'

export interface EnvComponent {
  owner: string
  repo: string
  url: string
  dir: string
  status: ComponentStatus
  sessionId?: string
}

export interface EnvRecord {
  id: string
  path: string
  components: EnvComponent[]
  running: boolean
  sessionIds: string[]
}

export interface EnvManifest {
  version: 1
  environments: EnvRecord[]
  selectedId?: string
  selectedRepoRef?: string
}

const STATUSES = new Set<ComponentStatus>(['ready', 'modified', 'installing'])

export function parseComponentStatus(status: unknown): ComponentStatus {
  if (typeof status !== 'string' || !STATUSES.has(status as ComponentStatus)) {
    throw new Error(`env-builder: invalid component status ${String(status)}`)
  }
  return status as ComponentStatus
}
