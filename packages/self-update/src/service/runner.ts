import { spawn, type ChildProcess } from 'node:child_process'
import { createWriteStream, mkdirSync, type WriteStream } from 'node:fs'
import { dirname, join } from 'node:path'

export type RunPhase = 'idle' | 'running' | 'ok' | 'failed'

export interface RunState {
  phase: RunPhase
  pid?: number
  startedAt?: number
  finishedAt?: number
  exitCode?: number | null
  error?: string
}

export interface RunnerOptions {
  /** Absolute path of the install script to execute. */
  script: string
  /** Working directory of the script (the repo root). */
  cwd: string
  /** Absolute path of the log file; truncated at the start of each run. */
  logPath: string
  /** Relaunch the host after a successful update. */
  restart: boolean
}

/**
 * Runs the repo install script once in the background and records its state.
 * At most one run may be active; a finished run keeps its final state and log.
 */
export class InstallRunner {
  private state: RunState = { phase: 'idle' }
  private child?: ChildProcess
  private stream?: WriteStream

  constructor(private readonly options: RunnerOptions) {}

  get logPath(): string {
    return this.options.logPath
  }

  snapshot(): Readonly<RunState> {
    return { ...this.state }
  }

  /** Whether a run has been started (so a log file may exist). */
  logExists(): boolean {
    return this.state.phase !== 'idle'
  }

  /**
   * Start a new background run. Returns the new state, or undefined when a run
   * is already active. The log is truncated before the script starts.
   */
  start(): Readonly<RunState> | undefined {
    if (this.state.phase === 'running') return undefined

    mkdirSync(dirname(this.options.logPath), { recursive: true })
    this.stream = createWriteStream(this.options.logPath, { flags: 'w' })
    this.state = { phase: 'running', startedAt: Date.now() }

    const child = spawn('bash', [this.options.script], {
      cwd: this.options.cwd,
      env: {
        ...process.env,
        DSH_HOME: join(this.options.cwd, '.dsh'),
        DSH_PROFILE: 'web',
        CI: 'true',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    this.child = child
    this.state.pid = child.pid

    child.stdout?.on('data', (chunk: Buffer) => this.write(chunk))
    child.stderr?.on('data', (chunk: Buffer) => this.write(chunk))
    child.on('error', (error: Error) => this.finish(error))
    child.on('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      if (signal) this.finish(new Error(`killed by ${signal}`), code)
      else if (code === 0) this.finish(undefined, 0)
      else this.finish(new Error(`exit code ${code ?? 'unknown'}`), code)
    })

    return this.snapshot()
  }

  /** Ask the active child to terminate (used on plugin dispose). */
  kill(): void {
    this.child?.kill('SIGTERM')
  }

  private write(chunk: Buffer): void {
    this.stream?.write(chunk)
  }

  private finish(error: Error | undefined, exitCode: number | null = null): void {
    this.stream?.end()
    this.stream = undefined
    this.child = undefined
    this.state = {
      phase: error ? 'failed' : 'ok',
      pid: this.state.pid,
      startedAt: this.state.startedAt,
      finishedAt: Date.now(),
      exitCode: error ? exitCode : 0,
      ...(error ? { error: error.message } : {}),
    }
    if (!error && this.options.restart) this.maybeRestart()
  }

  /**
   * Relaunch the host after a successful update. The restarter is detached so it
   * survives the host's death, then kills this process and starts a fresh
   * `./dsh web` with the inherited environment (DEEPSEEK_API_KEY etc.).
   */
  private maybeRestart(): void {
    const pid = process.pid
    const script = [
      'sleep 3',
      `kill ${pid} 2>/dev/null || true`,
      'sleep 1',
      `cd ${JSON.stringify(this.options.cwd)} && nohup ./dsh web >> ${JSON.stringify(this.options.logPath)} 2>&1 &`,
    ].join('; ')
    spawn('bash', ['-c', script], { detached: true, stdio: 'ignore', env: process.env }).unref()
  }
}
