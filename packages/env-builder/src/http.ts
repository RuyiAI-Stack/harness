export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

export function statusOf(err: unknown): number {
  if (err instanceof HttpError) return err.status
  return 500
}
