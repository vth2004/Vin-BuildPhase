/** Sequential polling: no overlapping requests, and no updates after cancellation. */
export async function pollUntil<T>(
  read: () => Promise<T>,
  update: (value: T) => void | Promise<void>,
  done: (value: T) => boolean,
  signal: AbortSignal,
  intervalMs = 1000,
): Promise<void> {
  while (!signal.aborted) {
    const value = await read()
    if (signal.aborted) return
    await update(value)
    if (done(value) || signal.aborted) return
    await new Promise<void>(resolve => {
      const finish = () => {
        clearTimeout(timer)
        signal.removeEventListener('abort', finish)
        resolve()
      }
      const timer = setTimeout(finish, intervalMs)
      signal.addEventListener('abort', finish, { once: true })
      if (signal.aborted) finish()
    })
  }
}
