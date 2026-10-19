import { useEffect, useRef } from 'react'
import { requestJson } from './api'
import { pollUntil } from './poll'
import type { ProjectCredentials } from './types'

export type RunState = {
  status: string
  progress: number
  mode?: string
  input_mode?: 'annotated' | 'ai_qc'
  error_message?: string
}

export function useRunMonitor(session: ProjectCredentials | null) {
  const active = useRef<AbortController | null>(null)
  const cancel = () => active.current?.abort()
  useEffect(() => cancel, [session?.projectId, session?.recoveryKey])

  const monitor = async (runId: string, update: (state: RunState) => void | Promise<void>) => {
    cancel()
    if (!session) return
    const controller = new AbortController()
    active.current = controller
    try {
      await pollUntil(
        () => requestJson<RunState>(`/projects/${session.projectId}/runs/${runId}`, { signal: controller.signal }, session.recoveryKey),
        update,
        state => state.status === 'completed' || state.status === 'failed',
        controller.signal,
      )
    } catch (error) {
      if (!controller.signal.aborted) throw error
    }
  }
  return { monitor, cancel }
}
