export const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export function authHeaders(recoveryKey: string): Record<string, string> {
  return { Authorization: `Bearer ${recoveryKey}` }
}

export async function requestJson<T>(path: string, init: RequestInit = {}, recoveryKey = '', fallback?: string): Promise<T> {
  const headers = new Headers(authHeaders(recoveryKey))
  new Headers(init.headers).forEach((value, name) => headers.set(name, value))
  const response = await fetch(`${API}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = typeof payload?.detail === 'string' ? payload.detail : payload?.detail?.message
    throw new Error(detail || fallback || `Request failed (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return response.json()
}
