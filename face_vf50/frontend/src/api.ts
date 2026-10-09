import type {
  FrameDetail,
  FramesResponse,
  SessionInfo,
} from './types';

const API_BASE = '/api';

export async function checkBackendHealth(): Promise<{ status: string; version: string; port: number }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function getSessions(): Promise<SessionInfo[]> {
  const res = await fetch(`${API_BASE}/sessions`);
  if (!res.ok) throw new Error('Failed to fetch sessions');
  return res.json();
}

export async function getSessionDetail(sessionId: string): Promise<SessionInfo> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
  if (!res.ok) throw new Error('Failed to fetch session detail');
  return res.json();
}

export async function getSessionFrames(
  sessionId: string,
  page: number = 1,
  pageSize: number = 50,
  sortBy: string = 'severity',
  filterRule?: string
): Promise<FramesResponse> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
    sort_by: sortBy,
  });
  if (filterRule && filterRule !== 'all') {
    params.append('filter_rule', filterRule);
  }
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frames?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch session frames');
  return res.json();
}

export async function getFrameDetail(sessionId: string, frameIndex: number): Promise<FrameDetail> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frames/${frameIndex}`);
  if (!res.ok) throw new Error('Failed to fetch frame detail');
  return res.json();
}

export async function setFrameStatus(
  sessionId: string,
  frameIndex: number,
  status: 'pending' | 'reviewed' | 'fixed'
): Promise<{ status: string }> {
  const formData = new FormData();
  formData.append('status', status);
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frames/${frameIndex}/status`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to update frame status');
  return res.json();
}

export async function createDemoSession(): Promise<{ session_id: string; total_issues: number }> {
  const res = await fetch(`${API_BASE}/demo/create-sample`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to create demo session');
  return res.json();
}

export async function uploadDataset(
  formData: FormData
): Promise<{ session_id: string; total_frames: number; total_issues: number }> {
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export function getImageUrl(sessionId: string, filename: string): string {
  return `${API_BASE}/images/${sessionId}/${encodeURIComponent(filename)}`;
}

export function getChecklistDownloadUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/export/checklist`;
}

export function getCsvDownloadUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/export/csv`;
}

export function getCvatXmlDownloadUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/export/cvat-xml`;
}

export function getJsonDownloadUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/export/json`;
}

export async function applyModelFix(
  sessionId: string,
  frameIndex: number,
  pointIds?: number[]
): Promise<FrameDetail> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frames/${frameIndex}/apply-model-fix`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ point_ids: pointIds || null }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to apply model fix' }));
    throw new Error(err.detail || 'Failed to apply model fix');
  }
  return res.json();
}

export async function batchApplyModelFix(
  sessionId: string,
  onlyViolated: boolean = true,
  minNme: number = 0.035
): Promise<{
  success: boolean;
  session_id: string;
  updated_count: number;
  updated_frame_indices: number[];
  total_frames: number;
  session_stats: { total_frames: number; total_issues: number; avg_nme: number };
}> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/batch-apply-model-fix`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ only_violated: onlyViolated, min_nme: minNme }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to batch apply model fix' }));
    throw new Error(err.detail || 'Failed to batch apply model fix');
  }
  return res.json();
}

export async function setPointSource(
  sessionId: string,
  frameIndex: number,
  pointId: number,
  source: 'human' | 'model'
): Promise<FrameDetail> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frames/${frameIndex}/point-source`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ point_id: pointId, source }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to update point source' }));
    throw new Error(err.detail || 'Failed to update point source');
  }
  return res.json();
}
