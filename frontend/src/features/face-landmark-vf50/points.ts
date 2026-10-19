import type { Point, FaceResultDetail } from './types'
import { VF50_GROUPS } from './vf50'

export type PointDisplayMode = 'suspect' | 'all'

export function pointAt(points: Record<string, Point>, id: number): Point | undefined {
  return points[String(id)]
}

export function usable(point?: Point): point is Point {
  return !!point && point.state !== 'outside' && point.x !== null && point.y !== null
}

export function suspiciousPointIds(result: FaceResultDetail): number[] {
  const ids = new Set<number>()
  result.violations.forEach(item => item.points.forEach(id => ids.add(id)))
  Object.entries(result.quality?.point_evaluations || {}).forEach(([id, evaluation]) => {
    if (evaluation.exceeded || evaluation.severity === 'CRITICAL' || evaluation.severity === 'WARNING') ids.add(Number(id))
  })
  return [...ids].filter(id => id >= 0 && id < 50).sort((a, b) => a - b)
}

export function pointGroup(id: number): string {
  const group = VF50_GROUPS.find(([, start, end]) => id >= start && id <= end)
  return group?.[0] || 'VF50'
}
