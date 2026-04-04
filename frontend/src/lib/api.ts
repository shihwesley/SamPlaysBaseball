export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export type Pitcher = { id: string; name: string; team: string; throwingArm: 'L' | 'R' }
export type Outing = { id: string; date: string; pitchCount: number; pitchTypes: string[] }
export type Pitch = { id: string; type: string; velocity: number; spin: number }
export type Analysis = {
  pitchId: string
  angles: Record<string, number[]>
  deviations: Record<string, number>
  phases: Record<string, number>
  fatigue: number[]
}
export type ComparisonResult = {
  pitch1Id: string
  pitch2Id: string
  divergenceScore: number
  phaseAlignment: number[]
}

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts?.headers ?? {}) },
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export const getPitchers = () => apiFetch<Pitcher[]>('/api/pitchers')
export const getPitcher = (id: string) => apiFetch<Pitcher>('/api/pitchers/' + id)
export const getOutings = (pitcherId: string) =>
  apiFetch<Outing[]>('/api/pitchers/' + pitcherId + '/outings')
export const getOuting = (pitcherId: string, outingId: string) =>
  apiFetch<Outing>('/api/pitchers/' + pitcherId + '/outings/' + outingId)
export const getPitches = (pitcherId: string, outingId: string) =>
  apiFetch<Pitch[]>('/api/pitchers/' + pitcherId + '/outings/' + outingId + '/pitches')
export const getPitch = (pitchId: string) => apiFetch<Pitch>('/api/pitches/' + pitchId)
export const getAnalysis = (pitchId: string) => apiFetch<Analysis>('/api/analysis/' + pitchId)
export const comparePitches = (id1: string, id2: string) =>
  apiFetch<ComparisonResult>('/api/compare?pitch1=' + id1 + '&pitch2=' + id2)

export async function uploadVideo(
  file: File,
  pitcherId: string,
  metadata: Record<string, unknown>,
): Promise<{ jobId: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('pitcherId', pitcherId)
  form.append('metadata', JSON.stringify(metadata))
  const res = await fetch(API_BASE + '/api/upload', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<{ jobId: string }>
}

export const getExportUrl = (pitchId: string): string =>
  API_BASE + '/api/export/glb/' + pitchId
