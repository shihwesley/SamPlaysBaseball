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
  if (!res.ok) {
    const text = await res.text()
    let msg = text
    try { msg = JSON.parse(text).detail ?? text } catch {}
    throw new Error(msg)
  }
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
  if (!res.ok) {
    const text = await res.text()
    let msg = text
    try { msg = JSON.parse(text).detail ?? text } catch {}
    throw new Error(msg)
  }
  return res.json() as Promise<{ jobId: string }>
}

export const getExportUrl = (pitchId: string): string =>
  API_BASE + '/api/export/glb/' + pitchId

// ---------------------------------------------------------------------------
// Query / Diagnostic API
// ---------------------------------------------------------------------------

export type DiagnosticReport = {
  narrative: string
  recommendations: string[]
  risk_flags: string[]
  confidence: 'high' | 'moderate' | 'low'
  pitches_analyzed: number
}

export type ViewerInfo = {
  glb_url: string
  phase_markers: Record<string, number>
  total_frames: number
}

export type QueryResponse = {
  status: 'complete'
  report: DiagnosticReport
  statcast: {
    group_a: Record<string, number>
    group_b: Record<string, number>
  }
  viewer: ViewerInfo
  query: {
    raw_text?: string
    parsed?: Record<string, unknown>
    pitches_used?: string[]
  }
}

export type ProgressResponse = {
  status: 'processing'
  token: string
  pitches_needing_inference: string[]
  total_pitches: number
  eta_seconds: number | null
}

export type QueryResult = QueryResponse | ProgressResponse

export const submitQuery = (text: string, gameDate?: string) =>
  apiFetch<QueryResult>('/api/query', {
    method: 'POST',
    body: JSON.stringify({ text, game_date: gameDate ?? null }),
  })

export type BlenderOpenResponse = {
  status: string
  mesh_path: string
  pid: number
  blender_bin: string
}

export const openInBlender = (playId: string) =>
  apiFetch<BlenderOpenResponse>('/api/blender/open', {
    method: 'POST',
    body: JSON.stringify({ play_id: playId }),
  })

export const pollQueryStatus = (token: string) =>
  apiFetch<{ status: string; result?: QueryResponse; progress?: Record<string, unknown> }>(
    '/api/query/' + token + '/status',
  )
