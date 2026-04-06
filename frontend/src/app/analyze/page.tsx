'use client'
import { useState, useCallback, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import QueryBar from '@/components/ui/QueryBar'
import ReportPanel from '@/components/ui/ReportPanel'
import MetricsPanel from '@/components/ui/MetricsPanel'
import StatcastPanel from '@/components/ui/StatcastPanel'
import CameraPresets from '@/components/three/CameraPresets'
import {
  submitQuery,
  pollQueryStatus,
  API_BASE,
  type QueryResponse,
  type ProgressResponse,
} from '@/lib/api'

const MoundScene = dynamic(() => import('@/components/three/MoundScene'), { ssr: false })
const PitcherMesh = dynamic(() => import('@/components/three/PitcherMesh'), { ssr: false })
const GhostOverlay = dynamic(() => import('@/components/three/GhostOverlay'), { ssr: false })
const TimelineScrubber = dynamic(() => import('@/components/three/TimelineScrubber'), { ssr: false })

export default function AnalyzePage() {
  // Query state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progressText, setProgressText] = useState<string | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // 3D viewer state
  const [currentFrame, setCurrentFrame] = useState(0)
  const [cameraPreset, setCameraPreset] = useState('catcher')
  const [showGhost, setShowGhost] = useState(true)

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const handleSubmit = useCallback(async (query: string) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setProgressText(null)
    setCurrentFrame(0)

    try {
      const res = await submitQuery(query)

      if (res.status === 'complete') {
        setResult(res as QueryResponse)
        setLoading(false)
        return
      }

      // Processing — start polling
      const progress = res as ProgressResponse
      const n = progress.pitches_needing_inference.length
      setProgressText(
        `Running 3D reconstruction on ${n} pitch${n > 1 ? 'es' : ''}... ` +
        (progress.eta_seconds ? `(~${Math.ceil(progress.eta_seconds / 60)} min)` : ''),
      )

      pollRef.current = setInterval(async () => {
        try {
          const status = await pollQueryStatus(progress.token)
          if (status.status === 'complete' && status.result) {
            if (pollRef.current) clearInterval(pollRef.current)
            pollRef.current = null
            setResult(status.result)
            setProgressText(null)
            setLoading(false)
          }
        } catch {
          // Poll failures are transient — keep trying
        }
      }, 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query failed')
      setLoading(false)
    }
  }, [])

  const glbUrl = result?.viewer.glb_url ? API_BASE + result.viewer.glb_url : null
  const phaseMarkers = result?.viewer.phase_markers ?? {}
  const totalFrames = result?.viewer.total_frames ?? 0

  // Extract mechanical diffs from the query response
  const diffs = result?.query?.parsed
    ? [] // Diffs come from the comparison object — for now use what's in report
    : []

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-bold">Mechanics Diagnostic</h1>

      {/* Query bar */}
      <QueryBar
        onSubmit={handleSubmit}
        loading={loading}
        error={error}
        progressText={progressText}
      />

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left: 3D viewer (2 cols) */}
          <div className="xl:col-span-2 flex flex-col gap-3">
            {/* Camera presets + ghost toggle */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <CameraPresets
                onPresetChange={setCameraPreset}
                currentPreset={cameraPreset}
              />
              <button
                onClick={() => setShowGhost(!showGhost)}
                className={
                  'px-3 py-1 rounded text-sm transition-colors ' +
                  (showGhost
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700')
                }
              >
                Ghost Overlay
              </button>
            </div>

            {/* 3D canvas */}
            <div className="w-full aspect-video rounded-lg overflow-hidden border border-gray-800">
              {glbUrl ? (
                <MoundScene cameraPreset={cameraPreset as any}>
                  <PitcherMesh
                    glbUrl={glbUrl}
                    currentFrame={currentFrame}
                    totalFrames={totalFrames}
                  />
                  {showGhost && (
                    <GhostOverlay
                      glbUrl={glbUrl}
                      currentFrame={currentFrame}
                      opacity={0.3}
                    />
                  )}
                </MoundScene>
              ) : (
                <div className="w-full h-full bg-gray-900 flex items-center justify-center text-gray-600">
                  3D viewer
                </div>
              )}
            </div>

            {/* Timeline */}
            {totalFrames > 0 && (
              <TimelineScrubber
                currentFrame={currentFrame}
                totalFrames={totalFrames}
                phaseMarkers={phaseMarkers}
                onFrameChange={setCurrentFrame}
              />
            )}

            {/* Report below viewer on large screens */}
            <ReportPanel report={result.report} />
          </div>

          {/* Right: Stats + Metrics (1 col) */}
          <div className="flex flex-col gap-6">
            <StatcastPanel
              groupA={result.statcast.group_a}
              groupB={result.statcast.group_b}
              labelA="Group A"
              labelB="Group B"
            />
            {/* Metrics panel with mechanical diffs from the comparison */}
            {result.report.risk_flags.length > 0 && (
              <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
                <h2 className="text-lg font-semibold mb-2">Query Details</h2>
                <div className="text-xs text-gray-500">
                  Pitches: {result.query.pitches_used?.join(', ') ?? 'N/A'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Mode: {String(result.query.parsed?.comparison_mode ?? 'N/A')}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="text-center text-gray-600 py-20">
          Ask a question about pitcher mechanics to get started.
        </div>
      )}
    </div>
  )
}
