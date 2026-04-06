'use client'
import { useState, useCallback, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import QueryBar from '@/components/ui/QueryBar'
import ReportPanel from '@/components/ui/ReportPanel'
import MetricsPanel from '@/components/ui/MetricsPanel'
import StatcastPanel from '@/components/ui/StatcastPanel'
import CameraPresets from '@/components/three/CameraPresets'
import SpeedControl from '@/components/three/SpeedControl'
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
const FieldGeometry = dynamic(() => import('@/components/three/FieldGeometry'), { ssr: false })
const JointSelector = dynamic(() => import('@/components/three/JointSelector'), { ssr: false })
const MetricGraph = dynamic(() => import('@/components/three/MetricGraph'), { ssr: false })

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
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [selectedJoint, setSelectedJoint] = useState<number | null>(null)

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
    <div className="flex flex-col gap-5">
      <h1 className="text-2xl font-bold tracking-tight">Mechanics Diagnostic</h1>

      {/* Query bar */}
      <QueryBar
        onSubmit={handleSubmit}
        loading={loading}
        error={error}
        progressText={progressText}
      />

      {/* Results — War Room 3-column layout */}
      {result && (
        <div className="grid grid-cols-1 xl:grid-cols-[40%_35%_25%] gap-4">
          {/* Col 1: 3D Viewer */}
          <div className="flex flex-col gap-3">
            {/* Controls row */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <CameraPresets
                onPresetChange={setCameraPreset}
                currentPreset={cameraPreset}
              />
              <button
                onClick={() => setShowGhost(!showGhost)}
                className={
                  'px-3 py-1 rounded text-xs font-medium transition-colors ' +
                  (showGhost
                    ? 'bg-blue-600 text-white'
                    : 'bg-[#1a1a1a] text-gray-400 hover:bg-[#2a2a2a] border border-[#2a2a2a]')
                }
              >
                Ghost
              </button>
            </div>

            {/* 3D canvas */}
            <div className="w-full aspect-[4/3] rounded-lg overflow-hidden border border-[#2a2a2a]">
              {glbUrl ? (
                <MoundScene cameraPreset={cameraPreset as any}>
                  <FieldGeometry />
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
                <div className="w-full h-full bg-[#111] flex items-center justify-center text-gray-600 text-sm">
                  3D Pitcher Viewer
                </div>
              )}
            </div>

            {/* Timeline + speed */}
            {totalFrames > 0 && (
              <div className="flex flex-col gap-2">
                <TimelineScrubber
                  currentFrame={currentFrame}
                  totalFrames={totalFrames}
                  phaseMarkers={phaseMarkers}
                  onFrameChange={setCurrentFrame}
                />
                <div className="flex justify-end">
                  <SpeedControl speed={playbackSpeed} onSpeedChange={setPlaybackSpeed} />
                </div>
              </div>
            )}

            {/* Joint metric graph */}
            {selectedJoint != null && (
              <MetricGraph
                jointLabel={`Joint ${selectedJoint}`}
                anglesA={[]}
                anglesB={[]}
                phaseMarkers={phaseMarkers}
                currentFrame={currentFrame}
                labelA="Group A"
                labelB="Group B"
              />
            )}
          </div>

          {/* Col 2: Diagnostic Report */}
          <div className="flex flex-col gap-4">
            <ReportPanel report={result.report} />
          </div>

          {/* Col 3: Statcast + Metrics */}
          <div className="flex flex-col gap-4">
            <StatcastPanel
              groupA={result.statcast.group_a}
              groupB={result.statcast.group_b}
              labelA="Group A"
              labelB="Group B"
            />

            {/* Query details */}
            <div className="bg-[#111] rounded-lg border border-[#2a2a2a] p-4">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Query</h3>
              <div className="font-data text-xs text-gray-400">
                Pitches: {result.query.pitches_used?.join(', ') ?? 'N/A'}
              </div>
              <div className="font-data text-xs text-gray-400 mt-1">
                Mode: {String(result.query.parsed?.comparison_mode ?? 'N/A')}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="text-center text-gray-600 py-20 text-sm">
          Ask a question about pitcher mechanics to get started.
        </div>
      )}
    </div>
  )
}
