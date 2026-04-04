'use client'
import { useState, useEffect, useRef } from 'react'

type Props = {
  currentFrame: number
  totalFrames: number
  phaseMarkers?: Record<string, number>
  onFrameChange: (frame: number) => void
  onPlay?: () => void
  onPause?: () => void
}

export default function TimelineScrubber({
  currentFrame,
  totalFrames,
  phaseMarkers,
  onFrameChange,
  onPlay,
  onPause,
}: Props) {
  const [isPlaying, setIsPlaying] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        onFrameChange((currentFrame + 1) % totalFrames)
      }, 100) // 10fps
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isPlaying, currentFrame, totalFrames, onFrameChange])

  function togglePlay() {
    const next = !isPlaying
    setIsPlaying(next)
    if (next) onPlay?.()
    else onPause?.()
  }

  const pct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0

  return (
    <div className="flex flex-col gap-2 px-4 py-2 bg-gray-900 rounded-lg">
      {/* Phase markers */}
      {phaseMarkers && Object.keys(phaseMarkers).length > 0 && (
        <div className="relative h-4">
          {Object.entries(phaseMarkers).map(([label, frame]) => {
            const x = totalFrames > 1 ? (frame / (totalFrames - 1)) * 100 : 0
            return (
              <span
                key={label}
                style={{ left: `${x}%` }}
                className="absolute -translate-x-1/2 text-xs text-blue-400 whitespace-nowrap"
              >
                {label}
              </span>
            )
          })}
        </div>
      )}

      {/* Scrubber row */}
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-8 h-8 flex items-center justify-center bg-blue-600 hover:bg-blue-700 rounded text-sm font-bold shrink-0"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, totalFrames - 1)}
          value={currentFrame}
          onChange={(e) => onFrameChange(Number(e.target.value))}
          className="flex-1 accent-blue-500"
        />
        <span className="text-xs text-gray-400 shrink-0 tabular-nums">
          {currentFrame}/{totalFrames - 1}
        </span>
      </div>
    </div>
  )
}
