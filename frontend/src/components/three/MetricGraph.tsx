'use client'
import { useMemo } from 'react'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

type Props = {
  /** Joint label to display */
  jointLabel: string
  /** Angle time series for pitch A (degrees, one per frame) */
  anglesA: number[]
  /** Angle time series for pitch B */
  anglesB: number[]
  /** Angular velocity time series for pitch A (deg/s) */
  velocityA?: number[]
  /** Angular velocity time series for pitch B */
  velocityB?: number[]
  /** Phase markers as frame indices */
  phaseMarkers?: Record<string, number>
  /** Current playback frame (for cursor line) */
  currentFrame?: number
  labelA?: string
  labelB?: string
}

export default function MetricGraph({
  jointLabel,
  anglesA,
  anglesB,
  velocityA,
  velocityB,
  phaseMarkers,
  currentFrame,
  labelA = 'Pitch A',
  labelB = 'Pitch B',
}: Props) {
  const traces = useMemo(() => {
    const t: Plotly.Data[] = [
      {
        y: anglesA,
        name: `${labelA} angle`,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#3b82f6', width: 2 },
        yaxis: 'y',
      },
      {
        y: anglesB,
        name: `${labelB} angle`,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#f97316', width: 2 },
        yaxis: 'y',
      },
    ]

    if (velocityA?.length) {
      t.push({
        y: velocityA,
        name: `${labelA} velocity`,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#3b82f6', width: 1, dash: 'dot' },
        yaxis: 'y2',
      })
    }
    if (velocityB?.length) {
      t.push({
        y: velocityB,
        name: `${labelB} velocity`,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#f97316', width: 1, dash: 'dot' },
        yaxis: 'y2',
      })
    }
    return t
  }, [anglesA, anglesB, velocityA, velocityB, labelA, labelB])

  const shapes = useMemo(() => {
    const s: Partial<Plotly.Shape>[] = []
    // Phase marker vertical lines
    if (phaseMarkers) {
      for (const [label, frame] of Object.entries(phaseMarkers)) {
        s.push({
          type: 'line',
          x0: frame, x1: frame, y0: 0, y1: 1,
          yref: 'paper',
          line: { color: '#6b7280', width: 1, dash: 'dash' },
        })
      }
    }
    // Current frame cursor
    if (currentFrame != null) {
      s.push({
        type: 'line',
        x0: currentFrame, x1: currentFrame, y0: 0, y1: 1,
        yref: 'paper',
        line: { color: '#ef4444', width: 2 },
      })
    }
    return s
  }, [phaseMarkers, currentFrame])

  const layout: Partial<Plotly.Layout> = {
    title: { text: jointLabel, font: { color: '#e5e7eb', size: 14 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9ca3af', size: 11 },
    margin: { t: 36, r: 50, b: 36, l: 50 },
    xaxis: {
      title: { text: 'Frame' },
      gridcolor: '#1f2937',
      zerolinecolor: '#374151',
    },
    yaxis: {
      title: { text: 'Angle (deg)' },
      gridcolor: '#1f2937',
      zerolinecolor: '#374151',
    },
    yaxis2: {
      title: { text: 'Velocity (deg/s)' },
      overlaying: 'y',
      side: 'right',
      gridcolor: '#1f2937',
      showgrid: false,
    },
    legend: {
      orientation: 'h',
      y: -0.2,
      font: { size: 10 },
    },
    shapes,
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
      <Plot
        data={traces}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 250 }}
      />
    </div>
  )
}
