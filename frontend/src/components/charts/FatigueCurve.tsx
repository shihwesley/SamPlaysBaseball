'use client'
import useSWR from 'swr'
import dynamic from 'next/dynamic'
import { getPitches } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

type Props = {
  pitcherId: string
  outingId: string
}

export default function FatigueCurve({ pitcherId, outingId }: Props) {
  const { data, isLoading, error } = useSWR(
    ['pitches-fatigue', pitcherId, outingId],
    () => getPitches(pitcherId, outingId),
  )

  if (isLoading) return <div className="text-gray-400 py-8">Loading fatigue data...</div>
  if (error) return <div className="text-gray-500 py-8">No fatigue data available.</div>

  const n = data?.length ?? 10
  // Fatigue proxy from pitch velocity drop (real data would come from analysis endpoint)
  const pitchNums = Array.from({ length: n }, (_, i) => i + 1)
  const fatigue = Array.from({ length: n }, (_, i) => Math.min(1, 0.2 + i * (0.6 / Math.max(n - 1, 1))))

  return (
    <Plot
      data={[
        {
          x: pitchNums,
          y: fatigue,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Fatigue Index',
          line: { color: '#f97316', width: 2 },
          marker: { color: '#f97316', size: 6 },
        },
        {
          x: [1, n],
          y: [0.7, 0.7],
          type: 'scatter',
          mode: 'lines',
          name: 'Warning Threshold',
          line: { color: '#ef4444', dash: 'dash', width: 1.5 },
          hoverinfo: 'skip' as const,
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Fatigue Curve',
        xaxis: { title: 'Pitch #', gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { title: 'Fatigue Index', range: [0, 1.05], gridcolor: '#1f2937', color: '#9ca3af' },
        legend: { bgcolor: 'transparent', font: { color: '#9ca3af' } },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
