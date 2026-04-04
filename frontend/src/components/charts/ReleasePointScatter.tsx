'use client'
import useSWR from 'swr'
import dynamic from 'next/dynamic'
import { getPitches } from '@/lib/api'
import type { Pitch } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

type Props = {
  pitcherId: string
  outingId?: string
  pitchType?: string
}

export default function ReleasePointScatter({ pitcherId, outingId = 'latest', pitchType }: Props) {
  const { data, isLoading, error } = useSWR(
    ['pitches', pitcherId, outingId],
    () => getPitches(pitcherId, outingId),
  )

  if (isLoading) return <div className="text-gray-400 py-8">Loading release points...</div>
  if (error) return <div className="text-gray-500 py-8">No release point data available.</div>

  const pitches = (data ?? []).filter((p: Pitch) => !pitchType || p.type === pitchType)

  // In production these coordinates would come from the analysis endpoint
  // Using pitch index as placeholder until real release coordinates are available
  const byType: Record<string, { x: number[]; y: number[] }> = {}
  pitches.forEach((p: Pitch, i: number) => {
    if (!byType[p.type]) byType[p.type] = { x: [], y: [] }
    byType[p.type].x.push((i % 5) * 0.1 - 0.2)
    byType[p.type].y.push(Math.floor(i / 5) * 0.05 + 1.6)
  })

  const traces = Object.entries(byType).map(([type, coords]) => ({
    x: coords.x,
    y: coords.y,
    name: type,
    mode: 'markers' as const,
    type: 'scatter' as const,
    marker: { size: 8 },
  }))

  return (
    <Plot
      data={traces.length > 0 ? traces : [{ x: [], y: [], type: 'scatter', mode: 'markers', name: 'No data' }]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Release Points',
        xaxis: { title: 'Horizontal (m)', gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { title: 'Vertical (m)', gridcolor: '#1f2937', color: '#9ca3af' },
        legend: { bgcolor: 'transparent', font: { color: '#9ca3af' } },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
