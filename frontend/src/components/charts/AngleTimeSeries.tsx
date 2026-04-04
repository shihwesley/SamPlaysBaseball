'use client'
import useSWR from 'swr'
import dynamic from 'next/dynamic'
import { getAnalysis } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const DARK: any = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

export default function AngleTimeSeries({ pitchId }: { pitchId: string }) {
  const { data, isLoading, error } = useSWR(['analysis', pitchId], () => getAnalysis(pitchId))

  if (isLoading) return <div className="text-gray-400 py-8">Loading angle data...</div>
  if (error) return <div className="text-gray-500 py-8">No angle data available for this pitch.</div>

  const angles = data?.angles ?? {}
  const traces = Object.entries(angles).map(([joint, values]) => ({
    x: values.map((_: number, i: number) => i),
    y: values,
    name: joint,
    type: 'scatter' as const,
    mode: 'lines' as const,
  }))

  if (traces.length === 0) {
    return <div className="text-gray-500 py-8">No joint angle data in analysis result.</div>
  }

  return (
    <Plot
      data={traces}
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      layout={{
        ...DARK,
        title: 'Joint Angles Over Time',
        xaxis: { title: 'Frame', gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { title: 'Degrees', gridcolor: '#1f2937', color: '#9ca3af' },
        legend: { bgcolor: 'transparent', font: { color: '#9ca3af' } },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
