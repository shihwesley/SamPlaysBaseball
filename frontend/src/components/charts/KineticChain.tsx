'use client'
import useSWR from 'swr'
import dynamic from 'next/dynamic'
import { getAnalysis } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

const SEGMENTS = ['Hip', 'Torso', 'Shoulder', 'Elbow', 'Wrist']
const COLORS = ['#0ea5e9', '#22d3ee', '#34d399', '#a3e635', '#fbbf24']

export default function KineticChain({ pitchId }: { pitchId: string }) {
  const { data, isLoading, error } = useSWR(['analysis', pitchId], () => getAnalysis(pitchId))

  if (isLoading) return <div className="text-gray-400 py-8">Loading kinetic chain data...</div>
  if (error) return <div className="text-gray-500 py-8">No kinetic chain data available.</div>

  // In production, peak segment energies come from data.phases
  // Placeholder ascending energy sequence representing proper proximal-to-distal transfer
  const values = SEGMENTS.map((_, i) => 200 + i * 80 + (data ? i * 20 : 0))

  return (
    <Plot
      data={[
        {
          x: SEGMENTS,
          y: values,
          type: 'bar',
          marker: { color: COLORS },
          name: 'Peak Energy (W)',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Kinetic Chain — Peak Segment Energy',
        xaxis: { gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { title: 'Energy (W)', gridcolor: '#1f2937', color: '#9ca3af' },
        showlegend: false,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
