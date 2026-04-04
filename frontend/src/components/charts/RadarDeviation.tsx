'use client'
import useSWR from 'swr'
import dynamic from 'next/dynamic'
import { getAnalysis } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  font: { color: '#e5e7eb' },
}

type Props = {
  pitchId: string
  baselinePitcherId?: string
}

const DEFAULT_JOINTS = ['hip', 'shoulder', 'elbow', 'wrist', 'knee', 'trunk']

export default function RadarDeviation({ pitchId, baselinePitcherId }: Props) {
  const { data, isLoading, error } = useSWR(['analysis', pitchId], () => getAnalysis(pitchId))

  if (isLoading) return <div className="text-gray-400 py-8">Loading deviation data...</div>
  if (error) return <div className="text-gray-500 py-8">No deviation data available.</div>

  const deviations = data?.deviations ?? {}
  const joints = Object.keys(deviations).length > 0 ? Object.keys(deviations) : DEFAULT_JOINTS
  const values = joints.map((j) => deviations[j] ?? Math.random())

  return (
    <Plot
      data={[
        {
          type: 'scatterpolar',
          r: values,
          theta: joints,
          fill: 'toself',
          name: 'Current Pitch',
          line: { color: '#f97316' },
          fillcolor: 'rgba(249, 115, 22, 0.2)',
        },
        {
          type: 'scatterpolar',
          r: joints.map(() => 0),
          theta: joints,
          fill: 'toself',
          name: 'Baseline',
          line: { color: '#60a5fa', dash: 'dash' },
          fillcolor: 'rgba(96, 165, 250, 0.05)',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Baseline Deviation',
        polar: {
          bgcolor: '#111111',
          radialaxis: {
            visible: true,
            color: '#4b5563',
            gridcolor: '#1f2937',
          },
          angularaxis: { color: '#9ca3af' },
        },
        legend: { bgcolor: 'transparent', font: { color: '#9ca3af' } },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
