'use client'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

type Props = {
  pitchId: string
}

export default function TippingImportance({ pitchId: _pitchId }: Props) {
  // Placeholder — production would fetch from /api/analysis/{pitchId}/tipping
  const features = [
    'glove_pos_x',
    'arm_slot',
    'hip_rotation',
    'shoulder_tilt',
    'stride_length',
    'elbow_angle',
    'wrist_angle',
    'trunk_lean',
    'head_pos',
    'balance_point',
  ]
  const importance = [0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06]

  return (
    <Plot
      data={[
        {
          x: importance,
          y: features,
          type: 'bar',
          orientation: 'h',
          marker: {
            color: importance.map((v) => (v > 0.12 ? '#ef4444' : v > 0.08 ? '#f97316' : '#6b7280')),
          },
          name: 'SHAP Importance',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Pitch Tipping — Feature Importance',
        xaxis: { title: 'SHAP Value', gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { gridcolor: '#1f2937', color: '#9ca3af' },
        showlegend: false,
        margin: { l: 120 },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
