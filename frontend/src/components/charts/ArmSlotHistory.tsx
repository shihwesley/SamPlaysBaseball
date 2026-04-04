'use client'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111111',
  font: { color: '#e5e7eb' },
}

type Props = {
  pitcherId: string
}

export default function ArmSlotHistory({ pitcherId: _pitcherId }: Props) {
  // Placeholder data — production would fetch from /api/pitchers/{id}/arm-slot-history
  const dates = ['2024-03-01', '2024-03-08', '2024-03-15', '2024-03-22', '2024-03-29']
  const slots = [52.3, 51.8, 53.1, 52.5, 51.2]
  const fatigueLevels = [0.2, 0.3, 0.25, 0.45, 0.65]

  return (
    <Plot
      data={[
        {
          x: dates,
          y: slots,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Arm Slot (°)',
          line: { color: '#60a5fa', width: 2 },
          marker: {
            color: fatigueLevels.map((f) =>
              f > 0.6 ? '#ef4444' : f > 0.4 ? '#f97316' : '#22c55e',
            ),
            size: 8,
          },
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: 'Arm Slot History',
        xaxis: { title: 'Date', gridcolor: '#1f2937', color: '#9ca3af' },
        yaxis: { title: 'Arm Slot Angle (°)', gridcolor: '#1f2937', color: '#9ca3af' },
        showlegend: false,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any}
      style={{ width: '100%', height: 400 }}
      config={{ displayModeBar: false }}
    />
  )
}
