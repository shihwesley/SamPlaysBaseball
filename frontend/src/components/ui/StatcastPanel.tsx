'use client'

type StatGroup = {
  avg_velo?: number
  avg_spin?: number
  whiff_pct?: number
  zone_pct?: number
}

type Props = {
  groupA: StatGroup
  groupB: StatGroup
  labelA: string
  labelB: string
}

const STAT_CONFIG: { key: keyof StatGroup; label: string; unit: string; decimals: number }[] = [
  { key: 'avg_velo', label: 'Avg Velocity', unit: 'mph', decimals: 1 },
  { key: 'avg_spin', label: 'Avg Spin Rate', unit: 'rpm', decimals: 0 },
  { key: 'whiff_pct', label: 'Whiff %', unit: '%', decimals: 1 },
  { key: 'zone_pct', label: 'Zone %', unit: '%', decimals: 1 },
]

export default function StatcastPanel({ groupA, groupB, labelA, labelB }: Props) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
      <h2 className="text-lg font-semibold mb-3">Statcast</h2>
      <div className="grid grid-cols-2 gap-4">
        {/* Column headers */}
        <div className="text-xs text-gray-500 font-medium">{labelA}</div>
        <div className="text-xs text-gray-500 font-medium text-right">{labelB}</div>

        {STAT_CONFIG.map(({ key, label, unit, decimals }) => {
          const a = groupA[key]
          const b = groupB[key]
          if (a == null && b == null) return null

          const diff = a != null && b != null ? b - a : null
          const diffColor =
            diff == null ? '' :
            key === 'avg_velo' && diff < -1 ? 'text-red-400' :
            key === 'whiff_pct' && diff < -5 ? 'text-red-400' :
            key === 'zone_pct' && diff < -5 ? 'text-red-400' :
            'text-gray-500'

          return (
            <div key={key} className="contents">
              <div className="flex flex-col">
                <span className="text-xs text-gray-500">{label}</span>
                <span className="text-lg font-medium tabular-nums">
                  {a != null ? a.toFixed(decimals) : '—'}
                  <span className="text-xs text-gray-500 ml-1">{unit}</span>
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className={'text-xs ' + diffColor}>
                  {diff != null ? `${diff >= 0 ? '+' : ''}${diff.toFixed(decimals)}` : ''}
                </span>
                <span className="text-lg font-medium tabular-nums">
                  {b != null ? b.toFixed(decimals) : '—'}
                  <span className="text-xs text-gray-500 ml-1">{unit}</span>
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
