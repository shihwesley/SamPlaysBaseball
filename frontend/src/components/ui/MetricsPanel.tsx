'use client'

type Diff = {
  name: string
  early: number
  late: number
  change: number
  unit: string
}

type Props = {
  diffs: Diff[]
  labelA: string
  labelB: string
}

export default function MetricsPanel({ diffs, labelA, labelB }: Props) {
  // Sort by absolute change descending
  const sorted = [...diffs].sort((a, b) => Math.abs(b.change) - Math.abs(a.change))
  const maxAbs = Math.max(...sorted.map((d) => Math.abs(d.change)), 1)

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
      <h2 className="text-lg font-semibold mb-3">Mechanical Differences</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-xs border-b border-gray-800">
            <th className="text-left py-2 font-medium">Feature</th>
            <th className="text-right py-2 font-medium">{labelA}</th>
            <th className="text-right py-2 font-medium">{labelB}</th>
            <th className="text-right py-2 font-medium">Change</th>
            <th className="py-2 w-24" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((d) => {
            const pct = (Math.abs(d.change) / maxAbs) * 100
            const isNeg = d.change < 0
            return (
              <tr key={d.name} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-2 text-gray-300">{d.name}</td>
                <td className="py-2 text-right tabular-nums text-gray-400">
                  {d.early.toFixed(1)} {d.unit}
                </td>
                <td className="py-2 text-right tabular-nums text-gray-400">
                  {d.late.toFixed(1)} {d.unit}
                </td>
                <td className="py-2 text-right tabular-nums font-medium">
                  <span className={isNeg ? 'text-red-400' : 'text-blue-400'}>
                    {isNeg ? '' : '+'}{d.change.toFixed(1)} {d.unit}
                  </span>
                </td>
                <td className="py-2 px-2">
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={'h-full rounded-full ' + (isNeg ? 'bg-red-500' : 'bg-blue-500')}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
