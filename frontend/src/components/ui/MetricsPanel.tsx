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
    <div className="bg-[#111] rounded-lg border border-[#2a2a2a] p-4">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Mechanical Differences</h2>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-500 text-[11px] border-b border-[#2a2a2a]">
            <th className="text-left py-1.5 font-medium">Feature</th>
            <th className="text-right py-1.5 font-medium">{labelA}</th>
            <th className="text-right py-1.5 font-medium">{labelB}</th>
            <th className="text-right py-1.5 font-medium">Delta</th>
            <th className="py-1.5 w-20" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((d) => {
            const pct = (Math.abs(d.change) / maxAbs) * 100
            const isNeg = d.change < 0
            return (
              <tr key={d.name} className="border-b border-[#1a1a1a] hover:bg-[#1a1a1a]">
                <td className="py-2 text-gray-300">{d.name}</td>
                <td className="py-2 text-right font-data tabular-nums text-gray-400">
                  {d.early.toFixed(1)}
                </td>
                <td className="py-2 text-right font-data tabular-nums text-gray-400">
                  {d.late.toFixed(1)}
                </td>
                <td className="py-2 text-right font-data tabular-nums font-medium">
                  <span className={isNeg ? 'text-red-400' : 'text-blue-400'}>
                    {isNeg ? '' : '+'}{d.change.toFixed(1)}
                  </span>
                </td>
                <td className="py-2 px-1">
                  <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
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
