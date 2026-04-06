'use client'
import type { DiagnosticReport } from '@/lib/api'

type Props = {
  report: DiagnosticReport
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: 'bg-green-900 text-green-300 border-green-700',
  moderate: 'bg-yellow-900 text-yellow-300 border-yellow-700',
  low: 'bg-red-900 text-red-300 border-red-700',
}

export default function ReportPanel({ report }: Props) {
  return (
    <div className="bg-[#111] rounded-lg border border-[#2a2a2a] p-5 flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Diagnostic Report</h2>
        <div className="flex items-center gap-3">
          <span
            className={
              'text-xs font-data px-2 py-0.5 rounded border ' +
              (CONFIDENCE_COLORS[report.confidence] ?? CONFIDENCE_COLORS.low)
            }
          >
            {report.confidence}
          </span>
          <span className="text-xs font-data text-gray-500">
            {report.pitches_analyzed} pitches
          </span>
        </div>
      </div>

      {/* Narrative — generous line height for readability */}
      <div className="text-[13px] text-gray-300 leading-[1.7] whitespace-pre-line">
        {report.narrative}
      </div>

      {/* Risk flags */}
      {report.risk_flags.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-red-400">Risk Flags</h3>
          <ul className="flex flex-col gap-1.5">
            {report.risk_flags.map((flag, i) => (
              <li key={i} className="text-xs font-data text-red-300 bg-red-950/50 rounded px-3 py-2 border border-red-900/50">
                {flag}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-blue-400">Recommendations</h3>
          <ol className="list-decimal list-inside flex flex-col gap-1.5">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="text-[13px] text-gray-300 leading-relaxed">
                {rec}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
