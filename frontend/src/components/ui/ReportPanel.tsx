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
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Diagnostic Report</h2>
        <div className="flex items-center gap-3">
          <span
            className={
              'text-xs px-2 py-1 rounded border ' +
              (CONFIDENCE_COLORS[report.confidence] ?? CONFIDENCE_COLORS.low)
            }
          >
            {report.confidence} confidence
          </span>
          <span className="text-xs text-gray-500">
            {report.pitches_analyzed} pitches analyzed
          </span>
        </div>
      </div>

      {/* Narrative */}
      <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-line">
        {report.narrative}
      </div>

      {/* Risk flags */}
      {report.risk_flags.length > 0 && (
        <div className="flex flex-col gap-1">
          <h3 className="text-sm font-medium text-red-400">Risk Flags</h3>
          <ul className="flex flex-col gap-1">
            {report.risk_flags.map((flag, i) => (
              <li key={i} className="text-sm text-red-300 bg-red-950 rounded px-3 py-1.5 border border-red-900">
                {flag}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="flex flex-col gap-1">
          <h3 className="text-sm font-medium text-blue-400">Recommendations</h3>
          <ol className="list-decimal list-inside flex flex-col gap-1">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="text-sm text-gray-300">
                {rec}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
