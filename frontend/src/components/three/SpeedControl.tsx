'use client'

type Props = {
  speed: number
  onSpeedChange: (speed: number) => void
}

const SPEEDS = [0.25, 0.5, 1]

export default function SpeedControl({ speed, onSpeedChange }: Props) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-xs text-gray-500 mr-1">Speed</span>
      {SPEEDS.map((s) => (
        <button
          key={s}
          onClick={() => onSpeedChange(s)}
          className={
            'px-2 py-1 rounded text-xs transition-colors ' +
            (speed === s
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700')
          }
        >
          {s}x
        </button>
      ))}
    </div>
  )
}
