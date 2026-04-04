'use client'

type Props = {
  onPresetChange: (preset: string) => void
  currentPreset?: string
}

const PRESETS = [
  { id: 'catcher', label: 'Catcher View' },
  { id: 'first-base', label: '1B Side' },
  { id: 'third-base', label: '3B Side' },
  { id: 'overhead', label: 'Overhead' },
  { id: 'behind-pitcher', label: 'Behind Pitcher' },
]

export default function CameraPresets({ onPresetChange, currentPreset = 'catcher' }: Props) {
  return (
    <div className="flex flex-row gap-1 flex-wrap">
      {PRESETS.map((p) => (
        <button
          key={p.id}
          onClick={() => onPresetChange(p.id)}
          className={
            'px-3 py-1 rounded text-sm transition-colors ' +
            (currentPreset === p.id
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-600')
          }
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
