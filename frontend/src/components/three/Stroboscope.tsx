'use client'
import MoundScene from './MoundScene'
import PitcherMesh from './PitcherMesh'

type Props = {
  glbUrls: string[]
  phaseFrame?: number
  opacity?: number
}

export default function Stroboscope({ glbUrls, phaseFrame = 0, opacity = 0.12 }: Props) {
  if (!glbUrls || glbUrls.length === 0) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-gray-900 rounded-lg text-gray-500">
        No pitches selected
      </div>
    )
  }

  return (
    <div style={{ height: 500 }}>
      <MoundScene>
        {glbUrls.map((url, i) => (
          <PitcherMesh
            key={i}
            glbUrl={url}
            currentFrame={phaseFrame}
            totalFrames={60}
            // Each mesh rendered at low opacity stacked — color varies slightly by index
            color={`hsl(${200 + i * 15}, 70%, ${50 + i * 3}%)`}
          />
        ))}
      </MoundScene>
    </div>
  )
}
