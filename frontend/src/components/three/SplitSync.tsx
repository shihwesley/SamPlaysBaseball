'use client'
import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import MoundScene from './MoundScene'
import PitcherMesh from './PitcherMesh'
import TimelineScrubber from './TimelineScrubber'

type Props = {
  glbUrl1: string
  glbUrl2: string
  label1?: string
  label2?: string
}

export default function SplitSync({
  glbUrl1,
  glbUrl2,
  label1 = 'Pitch 1',
  label2 = 'Pitch 2',
}: Props) {
  const [currentFrame, setCurrentFrame] = useState(0)
  const totalFrames = 60

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-row gap-2">
        {/* Left viewport */}
        <div className="flex flex-col gap-1 flex-1">
          <span className="text-sm text-gray-400 text-center">{label1}</span>
          <div style={{ height: 500 }}>
            <MoundScene>
              <PitcherMesh glbUrl={glbUrl1} currentFrame={currentFrame} totalFrames={totalFrames} />
            </MoundScene>
          </div>
        </div>

        {/* Right viewport */}
        <div className="flex flex-col gap-1 flex-1">
          <span className="text-sm text-gray-400 text-center">{label2}</span>
          <div style={{ height: 500 }}>
            <MoundScene>
              <PitcherMesh glbUrl={glbUrl2} currentFrame={currentFrame} totalFrames={totalFrames} />
            </MoundScene>
          </div>
        </div>
      </div>

      {/* Shared timeline */}
      <TimelineScrubber
        currentFrame={currentFrame}
        totalFrames={totalFrames}
        onFrameChange={setCurrentFrame}
      />
    </div>
  )
}
