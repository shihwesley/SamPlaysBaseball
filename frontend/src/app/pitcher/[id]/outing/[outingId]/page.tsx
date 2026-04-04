'use client'
import { use, useState } from 'react'
import dynamic from 'next/dynamic'
import AngleTimeSeries from '@/components/charts/AngleTimeSeries'
import FatigueCurve from '@/components/charts/FatigueCurve'
import ArmSlotHistory from '@/components/charts/ArmSlotHistory'
import KineticChain from '@/components/charts/KineticChain'

const MoundScene = dynamic(() => import('@/components/three/MoundScene'), { ssr: false })
const PitcherMesh = dynamic(() => import('@/components/three/PitcherMesh'), { ssr: false })
const TimelineScrubber = dynamic(() => import('@/components/three/TimelineScrubber'), { ssr: false })

const TABS = ['3D View', 'Angles', 'Fatigue', 'Arm Slot', 'Kinetics'] as const
type Tab = (typeof TABS)[number]

export default function OutingPage({
  params,
}: {
  params: Promise<{ id: string; outingId: string }>
}) {
  const { id, outingId } = use(params)
  const [tab, setTab] = useState<Tab>('3D View')
  const [frame, setFrame] = useState(0)
  const totalFrames = 60

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Outing Analysis</h1>

      <div className="flex gap-2 mb-6 flex-wrap">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={
              'px-4 py-2 rounded text-sm transition-colors ' +
              (tab === t
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700')
            }
          >
            {t}
          </button>
        ))}
      </div>

      {tab === '3D View' && (
        <div className="flex flex-col gap-3">
          <div className="h-[560px] rounded-lg overflow-hidden">
            <MoundScene>
              <PitcherMesh
                glbUrl={`/api/export/glb/demo-${outingId}`}
                currentFrame={frame}
                totalFrames={totalFrames}
              />
            </MoundScene>
          </div>
          <TimelineScrubber
            currentFrame={frame}
            totalFrames={totalFrames}
            onFrameChange={setFrame}
          />
        </div>
      )}

      {tab === 'Angles' && <AngleTimeSeries pitchId={`demo-${outingId}`} />}
      {tab === 'Fatigue' && <FatigueCurve pitcherId={id} outingId={outingId} />}
      {tab === 'Arm Slot' && <ArmSlotHistory pitcherId={id} />}
      {tab === 'Kinetics' && <KineticChain pitchId={`demo-${outingId}`} />}
    </div>
  )
}
