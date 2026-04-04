'use client'
import { useState } from 'react'
import dynamic from 'next/dynamic'
import RadarDeviation from '@/components/charts/RadarDeviation'

const SplitSync = dynamic(() => import('@/components/three/SplitSync'), { ssr: false })

export default function ComparePage() {
  const [pitchId1, setPitchId1] = useState('')
  const [pitchId2, setPitchId2] = useState('')
  const [showGhost, setShowGhost] = useState(false)
  const ready = pitchId1.trim() !== '' && pitchId2.trim() !== ''

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Compare Pitches</h1>

      <div className="flex gap-4 mb-6 flex-wrap">
        <input
          value={pitchId1}
          onChange={(e) => setPitchId1(e.target.value)}
          placeholder="Pitch ID 1"
          className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm flex-1 min-w-40 focus:outline-none focus:border-blue-500"
        />
        <input
          value={pitchId2}
          onChange={(e) => setPitchId2(e.target.value)}
          placeholder="Pitch ID 2"
          className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm flex-1 min-w-40 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={() => setShowGhost(!showGhost)}
          className={
            'px-4 py-2 rounded text-sm transition-colors ' +
            (showGhost ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700')
          }
        >
          Ghost Overlay
        </button>
      </div>

      {ready ? (
        <div className="flex flex-col gap-6">
          <SplitSync
            glbUrl1={`/api/export/glb/${pitchId1}`}
            glbUrl2={`/api/export/glb/${pitchId2}`}
            label1={`Pitch ${pitchId1}`}
            label2={`Pitch ${pitchId2}`}
          />
          <RadarDeviation pitchId={pitchId1} />
        </div>
      ) : (
        <div className="text-gray-500 text-sm">Enter two pitch IDs above to compare.</div>
      )}
    </div>
  )
}
