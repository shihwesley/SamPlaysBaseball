'use client'
import { use } from 'react'
import useSWR from 'swr'
import Link from 'next/link'
import { getPitcher, getOutings } from '@/lib/api'

export default function PitcherPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const { data: pitcher, isLoading: pLoading } = useSWR(['pitcher', id], () => getPitcher(id))
  const { data: outings, isLoading: oLoading } = useSWR(['outings', id], () => getOutings(id))

  if (pLoading || oLoading) {
    return <div className="text-gray-400">Loading...</div>
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-1">{pitcher?.name ?? 'Unknown Pitcher'}</h1>
      <p className="text-gray-400 mb-6">
        {pitcher?.team} &middot; {pitcher?.throwingArm === 'L' ? 'Left-handed' : 'Right-handed'}
      </p>

      <h2 className="text-xl font-semibold mb-4">Outings</h2>
      {(!outings || outings.length === 0) ? (
        <p className="text-gray-500">No outings recorded yet.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {outings.map((o) => (
            <Link key={o.id} href={`/pitcher/${id}/outing/${o.id}`}>
              <div className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 cursor-pointer border border-gray-700 hover:border-gray-600 transition-colors">
                <div className="font-semibold">{o.date}</div>
                <div className="text-gray-400 text-sm mt-1">
                  {o.pitchCount} pitches &middot; {o.pitchTypes.join(', ')}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
