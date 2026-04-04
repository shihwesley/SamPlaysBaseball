'use client'
import useSWR from 'swr'
import { getPitchers } from '@/lib/api'
import PitcherCard from '@/components/ui/PitcherCard'

export default function HomePage() {
  const { data: pitchers, error, isLoading } = useSWR('pitchers', getPitchers)

  if (isLoading) {
    return (
      <div>
        <h1 className="text-3xl font-bold mb-6">Pitchers</h1>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="bg-gray-800 rounded-lg p-4 h-24 animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h1 className="text-3xl font-bold mb-6">Pitchers</h1>
        <div className="text-red-400 bg-red-950 border border-red-800 rounded p-4">
          Failed to load pitchers. Check that the API server is running at{' '}
          <code className="text-red-300">http://localhost:8000</code>.
        </div>
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Pitchers</h1>
      {(!pitchers || pitchers.length === 0) ? (
        <p className="text-gray-400">No pitchers found. Upload a video to get started.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {pitchers.map((p) => (
            <PitcherCard key={p.id} pitcher={p} />
          ))}
        </div>
      )}
    </div>
  )
}
