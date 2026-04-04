import Link from 'next/link'
import type { Pitcher } from '@/lib/api'

export default function PitcherCard({ pitcher }: { pitcher: Pitcher }) {
  return (
    <Link href={'/pitcher/' + pitcher.id}>
      <div className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 cursor-pointer transition-colors border border-gray-700 hover:border-gray-600">
        <div className="font-semibold text-lg text-gray-100">{pitcher.name}</div>
        <div className="text-gray-400 text-sm mt-0.5">{pitcher.team}</div>
        <div className="text-gray-500 text-xs mt-2">
          {pitcher.throwingArm === 'L' ? 'Left-handed' : 'Right-handed'}
        </div>
      </div>
    </Link>
  )
}
