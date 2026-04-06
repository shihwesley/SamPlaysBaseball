'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const NAV = [
  { href: '/', label: 'Pitchers' },
  { href: '/analyze', label: 'Analyze' },
  { href: '/compare', label: 'Compare' },
  { href: '/upload', label: 'Upload' },
]

export default function Sidebar() {
  const path = usePathname()
  return (
    <aside className="w-[200px] bg-[#111] flex flex-col min-h-screen p-4 shrink-0 border-r border-[#2a2a2a]">
      <div className="text-lg font-bold mb-8 text-blue-400 font-data tracking-tight">SPB</div>
      <nav className="flex flex-col gap-1">
        {NAV.map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className={
              'px-3 py-2 rounded text-sm transition-colors ' +
              (path === n.href
                ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                : 'text-gray-400 hover:bg-[#1a1a1a] border border-transparent')
            }
          >
            {n.label}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
