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
    <aside className="w-52 bg-gray-900 flex flex-col min-h-screen p-4 shrink-0 border-r border-gray-800">
      <div className="text-xl font-bold mb-8 text-blue-400 tracking-tight">SPB</div>
      <nav className="flex flex-col gap-1">
        {NAV.map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className={
              'px-3 py-2 rounded text-sm transition-colors ' +
              (path === n.href
                ? 'bg-blue-700 text-white'
                : 'text-gray-300 hover:bg-gray-800')
            }
          >
            {n.label}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
