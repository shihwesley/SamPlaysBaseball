import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Sidebar from '@/components/ui/Sidebar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SamPlaysBaseball',
  description: 'Baseball pitching motion analysis',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={inter.className + ' bg-gray-950 text-gray-100 flex min-h-screen'}
        style={{ background: '#030712' }}
      >
        <Sidebar />
        <main className="flex-1 p-6 overflow-auto min-w-0">{children}</main>
      </body>
    </html>
  )
}
