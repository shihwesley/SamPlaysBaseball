'use client'
import { useState, useRef, type FormEvent } from 'react'

type Props = {
  onSubmit: (query: string) => void
  loading?: boolean
  error?: string | null
  progressText?: string | null
}

const EXAMPLES = [
  "Compare Ohtani's 1st inning fastballs to his 6th inning fastballs",
  "Why did his slider command fall off after the 4th?",
  "Show me his cutter vs fastball delivery",
]

export default function QueryBar({ onSubmit, loading, error, progressText }: Props) {
  const [text, setText] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const q = text.trim()
    if (!q || loading) return
    onSubmit(q)
  }

  function useExample(ex: string) {
    setText(ex)
    inputRef.current?.focus()
  }

  return (
    <div className="flex flex-col gap-2">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask about pitcher mechanics..."
          disabled={loading}
          className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 text-sm
            focus:outline-none focus:border-blue-500 disabled:opacity-50
            placeholder:text-gray-500"
        />
        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700
            disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors shrink-0"
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {/* Progress indicator */}
      {progressText && (
        <div className="text-sm text-blue-400 flex items-center gap-2">
          <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          {progressText}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-sm text-red-400 bg-red-950 border border-red-800 rounded px-3 py-2">
          {error}
        </div>
      )}

      {/* Example queries */}
      {!loading && !text && (
        <div className="flex flex-wrap gap-2 mt-1">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => useExample(ex)}
              className="text-xs text-gray-400 bg-gray-800 hover:bg-gray-700 rounded-full
                px-3 py-1 transition-colors border border-gray-700"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
