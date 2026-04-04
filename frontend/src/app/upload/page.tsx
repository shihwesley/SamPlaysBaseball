'use client'
import { useState, useRef } from 'react'
import Link from 'next/link'
import { uploadVideo } from '@/lib/api'

const PITCH_TYPES = ['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'KC', 'FS']

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [pitcherId, setPitcherId] = useState('')
  const [pitchType, setPitchType] = useState('FF')
  const [status, setStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file || !pitcherId.trim()) return
    setStatus('uploading')
    setErrorMsg('')
    try {
      const result = await uploadVideo(file, pitcherId, { pitchType })
      setJobId(result.jobId)
      setStatus('done')
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unknown error')
      setStatus('error')
    }
  }

  return (
    <div className="max-w-lg">
      <h1 className="text-3xl font-bold mb-6">Upload Video</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Video File</label>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm file:bg-gray-700 file:text-gray-200 file:border-0 file:rounded file:px-3 file:py-1 file:mr-3 cursor-pointer"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">Pitcher ID</label>
          <input
            value={pitcherId}
            onChange={(e) => setPitcherId(e.target.value)}
            placeholder="e.g. pitcher-001"
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">Pitch Type</label>
          <select
            value={pitchType}
            onChange={(e) => setPitchType(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          >
            {PITCH_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          disabled={status === 'uploading' || !file || !pitcherId.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed rounded px-4 py-2 font-semibold transition-colors"
        >
          {status === 'uploading' ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {status === 'done' && (
        <div className="mt-4 p-4 bg-green-950 border border-green-800 rounded text-green-300">
          Upload complete. Job ID: <code className="text-green-200">{jobId}</code>.{' '}
          <Link href="/" className="underline hover:text-green-100">
            View pitchers
          </Link>
        </div>
      )}
      {status === 'error' && (
        <div className="mt-4 p-4 bg-red-950 border border-red-800 rounded text-red-300">
          Upload failed: {errorMsg || 'Check server logs.'}
        </div>
      )}
    </div>
  )
}
