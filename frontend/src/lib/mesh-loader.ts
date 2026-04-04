export type MeshMetadata = {
  pitchType: string
  pitcherId: string
  frameTimestamps: number[]
  phaseMarkers: Record<string, number>
}

export async function loadGLB(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load GLB from ${url}: ${res.statusText}`)
  return res.arrayBuffer()
}

export async function loadGLBFromPath(path: string): Promise<ArrayBuffer> {
  return loadGLB(path)
}

/**
 * Extract extras metadata from GLB JSON chunk.
 * GLB layout: 12-byte header, then chunk 0 (JSON) at offset 12.
 * Chunk format: 4-byte length, 4-byte type (0x4E4F534A = JSON), then data.
 */
export function parseMeshMetadata(buffer: ArrayBuffer): MeshMetadata {
  const defaultMeta: MeshMetadata = {
    pitchType: '',
    pitcherId: '',
    frameTimestamps: [],
    phaseMarkers: {},
  }
  try {
    const view = new DataView(buffer)
    const magic = view.getUint32(0, true)
    if (magic !== 0x46546C67) return defaultMeta // not glTF

    const jsonChunkLength = view.getUint32(12, true)
    const jsonChunkType = view.getUint32(16, true)
    if (jsonChunkType !== 0x4E4F534A) return defaultMeta // not JSON

    const jsonBytes = new Uint8Array(buffer, 20, jsonChunkLength)
    const jsonStr = new TextDecoder().decode(jsonBytes)
    const gltf = JSON.parse(jsonStr)
    const extras = gltf.extras ?? {}

    return {
      pitchType: String(extras.pitch_type ?? ''),
      pitcherId: String(extras.pitcher_id ?? ''),
      frameTimestamps: Array.isArray(extras.frame_timestamps) ? extras.frame_timestamps : [],
      phaseMarkers:
        typeof extras.phase_markers === 'object' && extras.phase_markers !== null
          ? extras.phase_markers
          : {},
    }
  } catch {
    return defaultMeta
  }
}
