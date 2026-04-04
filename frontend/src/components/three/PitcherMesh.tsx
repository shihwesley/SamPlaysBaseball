'use client'
import { useEffect, useRef } from 'react'
import { useGLTF } from '@react-three/drei'
import type { Mesh } from 'three'

type Props = {
  glbUrl: string
  currentFrame: number
  totalFrames: number
  color?: string
}

export default function PitcherMesh({ glbUrl, currentFrame, totalFrames, color = '#e8b88a' }: Props) {
  const { scene } = useGLTF(glbUrl)
  const meshRef = useRef<Mesh>(null)

  useEffect(() => {
    scene.traverse((obj) => {
      const mesh = obj as Mesh
      if (!mesh.isMesh) return
      if (!mesh.morphTargetInfluences || mesh.morphTargetInfluences.length === 0) return

      // Zero out all morph targets then activate the current frame
      for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
        mesh.morphTargetInfluences[i] = 0
      }
      const morphIdx = currentFrame - 1 // frame 0 is base mesh
      if (morphIdx >= 0 && morphIdx < mesh.morphTargetInfluences.length) {
        mesh.morphTargetInfluences[morphIdx] = 1
      }
    })
  }, [scene, currentFrame])

  // Check if any mesh has morph targets
  let hasMorphTargets = false
  scene.traverse((obj) => {
    const mesh = obj as Mesh
    if (mesh.isMesh && mesh.morphTargetInfluences && mesh.morphTargetInfluences.length > 0) {
      hasMorphTargets = true
    }
  })

  if (!hasMorphTargets) {
    return (
      <mesh ref={meshRef} position={[0, 0.9, 0]}>
        <boxGeometry args={[0.5, 1.8, 0.3]} />
        <meshStandardMaterial color={color} />
      </mesh>
    )
  }

  return <primitive object={scene} />
}
