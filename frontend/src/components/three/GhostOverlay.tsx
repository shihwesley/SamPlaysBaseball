'use client'
import { useEffect } from 'react'
import { useGLTF } from '@react-three/drei'
import { clone } from 'three/examples/jsm/utils/SkeletonUtils.js'
import type { Mesh, MeshStandardMaterial } from 'three'

type Props = {
  glbUrl: string
  currentFrame: number
  opacity?: number
}

export default function GhostOverlay({ glbUrl, currentFrame, opacity = 0.3 }: Props) {
  const { scene } = useGLTF(glbUrl)
  const clonedScene = clone(scene)

  useEffect(() => {
    clonedScene.traverse((obj) => {
      const mesh = obj as Mesh
      if (!mesh.isMesh) return

      // Apply transparent material
      const mat = mesh.material as MeshStandardMaterial
      if (mat && 'opacity' in mat) {
        mat.transparent = true
        mat.opacity = opacity
      }

      // Apply morph target frame
      if (mesh.morphTargetInfluences && mesh.morphTargetInfluences.length > 0) {
        for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
          mesh.morphTargetInfluences[i] = 0
        }
        const morphIdx = currentFrame - 1
        if (morphIdx >= 0 && morphIdx < mesh.morphTargetInfluences.length) {
          mesh.morphTargetInfluences[morphIdx] = 1
        }
      }
    })
  }, [clonedScene, currentFrame, opacity])

  return <primitive object={clonedScene} />
}
