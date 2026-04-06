'use client'
import { useMemo } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'

type JointDef = { index: number; label: string; position: [number, number, number] }

type Props = {
  /** Joint positions as flat array of [x,y,z] triples from the current frame */
  joints: [number, number, number][]
  /** Which joint indices to show as selectable (biomechanically interesting) */
  selectableIndices?: number[]
  /** Currently selected joint index, or null */
  selectedIndex: number | null
  onSelect: (index: number) => void
  /** Joint names keyed by index */
  jointLabels?: Record<number, string>
}

// Default selectable joints — the ones an analyst cares about
const DEFAULT_SELECTABLE = [
  5,  // left_shoulder
  6,  // right_shoulder
  7,  // left_elbow
  8,  // right_elbow
  9,  // left_hip
  10, // right_hip
  11, // left_knee
  12, // right_knee
  13, // left_ankle
  14, // right_ankle
  41, // right_wrist
  62, // left_wrist
]

const DEFAULT_LABELS: Record<number, string> = {
  5: 'L Shoulder', 6: 'R Shoulder',
  7: 'L Elbow', 8: 'R Elbow',
  9: 'L Hip', 10: 'R Hip',
  11: 'L Knee', 12: 'R Knee',
  13: 'L Ankle', 14: 'R Ankle',
  41: 'R Wrist', 62: 'L Wrist',
}

const sphereGeo = new THREE.SphereGeometry(0.02, 8, 8)
const matDefault = new THREE.MeshBasicMaterial({ color: '#4a90d9', transparent: true, opacity: 0.5 })
const matHover = new THREE.MeshBasicMaterial({ color: '#60a5fa' })
const matSelected = new THREE.MeshBasicMaterial({ color: '#f59e0b' })

export default function JointSelector({
  joints,
  selectableIndices = DEFAULT_SELECTABLE,
  selectedIndex,
  onSelect,
  jointLabels = DEFAULT_LABELS,
}: Props) {
  const visibleJoints = useMemo<JointDef[]>(() => {
    return selectableIndices
      .filter((i) => i < joints.length)
      .map((i) => ({
        index: i,
        label: jointLabels[i] ?? `Joint ${i}`,
        position: joints[i],
      }))
  }, [joints, selectableIndices, jointLabels])

  return (
    <group>
      {visibleJoints.map((j) => (
        <mesh
          key={j.index}
          position={j.position}
          geometry={sphereGeo}
          material={j.index === selectedIndex ? matSelected : matDefault}
          onClick={(e: ThreeEvent<MouseEvent>) => {
            e.stopPropagation()
            onSelect(j.index)
          }}
          onPointerOver={(e: ThreeEvent<PointerEvent>) => {
            e.stopPropagation()
            document.body.style.cursor = 'pointer'
            const mesh = e.object as THREE.Mesh
            if (j.index !== selectedIndex) mesh.material = matHover
          }}
          onPointerOut={(e: ThreeEvent<PointerEvent>) => {
            document.body.style.cursor = 'auto'
            const mesh = e.object as THREE.Mesh
            if (j.index !== selectedIndex) mesh.material = matDefault
          }}
        />
      ))}
    </group>
  )
}
