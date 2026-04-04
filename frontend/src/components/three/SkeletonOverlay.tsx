'use client'
import { Line } from '@react-three/drei'

type Props = {
  joints: number[][]
  color?: string
  jointRadius?: number
}

// MHR70 bone pairs: spine, left arm, right arm, legs
const BONE_PAIRS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // spine
  [0, 5], [5, 6], [6, 7], [7, 8],       // left arm
  [0, 9], [9, 10], [10, 11], [11, 12],  // right arm
  [0, 13], [0, 14],                      // hips to ankles (simplified)
  [13, 15], [14, 16],                    // ankles to feet
]

export default function SkeletonOverlay({ joints, color = '#00ff88', jointRadius = 0.03 }: Props) {
  if (!joints || joints.length === 0) return null

  return (
    <group>
      {/* Joint spheres */}
      {joints.map((j, i) => (
        <mesh key={i} position={[j[0], j[1], j[2]]}>
          <sphereGeometry args={[jointRadius, 8, 8]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}

      {/* Bone lines */}
      {BONE_PAIRS.map(([a, b], i) => {
        if (!joints[a] || !joints[b]) return null
        const points: [number, number, number][] = [
          [joints[a][0], joints[a][1], joints[a][2]],
          [joints[b][0], joints[b][1], joints[b][2]],
        ]
        return (
          <Line key={i} points={points} color={color} lineWidth={1} />
        )
      })}
    </group>
  )
}
