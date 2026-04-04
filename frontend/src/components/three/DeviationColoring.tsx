'use client'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Mesh, MeshBasicMaterial } from 'three'

type Props = {
  joints: number[][]
  zScores: number[]
  threshold?: number
}

function getColor(z: number): string {
  if (z < 1) return '#00ff88'
  if (z < 2) return '#ffff00'
  return '#ff2200'
}

type JointSphereProps = {
  position: [number, number, number]
  color: string
  pulse: boolean
}

function JointSphere({ position, color, pulse }: JointSphereProps) {
  const meshRef = useRef<Mesh>(null)

  useFrame(({ clock }) => {
    if (!pulse || !meshRef.current) return
    const mat = meshRef.current.material as MeshBasicMaterial
    mat.opacity = 0.5 + 0.5 * Math.sin(clock.elapsedTime * 4)
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.04, 8, 8]} />
      <meshBasicMaterial color={color} transparent={pulse} opacity={1} />
    </mesh>
  )
}

export default function DeviationColoring({ joints, zScores, threshold = 2.5 }: Props) {
  if (!joints || joints.length === 0) return null

  return (
    <group>
      {joints.map((j, i) => {
        const z = zScores[i] ?? 0
        return (
          <JointSphere
            key={i}
            position={[j[0], j[1], j[2]]}
            color={getColor(z)}
            pulse={z > threshold}
          />
        )
      })}
    </group>
  )
}
