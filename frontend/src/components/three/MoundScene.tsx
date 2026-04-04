'use client'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { ReactNode } from 'react'

type CameraPreset = 'catcher' | 'first-base' | 'third-base' | 'overhead' | 'behind-pitcher'

const CAMERA_POSITIONS: Record<CameraPreset, [number, number, number]> = {
  'catcher': [0, 1.5, 8],
  'first-base': [8, 1.5, -4],
  'third-base': [-8, 1.5, -4],
  'overhead': [0, 10, 0.1],
  'behind-pitcher': [0, 1.5, -8],
}

type Props = {
  children?: ReactNode
  cameraPreset?: CameraPreset
}

export default function MoundScene({ children, cameraPreset = 'catcher' }: Props) {
  const camPos = CAMERA_POSITIONS[cameraPreset] ?? CAMERA_POSITIONS['catcher']

  return (
    <Canvas
      style={{ background: '#0a0a0a', width: '100%', height: '100%' }}
      camera={{ position: camPos, fov: 50, near: 0.1, far: 200 }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.5} castShadow />
      <directionalLight position={[-5, 4, 5]} intensity={0.5} />

      {/* Pitcher's mound */}
      <mesh position={[0, 0, 0]} receiveShadow>
        <cylinderGeometry args={[2.74, 2.74, 0.267, 32]} />
        <meshStandardMaterial color="#8B6914" />
      </mesh>

      {/* Camera controls */}
      <OrbitControls target={[0, 0.5, 0]} enableDamping dampingFactor={0.05} />

      {children}
    </Canvas>
  )
}
