'use client'
import { useMemo } from 'react'
import * as THREE from 'three'

/**
 * Low-poly baseball field: grass outfield, dirt infield, baselines, bases.
 * Not photorealistic — just enough context to orient the viewer.
 *
 * MLB regulation dimensions (in meters):
 *   Base paths: 27.43m (90ft)
 *   Pitching rubber to home: 18.44m (60ft 6in)
 *   Infield dirt radius: ~28m from pitcher's mound
 */
export default function FieldGeometry() {
  const baselineGeo = useMemo(() => {
    // Two foul lines from home plate
    const pts1 = [new THREE.Vector3(0, 0.005, 0), new THREE.Vector3(40, 0.005, 40)]
    const pts2 = [new THREE.Vector3(0, 0.005, 0), new THREE.Vector3(-40, 0.005, 40)]
    return { pts1, pts2 }
  }, [])

  return (
    <group>
      {/* Grass outfield — large flat plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 20]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#1a5c1a" />
      </mesh>

      {/* Dirt infield — diamond-shaped */}
      <mesh rotation={[-Math.PI / 2, 0, Math.PI / 4]} position={[0, 0.001, 19.4]} receiveShadow>
        <planeGeometry args={[28, 28]} />
        <meshStandardMaterial color="#8B6914" opacity={0.8} transparent />
      </mesh>

      {/* Home plate — white pentagon approximated as small box */}
      <mesh position={[0, 0.01, 0]}>
        <boxGeometry args={[0.43, 0.02, 0.43]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* 1st base */}
      <mesh position={[19.4, 0.01, 19.4]}>
        <boxGeometry args={[0.38, 0.02, 0.38]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* 2nd base */}
      <mesh position={[0, 0.01, 38.8]}>
        <boxGeometry args={[0.38, 0.02, 0.38]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* 3rd base */}
      <mesh position={[-19.4, 0.01, 19.4]}>
        <boxGeometry args={[0.38, 0.02, 0.38]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* Foul lines */}
      {[baselineGeo.pts1, baselineGeo.pts2].map((pts, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array(pts.flatMap((p) => [p.x, p.y, p.z])), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ffffff" opacity={0.4} transparent />
        </line>
      ))}

      {/* Pitching rubber */}
      <mesh position={[0, 0.27, 18.44]}>
        <boxGeometry args={[0.61, 0.02, 0.15]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
    </group>
  )
}
