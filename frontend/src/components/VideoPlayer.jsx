import React, { useEffect, useRef } from 'react'
import { C, radius, shadow } from '../theme'

export default function VideoPlayer({ chunkId, startTime, onClose }) {
  const videoRef = useRef(null)

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }, [chunkId])

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      onClick={onClose}
    >
      <div
        style={{ position: 'relative', maxWidth: '92vw' }}
        onClick={e => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: -40,
            right: 0,
            background: 'none',
            border: `1px solid ${C.border}`,
            borderRadius: radius.sm,
            color: C.textSecondary,
            padding: '4px 12px',
            fontSize: 13,
          }}
        >
          ✕ Close
        </button>
        <video
          ref={videoRef}
          src={`/api/preview/${chunkId}`}
          controls
          autoPlay
          style={{
            maxWidth: '92vw',
            maxHeight: '85vh',
            borderRadius: radius.md,
            boxShadow: shadow.modal,
            display: 'block',
          }}
        />
      </div>
    </div>
  )
}
