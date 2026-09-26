import React from 'react'
import { C } from '../theme'

const CIRCUMFERENCE = 2 * Math.PI * 36  // r=36

export default function SearchLoader() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 24,
      padding: '48px 0 32px',
    }}>
      {/* Circular progress ring */}
      <div style={{ position: 'relative', width: 80, height: 80 }}>
        <svg
          width={80}
          height={80}
          viewBox="0 0 80 80"
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Track */}
          <circle
            cx={40} cy={40} r={36}
            fill="none"
            stroke={C.surfaceHigh}
            strokeWidth={5}
          />
          {/* Spinning arc */}
          <circle
            cx={40} cy={40} r={36}
            fill="none"
            stroke={C.purple}
            strokeWidth={5}
            strokeLinecap="round"
            strokeDasharray={`${CIRCUMFERENCE * 0.25} ${CIRCUMFERENCE * 0.75}`}
            style={{
              transformOrigin: '40px 40px',
              animation: 'jitsu-spin 0.9s linear infinite',
            }}
          />
        </svg>

        {/* Inner label */}
        <span style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: '0.1em',
          color: C.purpleLight,
          textTransform: 'uppercase',
        }}>
          BJJ
        </span>
      </div>

      <p style={{
        fontSize: 12,
        color: C.textMuted,
        letterSpacing: '0.06em',
        margin: 0,
      }}>
        Searching…
      </p>

      <style>{`
        @keyframes jitsu-spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
