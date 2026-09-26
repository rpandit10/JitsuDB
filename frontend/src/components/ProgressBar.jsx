import React from 'react'
import { C, radius } from '../theme'

export default function ProgressBar({ value = 0, label, accentColor }) {
  const pct   = Math.round(value * 100)
  const color = accentColor || C.blue
  const grad  = accentColor
    ? `linear-gradient(90deg, ${accentColor}cc, ${accentColor})`
    : `linear-gradient(90deg, ${C.blue}, ${C.purple})`

  return (
    <div title={label}>
      <div style={{ background: C.surfaceHigh, height: 7, width: '100%', borderRadius: radius.pill }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: grad,
          borderRadius: radius.pill,
          transition: 'width 0.4s ease',
        }} />
      </div>
      <span style={{ fontSize: 11, color: C.textMuted, marginTop: 3, display: 'inline-block' }}>
        {pct}%
      </span>
    </div>
  )
}
