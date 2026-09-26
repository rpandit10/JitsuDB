import React from 'react'
import { C, radius } from '../theme'

const POSITIONS = [
  'mount', 'side_control', 'half_guard', 'closed_guard', 'open_guard',
  'back_control', 'turtle', 'north_south', 'knee_on_belly', 'standing', '50_50',
]
const INTENTS   = ['attack', 'defense', 'sweep', 'pass', 'transition', 'control', 'takedown']
const CONTEXTS  = ['top', 'bottom', 'neutral', 'standing']
const GI_OPTS   = ['gi', 'nogi', 'both']

function SectionLabel({ children }) {
  return (
    <p style={{
      fontSize: 10,
      fontWeight: 700,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      color: C.textMuted,
      marginBottom: 8,
      marginTop: 20,
    }}>
      {children}
    </p>
  )
}

function FilterButton({ label, active, color, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'inline-block',
        margin: '0 4px 6px 0',
        padding: '4px 10px',
        borderRadius: radius.pill,
        fontSize: 12,
        fontWeight: active ? 600 : 400,
        border: `1px solid ${active ? color : C.border}`,
        background: active ? color : 'transparent',
        color: active ? C.white : C.textSecondary,
        transition: 'all 0.15s',
        lineHeight: 1.4,
      }}
    >
      {label}
    </button>
  )
}

export default function FilterPanel({ filters, onChange }) {
  function toggle(key, value) {
    const current = filters[key] || []
    const updated = current.includes(value)
      ? current.filter(v => v !== value)
      : [...current, value]
    onChange({ ...filters, [key]: updated })
  }

  function setExclusive(key, value) {
    onChange({ ...filters, [key]: value === filters[key] ? null : value })
  }

  const hasAny =
    (filters.positions?.length || 0) +
    (filters.intents?.length   || 0) +
    (filters.context ? 1 : 0) +
    (filters.gi_nogi ? 1 : 0) > 0

  return (
    <div style={{
      background: C.surface,
      borderRadius: 10,
      padding: '16px 14px',
      border: `1px solid ${C.border}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: C.textPrimary }}>Filters</span>
        {hasAny && (
          <button
            onClick={() => onChange({ positions: [], intents: [], techniques: [], context: null, gi_nogi: null, instructor: null })}
            style={{ background: 'none', border: 'none', fontSize: 11, color: C.textMuted, padding: 0 }}
          >
            Clear all
          </button>
        )}
      </div>

      <SectionLabel>Position</SectionLabel>
      <div>
        {POSITIONS.map(p => (
          <FilterButton
            key={p}
            label={p.replace(/_/g, ' ')}
            active={(filters.positions || []).includes(p)}
            color={C.blue}
            onClick={() => toggle('positions', p)}
          />
        ))}
      </div>

      <SectionLabel>Intent</SectionLabel>
      <div>
        {INTENTS.map(i => (
          <FilterButton
            key={i}
            label={i}
            active={(filters.intents || []).includes(i)}
            color={C.purple}
            onClick={() => toggle('intents', i)}
          />
        ))}
      </div>

      <SectionLabel>Context</SectionLabel>
      <div>
        {CONTEXTS.map(c => (
          <FilterButton
            key={c}
            label={c}
            active={filters.context === c}
            color={C.brown}
            onClick={() => setExclusive('context', c)}
          />
        ))}
      </div>

      <SectionLabel>Gi / No-Gi</SectionLabel>
      <div>
        {GI_OPTS.map(g => (
          <FilterButton
            key={g}
            label={g}
            active={filters.gi_nogi === g}
            color={C.blue}
            onClick={() => setExclusive('gi_nogi', g)}
          />
        ))}
      </div>
    </div>
  )
}
