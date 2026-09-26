import React, { useState } from 'react'
import { C, radius } from '../theme'

export default function SearchBar({ onSearch, detectedFilters = {} }) {
  const [query, setQuery]           = useState('')
  const [activeChips, setActiveChips] = useState({})

  function handleSubmit(e) {
    e.preventDefault()
    onSearch(query, activeChips)
  }

  function toggleChip(type, label) {
    setActiveChips(prev => {
      const current = prev[type] || []
      const next = current.includes(label)
        ? current.filter(l => l !== label)
        : [...current, label]
      const updated = { ...prev, [type]: next }
      onSearch(query, updated)
      return updated
    })
  }

  const hasChips = Object.values(detectedFilters).some(v => v?.length > 0)

  return (
    <div style={{ marginBottom: 20 }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 10 }}>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search techniques, positions, escapes..."
          style={{
            flex: 1,
            padding: '10px 16px',
            fontSize: 15,
            background: C.surface,
            border: `1px solid ${C.border}`,
            borderRadius: radius.md,
            color: C.textPrimary,
            outline: 'none',
          }}
        />
        <button
          type="submit"
          style={{
            padding: '10px 24px',
            fontSize: 14,
            fontWeight: 600,
            background: C.blue,
            color: C.white,
            border: 'none',
            borderRadius: radius.md,
            letterSpacing: '0.02em',
          }}
        >
          Search
        </button>
      </form>

      {hasChips && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 11, color: C.textMuted, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Detected
          </span>
          {Object.entries(detectedFilters).map(([type, labels]) =>
            labels.map(label => {
              const active = activeChips[type]?.includes(label)
              return (
                <button
                  key={`${type}-${label}`}
                  type="button"
                  onClick={() => toggleChip(type, label)}
                  style={{
                    padding: '3px 10px',
                    borderRadius: radius.pill,
                    fontSize: 12,
                    fontWeight: active ? 600 : 400,
                    border: `1px solid ${active ? C.purple : C.border}`,
                    background: active ? C.purple : C.surfaceHigh,
                    color: active ? C.white : C.textSecondary,
                    transition: 'all 0.15s',
                  }}
                >
                  {label}
                </button>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}
