import React, { useState, useEffect, useCallback } from 'react'
import { fetchAvailableFolders, seedFolder } from '../utils/api'
import { C, radius, shadow } from '../theme'

function FolderRow({ folder, selected, onClick }) {
  const isSelected = selected === folder.path
  return (
    <button
      onClick={() => onClick(folder.path)}
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        width: '100%',
        padding: '10px 16px',
        background: isSelected ? `${C.purple}22` : 'transparent',
        border: 'none',
        borderBottom: `1px solid ${C.border}`,
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'background 0.1s',
      }}
    >
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: C.textPrimary }}>
          {folder.name}
        </div>
        <div style={{ fontSize: 11, color: C.textMuted, marginTop: 2 }}>
          {folder.relative_path}
        </div>
      </div>
      <span style={{
        fontSize: 11,
        fontWeight: 700,
        color: C.textMuted,
        background: C.surfaceHigh,
        padding: '2px 8px',
        borderRadius: radius.pill,
        whiteSpace: 'nowrap',
        marginLeft: 12,
        flexShrink: 0,
      }}>
        {folder.video_count} video{folder.video_count !== 1 ? 's' : ''}
      </span>
    </button>
  )
}

export default function AddInstructional() {
  const [folders, setFolders] = useState([])
  const [scanning, setScanning] = useState(false)
  const [selected, setSelected] = useState(null)
  const [customPath, setCustomPath] = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const [autoStart, setAutoStart] = useState(true)
  const [seeding, setSeeding] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const scan = useCallback(async () => {
    setScanning(true)
    setResult(null)
    setError(null)
    try {
      const data = await fetchAvailableFolders()
      setFolders(data)
    } catch (e) {
      setError('Failed to scan library: ' + e.message)
    } finally {
      setScanning(false)
    }
  }, [])

  useEffect(() => { scan() }, [scan])

  const activePath = useCustom ? customPath.trim() : selected

  async function handleSeed() {
    if (!activePath) return
    setSeeding(true)
    setResult(null)
    setError(null)
    try {
      const res = await seedFolder(activePath, autoStart)
      setResult(res)
      setSelected(null)
      setCustomPath('')
      scan()
    } catch (e) {
      setError(e.message)
    } finally {
      setSeeding(false)
    }
  }

  return (
    <div style={{ maxWidth: 720 }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: C.textPrimary, margin: '0 0 4px' }}>
          Add Instructional
        </h2>
        <p style={{ fontSize: 13, color: C.textMuted, margin: 0 }}>
          Select a folder from your library or paste a custom path to seed videos into the pipeline.
        </p>
      </div>

      {/* Folder list */}
      <div style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: radius.md,
        overflow: 'hidden',
        boxShadow: shadow.card,
        marginBottom: 16,
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 16px',
          borderBottom: `1px solid ${C.border}`,
          background: C.surfaceHigh,
        }}>
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.textMuted }}>
            Available Folders
          </span>
          <button
            onClick={scan}
            disabled={scanning}
            style={{
              fontSize: 11,
              padding: '3px 10px',
              borderRadius: radius.sm,
              border: `1px solid ${C.border}`,
              background: 'transparent',
              color: scanning ? C.textMuted : C.textSecondary,
              cursor: scanning ? 'default' : 'pointer',
            }}
          >
            {scanning ? 'Scanning…' : 'Rescan'}
          </button>
        </div>

        {scanning ? (
          <div style={{ padding: '24px 0', textAlign: 'center', color: C.textMuted, fontSize: 13 }}>
            Scanning library…
          </div>
        ) : folders.length === 0 ? (
          <div style={{ padding: '24px 0', textAlign: 'center', color: C.textMuted, fontSize: 13 }}>
            No un-indexed folders found in library.
          </div>
        ) : (
          <div style={{ maxHeight: 320, overflowY: 'auto' }}>
            {folders.map(f => (
              <FolderRow
                key={f.path}
                folder={f}
                selected={!useCustom ? selected : null}
                onClick={(path) => { setSelected(path); setUseCustom(false) }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Custom path */}
      <div style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: radius.md,
        padding: '14px 16px',
        marginBottom: 16,
        boxShadow: shadow.card,
      }}>
        <label style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.textMuted, display: 'block', marginBottom: 8 }}>
          Or paste a custom path
        </label>
        <input
          type="text"
          value={customPath}
          onChange={e => { setCustomPath(e.target.value); if (e.target.value.trim()) setUseCustom(true) }}
          onFocus={() => { if (customPath.trim()) setUseCustom(true) }}
          placeholder="/path/to/Instructional Folder"
          style={{
            width: '100%',
            boxSizing: 'border-box',
            background: C.surfaceHigh,
            border: `1px solid ${useCustom && customPath.trim() ? C.purple : C.border}`,
            borderRadius: radius.sm,
            padding: '8px 12px',
            fontSize: 13,
            color: C.textPrimary,
            outline: 'none',
          }}
        />
      </div>

      {/* Options + action */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: C.textSecondary, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={autoStart}
            onChange={e => setAutoStart(e.target.checked)}
            style={{ accentColor: C.purple }}
          />
          Auto-start pipeline after seeding
        </label>

        <button
          onClick={handleSeed}
          disabled={!activePath || seeding}
          style={{
            marginLeft: 'auto',
            padding: '8px 22px',
            fontSize: 14,
            fontWeight: 600,
            borderRadius: radius.sm,
            border: 'none',
            background: !activePath || seeding ? C.surfaceHigh : C.purple,
            color: !activePath || seeding ? C.textMuted : C.white,
            cursor: !activePath || seeding ? 'default' : 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {seeding ? 'Seeding…' : 'Seed & Enqueue'}
        </button>
      </div>

      {/* Active selection indicator */}
      {activePath && !seeding && !result && (
        <div style={{
          fontSize: 12,
          color: C.textMuted,
          padding: '8px 12px',
          background: `${C.purple}11`,
          border: `1px solid ${C.purple}33`,
          borderRadius: radius.sm,
          marginBottom: 12,
        }}>
          Selected: <span style={{ color: C.purpleLight, fontWeight: 500 }}>{activePath}</span>
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{
          padding: '14px 16px',
          background: result.seeded > 0 ? `${C.success}15` : C.surfaceHigh,
          border: `1px solid ${result.seeded > 0 ? C.success : C.border}`,
          borderRadius: radius.md,
          fontSize: 13,
          color: result.seeded > 0 ? C.success : C.textMuted,
        }}>
          {result.message}
          {result.seeded > 0 && autoStart && (
            <span style={{ color: C.textMuted }}> Pipeline started — monitor in Processing Queue.</span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '12px 16px',
          background: `${C.danger}15`,
          border: `1px solid ${C.danger}55`,
          borderRadius: radius.md,
          fontSize: 13,
          color: C.danger,
        }}>
          {error}
        </div>
      )}
    </div>
  )
}
