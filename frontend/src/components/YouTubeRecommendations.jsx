import React, { useState, useEffect, useCallback } from 'react'
import { fetchRecommendations, confirmRecommendation, dismissRecommendation } from '../utils/api'
import { C, radius, shadow } from '../theme'

function formatDuration(secs) {
  if (!secs) return ''
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = Math.floor(secs % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

function formatViews(n) {
  if (!n) return ''
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M views`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K views`
  return `${n} views`
}

const CATEGORIES = [
  'Uncategorised',
  'Guard',
  'Guard Passing',
  'Submissions',
  'Leg Locks',
  'Takedowns',
  'Escapes',
  'Back Control',
  'Half Guard',
  'Mount',
  'Side Control',
  'Wrestling',
]

function ConfirmForm({ rec, onConfirm, onCancel }) {
  const [giNogi, setGiNogi]         = useState('nogi')
  const [instructor, setInstructor] = useState(rec.uploader || '')
  const [category, setCategory]     = useState('Uncategorised')
  const [busy, setBusy]             = useState(false)

  const inputStyle = {
    width: '100%',
    boxSizing: 'border-box',
    background: C.surfaceHigh,
    border: `1px solid ${C.border}`,
    borderRadius: radius.sm,
    padding: '6px 10px',
    fontSize: 12,
    color: C.textPrimary,
    outline: 'none',
  }
  const labelStyle = {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color: C.textMuted,
    display: 'block',
    marginBottom: 6,
  }

  async function handleConfirm() {
    setBusy(true)
    try {
      await onConfirm(rec, giNogi, instructor, category)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{
      marginTop: 10,
      padding: '12px 14px',
      background: C.surfaceHigh,
      borderRadius: radius.sm,
      border: `1px solid ${C.blue}44`,
    }}>
      <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Gi / No-Gi</label>
          <select value={giNogi} onChange={e => setGiNogi(e.target.value)} style={inputStyle}>
            <option value="nogi">No-Gi</option>
            <option value="gi">Gi</option>
            <option value="both">Both</option>
          </select>
        </div>
        <div style={{ flex: 2 }}>
          <label style={labelStyle}>Instructor</label>
          <input
            type="text"
            value={instructor}
            onChange={e => setInstructor(e.target.value)}
            style={inputStyle}
            placeholder="Instructor name"
          />
        </div>
      </div>
      <div style={{ marginBottom: 10 }}>
        <label style={labelStyle}>Category</label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {CATEGORIES.map(cat => {
            const active = category === cat
            return (
              <button
                key={cat}
                type="button"
                onClick={() => setCategory(cat)}
                style={{
                  padding: '3px 11px',
                  fontSize: 11,
                  fontWeight: active ? 700 : 500,
                  borderRadius: radius.pill,
                  border: `1px solid ${active ? C.blue : C.border}`,
                  background: active ? `${C.blue}22` : 'transparent',
                  color: active ? C.blue : C.textSecondary,
                  cursor: 'pointer',
                  transition: 'all 0.12s',
                }}
              >
                {cat}
              </button>
            )
          })}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleConfirm}
          disabled={busy}
          style={{
            padding: '6px 16px',
            fontSize: 12,
            fontWeight: 600,
            borderRadius: radius.sm,
            border: 'none',
            background: busy ? C.surfaceHigh : C.blue,
            color: busy ? C.textMuted : C.white,
            cursor: busy ? 'default' : 'pointer',
          }}
        >
          {busy ? 'Queuing…' : 'Download & Ingest'}
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: '6px 14px',
            fontSize: 12,
            fontWeight: 500,
            borderRadius: radius.sm,
            border: `1px solid ${C.border}`,
            background: 'transparent',
            color: C.textSecondary,
            cursor: 'pointer',
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

function RecommendationCard({ rec, onConfirmed, onDismissed }) {
  const [expanded, setExpanded]   = useState(false)
  const [dismissed, setDismissed] = useState(false)
  const [queued, setQueued]       = useState(false)

  if (dismissed || queued) return null

  async function handleConfirm(r, gi_nogi, instructor, primary_category) {
    await confirmRecommendation(r, gi_nogi, instructor, primary_category)
    setQueued(true)
    onConfirmed(r.yt_video_id)
  }

  async function handleDismiss() {
    await dismissRecommendation(rec)
    setDismissed(true)
    onDismissed(rec.yt_video_id)
  }

  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: radius.md,
      overflow: 'hidden',
      boxShadow: shadow.card,
    }}>
      <div style={{ display: 'flex', gap: 0 }}>
        {/* Thumbnail */}
        {rec.thumbnail_url ? (
          <div style={{ width: 140, flexShrink: 0, position: 'relative' }}>
            <img
              src={rec.thumbnail_url}
              alt=""
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              onError={e => { e.target.style.display = 'none' }}
            />
            {rec.duration_seconds > 0 && (
              <div style={{
                position: 'absolute',
                bottom: 4,
                right: 4,
                background: 'rgba(0,0,0,0.75)',
                color: '#fff',
                fontSize: 10,
                padding: '1px 5px',
                borderRadius: 3,
                fontVariantNumeric: 'tabular-nums',
              }}>
                {formatDuration(rec.duration_seconds)}
              </div>
            )}
          </div>
        ) : (
          <div style={{
            width: 140,
            flexShrink: 0,
            background: C.surfaceHigh,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: C.textMuted,
            fontSize: 11,
          }}>
            No thumbnail
          </div>
        )}

        {/* Info + actions */}
        <div style={{ flex: 1, padding: '12px 14px', minWidth: 0 }}>
          <a
            href={rec.url}
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: C.textPrimary,
              textDecoration: 'none',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              lineHeight: 1.4,
            }}
          >
            {rec.title}
          </a>

          <div style={{ marginTop: 4, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {rec.uploader && (
              <span style={{ fontSize: 11, color: C.textSecondary }}>{rec.uploader}</span>
            )}
            {rec.view_count > 0 && (
              <span style={{ fontSize: 11, color: C.textMuted }}>{formatViews(rec.view_count)}</span>
            )}
            {rec.source?.startsWith('search:') && (
              <span style={{
                fontSize: 10,
                color: C.textMuted,
                background: C.surfaceHigh,
                border: `1px solid ${C.border}`,
                borderRadius: radius.pill,
                padding: '1px 7px',
              }}>
                {rec.source.replace('search:', '')}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <button
              onClick={() => setExpanded(e => !e)}
              style={{
                padding: '5px 14px',
                fontSize: 12,
                fontWeight: 600,
                borderRadius: radius.sm,
                border: 'none',
                background: expanded ? `${C.blue}22` : C.blue,
                color: expanded ? C.blue : C.white,
                cursor: 'pointer',
              }}
            >
              {expanded ? 'Cancel' : 'Confirm'}
            </button>
            <button
              onClick={handleDismiss}
              style={{
                padding: '5px 14px',
                fontSize: 12,
                fontWeight: 500,
                borderRadius: radius.sm,
                border: `1px solid ${C.border}`,
                background: 'transparent',
                color: C.textMuted,
                cursor: 'pointer',
              }}
            >
              Dismiss
            </button>
          </div>

          {expanded && (
            <ConfirmForm
              rec={rec}
              onConfirm={handleConfirm}
              onCancel={() => setExpanded(false)}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default function YouTubeRecommendations() {
  const [recs, setRecs]           = useState([])
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [confirmedIds, setConfirmedIds] = useState(new Set())
  const [dismissedIds, setDismissedIds] = useState(new Set())

  const load = useCallback(async (refresh = false) => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchRecommendations(refresh)
      setRecs(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(false) }, [load])

  const visible = recs.filter(r => !confirmedIds.has(r.yt_video_id) && !dismissedIds.has(r.yt_video_id))

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: C.white }}>
            Discover
          </h3>
          <p style={{ margin: '3px 0 0', fontSize: 12, color: C.textMuted }}>
            Unseen instructionals from your configured channels + search tags.
          </p>
        </div>
        <button
          onClick={() => load(true)}
          disabled={loading}
          style={{
            padding: '7px 16px',
            fontSize: 12,
            fontWeight: 600,
            borderRadius: radius.sm,
            border: `1px solid ${C.border}`,
            background: 'transparent',
            color: loading ? C.textMuted : C.textSecondary,
            cursor: loading ? 'default' : 'pointer',
          }}
        >
          {loading ? 'Fetching…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{
          padding: '10px 14px',
          background: `${C.danger}15`,
          border: `1px solid ${C.danger}55`,
          borderRadius: radius.md,
          fontSize: 13,
          color: C.danger,
          marginBottom: 12,
        }}>
          {error}
        </div>
      )}

      {loading && recs.length === 0 && (
        <div style={{ padding: '40px 0', textAlign: 'center', color: C.textMuted, fontSize: 13 }}>
          Searching YouTube — this may take a few seconds…
        </div>
      )}

      {!loading && visible.length === 0 && !error && (
        <div style={{ padding: '40px 0', textAlign: 'center', color: C.textMuted, fontSize: 13 }}>
          {recs.length === 0
            ? 'No results yet. Add channels or search tags in config.yaml, then Refresh.'
            : 'All recommendations confirmed or dismissed. Refresh to fetch more.'}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {visible.map(rec => (
          <RecommendationCard
            key={rec.yt_video_id}
            rec={rec}
            onConfirmed={id => setConfirmedIds(s => new Set([...s, id]))}
            onDismissed={id => setDismissedIds(s => new Set([...s, id]))}
          />
        ))}
      </div>
    </div>
  )
}
