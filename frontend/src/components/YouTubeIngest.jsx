import React, { useState, useEffect, useRef } from 'react'
import { submitYouTube, fetchYouTubeStatus, fetchYouTubeJobs } from '../utils/api'
import { C, radius, shadow } from '../theme'
import YouTubeRecommendations from './YouTubeRecommendations'

const STATUS_COLOR = {
  starting:          C.textMuted,
  fetching_metadata: C.brown,
  downloading:       C.blue,
  registering:       C.purple,
  complete:          C.success,
  error:             C.danger,
}

const STATUS_LABEL = {
  starting:          'Starting…',
  fetching_metadata: 'Fetching metadata…',
  downloading:       'Downloading…',
  registering:       'Registering in database…',
  complete:          'Complete',
  error:             'Failed',
}

function JobRow({ job }) {
  const color = STATUS_COLOR[job.status] || C.textMuted
  const label = STATUS_LABEL[job.status] || job.status
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '10px 16px',
      borderBottom: `1px solid ${C.border}`,
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: C.textPrimary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {job.title || job.url}
        </div>
        {job.title && (
          <div style={{ fontSize: 11, color: C.textMuted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 2 }}>
            {job.url}
          </div>
        )}
      </div>

      {job.status === 'downloading' && (
        <div style={{ fontSize: 11, color: C.blue, fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
          {job.progress_pct}%
        </div>
      )}

      <div style={{
        fontSize: 11,
        fontWeight: 700,
        color,
        background: `${color}18`,
        border: `1px solid ${color}44`,
        borderRadius: radius.pill,
        padding: '2px 10px',
        whiteSpace: 'nowrap',
        flexShrink: 0,
      }}>
        {label}
      </div>

      {job.error && (
        <div style={{ fontSize: 11, color: C.danger, maxWidth: 240, flexShrink: 0 }}>
          {job.error}
        </div>
      )}
    </div>
  )
}

export default function YouTubeIngest() {
  const [tab, setTab]                     = useState('add')
  const [url, setUrl]                     = useState('')
  const [giNogi, setGiNogi]               = useState('nogi')
  const [instructor, setInstructor]       = useState('')
  const [category, setCategory]           = useState('Uncategorised')
  const [submitting, setSubmitting]       = useState(false)
  const [error, setError]                 = useState(null)
  const [jobs, setJobs]                   = useState([])
  const [activeIds, setActiveIds]         = useState(new Set())
  const pollRef                           = useRef(null)

  // Initial load of all past jobs
  useEffect(() => {
    fetchYouTubeJobs().then(setJobs).catch(() => {})
  }, [])

  // Poll active jobs every 2s
  useEffect(() => {
    if (activeIds.size === 0) return
    const id = setInterval(async () => {
      const updated = await Promise.all(
        [...activeIds].map(did => fetchYouTubeStatus(did).catch(() => null))
      )
      setJobs(prev => {
        const map = Object.fromEntries(prev.map(j => [j.download_id, j]))
        for (const job of updated) {
          if (job) map[job.download_id] = job
        }
        return Object.values(map)
      })
      setActiveIds(prev => {
        const next = new Set(prev)
        for (const job of updated) {
          if (job && (job.status === 'complete' || job.status === 'error')) {
            next.delete(job.download_id)
          }
        }
        return next
      })
    }, 2000)
    return () => clearInterval(id)
  }, [activeIds])

  async function handleSubmit(e) {
    e.preventDefault()
    if (!url.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await submitYouTube(url.trim(), giNogi, instructor.trim(), category)
      setJobs(prev => [{ download_id: res.download_id, status: 'starting', url: url.trim(), title: null, progress_pct: 0, error: null }, ...prev])
      setActiveIds(prev => new Set([...prev, res.download_id]))
      setUrl('')
      setInstructor('')
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  const inputStyle = {
    width: '100%',
    boxSizing: 'border-box',
    background: C.surfaceHigh,
    border: `1px solid ${C.border}`,
    borderRadius: radius.sm,
    padding: '8px 12px',
    fontSize: 13,
    color: C.textPrimary,
    outline: 'none',
  }

  const labelStyle = {
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color: C.textMuted,
    display: 'block',
    marginBottom: 6,
  }

  const tabBtn = (key, label) => (
    <button
      key={key}
      onClick={() => setTab(key)}
      style={{
        padding: '6px 18px',
        fontSize: 13,
        fontWeight: 600,
        borderRadius: radius.sm,
        border: 'none',
        background: tab === key ? C.surfaceHigh : 'transparent',
        color: tab === key ? C.white : C.textSecondary,
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  )

  return (
    <div style={{ maxWidth: 720 }}>
      {/* Page header + tab switcher */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: C.white, margin: '0 0 12px' }}>
          YouTube
        </h2>
        <div style={{
          display: 'inline-flex',
          gap: 2,
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
          padding: 3,
        }}>
          {tabBtn('add', 'Add URL')}
          {tabBtn('discover', 'Discover')}
        </div>
      </div>

      {tab === 'discover' && <YouTubeRecommendations />}

      {tab === 'add' && <>
      {/* Form */}
      <form onSubmit={handleSubmit}>
        <div style={{
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
          padding: '18px 20px',
          boxShadow: shadow.card,
          marginBottom: 16,
        }}>
          <div style={{ marginBottom: 14 }}>
            <label style={labelStyle}>YouTube URL</label>
            <input
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              required
              style={{ ...inputStyle, borderColor: url ? C.blue : C.border }}
            />
          </div>

          <div style={{ display: 'flex', gap: 12, marginBottom: 14 }}>
            <div style={{ flex: 1 }}>
              <label style={labelStyle}>Gi / No-Gi</label>
              <select
                value={giNogi}
                onChange={e => setGiNogi(e.target.value)}
                style={{ ...inputStyle }}
              >
                <option value="nogi">No-Gi</option>
                <option value="gi">Gi</option>
                <option value="both">Both</option>
              </select>
            </div>
            <div style={{ flex: 2 }}>
              <label style={labelStyle}>Instructor (optional)</label>
              <input
                type="text"
                value={instructor}
                onChange={e => setInstructor(e.target.value)}
                placeholder="Auto-detected from video uploader"
                style={inputStyle}
              />
            </div>
          </div>

          <div style={{ marginBottom: 18 }}>
            <label style={labelStyle}>Category</label>
            <input
              type="text"
              value={category}
              onChange={e => setCategory(e.target.value)}
              placeholder="e.g. Guard, Submissions, Passing…"
              style={inputStyle}
            />
          </div>

          <button
            type="submit"
            disabled={!url.trim() || submitting}
            style={{
              padding: '9px 24px',
              fontSize: 14,
              fontWeight: 600,
              borderRadius: radius.sm,
              border: 'none',
              background: !url.trim() || submitting ? C.surfaceHigh : C.blue,
              color: !url.trim() || submitting ? C.textMuted : C.white,
              cursor: !url.trim() || submitting ? 'default' : 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {submitting ? 'Submitting…' : 'Download & Ingest'}
          </button>
        </div>
      </form>

      {error && (
        <div style={{
          padding: '12px 16px',
          background: `${C.danger}15`,
          border: `1px solid ${C.danger}55`,
          borderRadius: radius.md,
          fontSize: 13,
          color: C.danger,
          marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {/* Job history */}
      {jobs.length > 0 && (
        <div style={{
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
          overflow: 'hidden',
          boxShadow: shadow.card,
        }}>
          <div style={{
            padding: '8px 16px',
            borderBottom: `1px solid ${C.border}`,
            background: C.surfaceHigh,
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: C.textMuted,
          }}>
            Download History
          </div>
          {jobs.map(job => <JobRow key={job.download_id} job={job} />)}
        </div>
      )}

      {jobs.length === 0 && (
        <div style={{
          padding: '32px 0',
          textAlign: 'center',
          color: C.textMuted,
          fontSize: 13,
        }}>
          No downloads yet. Paste a URL above to get started.
        </div>
      )}
      </>}
    </div>
  )
}
