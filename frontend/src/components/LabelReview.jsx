import React, { useState, useEffect, useCallback } from 'react'
import { fetchLabelStats, fetchReviewQueue, fetchDriftLog, saveLabel, triggerBackfill, triggerClustering, fetchProposal } from '../utils/api'
import { C, radius, shadow } from '../theme'

const ALL_POSITIONS = [
  '50_50','ashi_garami','back_control','closed_guard','crucifix',
  'front_headlock','half_guard','knee_on_belly','mount','north_south',
  'open_guard','side_control','standing','turtle',
]
const ALL_INTENTS = ['attack','control','defense','pass','sweep','takedown','transition']
const ALL_TECHNIQUES = [
  'Von_flue_choke','americana','anaconda','ankle_lock','arm_drag','arm_triangle',
  'armbar','baseball_bat_choke','berimbolo','body_lock','bow_and_arrow','bridge',
  'clock_choke','collar_tie','crossface','crucifix_choke','darce','ezekiel','frame',
  'gogoplata','guillotine','heel_hook','hip_escape','kimura','knee_bar','lasso',
  'leg_drag','loop_choke','north_south_choke','omoplata','overhook','peruvian_necktie',
  'rear_naked_choke','sit_up_guard','spider_guard','toe_hold','triangle','underhook','wrist_lock',
]
const CONTEXTS  = ['top','bottom','neutral','standing']
const GI_NOGI   = ['gi','nogi','both']

function fmt(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function entropyColor(e) {
  if (e >= 0.75) return C.danger
  if (e >= 0.5)  return '#e09030'
  return C.textMuted
}

function MultiSelect({ label, options, value, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.textMuted, marginBottom: 5 }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {options.map(opt => {
          const active = value.includes(opt)
          return (
            <button
              key={opt}
              onClick={() => onChange(active ? value.filter(v => v !== opt) : [...value, opt])}
              style={{
                fontSize: 11,
                padding: '3px 9px',
                borderRadius: radius.pill,
                border: `1px solid ${active ? C.purple : C.border}`,
                background: active ? `${C.purple}33` : 'transparent',
                color: active ? C.purpleLight : C.textMuted,
                cursor: 'pointer',
                transition: 'all 0.1s',
              }}
            >
              {opt.replace(/_/g, ' ')}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function SingleSelect({ label, options, value, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.textMuted, marginBottom: 5 }}>
        {label}
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {options.map(opt => {
          const active = value === opt
          return (
            <button
              key={opt}
              onClick={() => onChange(active ? null : opt)}
              style={{
                fontSize: 11,
                padding: '3px 9px',
                borderRadius: radius.pill,
                border: `1px solid ${active ? C.blue : C.border}`,
                background: active ? `${C.blue}33` : 'transparent',
                color: active ? '#90c0f0' : C.textMuted,
                cursor: 'pointer',
                transition: 'all 0.1s',
              }}
            >
              {opt}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function ChunkCard({ item, onDone }) {
  const [positions,  setPositions]  = useState(item.positions)
  const [intents,    setIntents]    = useState(item.intents)
  const [techniques, setTechniques] = useState(item.techniques)
  const [context,    setContext]    = useState(item.context)
  const [giNogi,     setGiNogi]     = useState(item.gi_nogi)
  const [saving,     setSaving]     = useState(false)
  const [expanded,   setExpanded]   = useState(false)
  const [suggesting, setSuggesting] = useState(false)
  const [suggestErr, setSuggestErr] = useState(null)

  async function handleSuggest() {
    setSuggesting(true)
    setSuggestErr(null)
    try {
      const p = await fetchProposal(item.chunk_id)
      if (p.positions.length)  setPositions(p.positions)
      if (p.intents.length)    setIntents(p.intents)
      if (p.techniques.length) setTechniques(p.techniques)
    } catch {
      setSuggestErr('No proposal — run Cluster first')
    } finally {
      setSuggesting(false)
    }
  }

  async function submit(approved) {
    setSaving(true)
    try {
      await saveLabel(item.chunk_id, { positions, intents, techniques, context, gi_nogi: giNogi, approved })
      onDone(item.chunk_id)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: radius.md,
      marginBottom: 14,
      overflow: 'hidden',
      boxShadow: shadow.card,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 14px',
        background: C.surfaceHigh,
        borderBottom: `1px solid ${C.border}`,
      }}>
        <div>
          <span style={{ fontSize: 13, fontWeight: 600, color: C.textPrimary }}>{item.video_title}</span>
          <span style={{ fontSize: 11, color: C.textMuted, marginLeft: 8 }}>
            {item.instructor} · {fmt(item.start_time)}–{fmt(item.end_time)}
          </span>
        </div>
        <span style={{
          fontSize: 11,
          fontWeight: 700,
          color: entropyColor(item.entropy),
          background: `${entropyColor(item.entropy)}22`,
          border: `1px solid ${entropyColor(item.entropy)}55`,
          padding: '2px 8px',
          borderRadius: radius.pill,
        }}>
          entropy {item.entropy.toFixed(2)}
        </span>
      </div>

      {/* Transcript */}
      <div style={{ padding: '10px 14px', borderBottom: `1px solid ${C.border}` }}>
        <p style={{
          fontSize: 13,
          lineHeight: 1.6,
          color: C.textSecondary,
          margin: 0,
          maxHeight: expanded ? 'none' : 80,
          overflow: 'hidden',
          position: 'relative',
        }}>
          {item.transcript}
        </p>
        {item.transcript.length > 240 && (
          <button
            onClick={() => setExpanded(e => !e)}
            style={{ fontSize: 11, color: C.purple, background: 'none', border: 'none', cursor: 'pointer', padding: '4px 0 0' }}
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>

      {/* Label editors */}
      <div style={{ padding: '12px 14px', borderBottom: `1px solid ${C.border}` }}>
        <MultiSelect label="Positions"  options={ALL_POSITIONS}  value={positions}  onChange={setPositions} />
        <MultiSelect label="Intents"    options={ALL_INTENTS}    value={intents}    onChange={setIntents} />
        <MultiSelect label="Techniques" options={ALL_TECHNIQUES} value={techniques} onChange={setTechniques} />
        <div style={{ display: 'flex', gap: 32 }}>
          <SingleSelect label="Context" options={CONTEXTS} value={context} onChange={setContext} />
          <SingleSelect label="Gi / Nogi" options={GI_NOGI} value={giNogi} onChange={setGiNogi} />
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8, padding: '10px 14px', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            onClick={handleSuggest}
            disabled={suggesting || saving}
            title="Fill labels from HDBSCAN cluster proposal"
            style={{
              fontSize: 11,
              padding: '4px 12px',
              borderRadius: radius.sm,
              border: `1px solid ${C.border}`,
              background: 'transparent',
              color: suggesting ? C.textMuted : C.textSecondary,
              cursor: suggesting || saving ? 'default' : 'pointer',
            }}
          >
            {suggesting ? 'Fetching…' : 'Suggest'}
          </button>
          {suggestErr && (
            <span style={{ fontSize: 11, color: C.textMuted }}>{suggestErr}</span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => submit(true)}
            disabled={saving}
            style={{
              fontSize: 12,
              padding: '6px 16px',
              borderRadius: radius.sm,
              border: `1px solid ${C.border}`,
              background: 'transparent',
              color: saving ? C.textMuted : C.success,
              cursor: saving ? 'default' : 'pointer',
            }}
          >
            {saving ? '…' : 'Approve as-is'}
          </button>
          <button
            onClick={() => submit(false)}
            disabled={saving}
            style={{
              fontSize: 12,
              padding: '6px 16px',
              borderRadius: radius.sm,
              border: 'none',
              background: saving ? C.surfaceHigh : C.purple,
              color: saving ? C.textMuted : C.white,
              cursor: saving ? 'default' : 'pointer',
            }}
          >
            {saving ? 'Saving…' : 'Save corrections'}
          </button>
        </div>
      </div>
    </div>
  )
}

function DriftPanel({ events }) {
  if (events.length === 0) return null
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: radius.md,
      overflow: 'hidden',
      boxShadow: shadow.card,
      marginTop: 24,
    }}>
      <div style={{ padding: '10px 14px', background: C.surfaceHigh, borderBottom: `1px solid ${C.border}` }}>
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.danger }}>
          Taxonomy Drift Events
        </span>
      </div>
      {events.map(e => (
        <div key={e.id} style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '9px 14px',
          borderBottom: `1px solid ${C.border}`,
          fontSize: 13,
        }}>
          <span style={{ color: C.textPrimary }}>{e.taxonomy_class.replace(/_/g, ' ')}</span>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <span style={{ color: C.danger, fontWeight: 600 }}>
              dist {e.distance.toFixed(3)}
            </span>
            <span style={{ fontSize: 11, color: C.textMuted }}>
              {new Date(e.detected_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function LabelReview() {
  const [stats,    setStats]    = useState(null)
  const [queue,    setQueue]    = useState([])
  const [drift,    setDrift]    = useState([])
  const [loading,  setLoading]  = useState(true)
  const [backfilling,    setBackfilling]    = useState(false)
  const [backfillResult, setBackfillResult] = useState(null)
  const [clustering,     setClustering]     = useState(false)
  const [clusterResult,  setClusterResult]  = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [s, q, d] = await Promise.all([fetchLabelStats(), fetchReviewQueue(20), fetchDriftLog()])
      setStats(s)
      setQueue(q)
      setDrift(d)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  function handleDone(chunkId) {
    setQueue(q => q.filter(c => c.chunk_id !== chunkId))
    setStats(s => s ? { ...s, pending: s.pending - 1, reviewed: s.reviewed + 1 } : s)
  }

  async function handleBackfill() {
    setBackfilling(true)
    setBackfillResult(null)
    try {
      const res = await triggerBackfill()
      setBackfillResult(res.written)
      load()
    } finally {
      setBackfilling(false)
    }
  }

  async function handleCluster() {
    setClustering(true)
    setClusterResult(null)
    try {
      const res = await triggerClustering()
      setClusterResult(res.written)
    } finally {
      setClustering(false)
    }
  }

  return (
    <div style={{ maxWidth: 780 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: C.textPrimary, margin: '0 0 4px' }}>
            Label Review
          </h2>
          <p style={{ fontSize: 13, color: C.textMuted, margin: 0 }}>
            Review and correct high-entropy chunks the classifier was uncertain about.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={handleCluster}
            disabled={clustering}
            title="Run HDBSCAN to generate label proposals for uncertain chunks"
            style={{
              fontSize: 12,
              padding: '6px 14px',
              borderRadius: radius.sm,
              border: `1px solid ${C.border}`,
              background: 'transparent',
              color: clustering ? C.textMuted : C.textSecondary,
              cursor: clustering ? 'default' : 'pointer',
            }}
          >
            {clustering ? 'Clustering…' : 'Cluster'}
          </button>
          <button
            onClick={handleBackfill}
            disabled={backfilling}
            style={{
              fontSize: 12,
              padding: '6px 14px',
              borderRadius: radius.sm,
              border: `1px solid ${C.border}`,
              background: 'transparent',
              color: backfilling ? C.textMuted : C.textSecondary,
              cursor: backfilling ? 'default' : 'pointer',
            }}
          >
            {backfilling ? 'Scoring…' : 'Backfill scores'}
          </button>
        </div>
      </div>

      {backfillResult !== null && (
        <div style={{ marginBottom: 8, fontSize: 13, color: C.success }}>
          Backfill complete — {backfillResult} uncertain chunk{backfillResult !== 1 ? 's' : ''} scored.
        </div>
      )}
      {clusterResult !== null && (
        <div style={{ marginBottom: 14, fontSize: 13, color: C.success }}>
          Clustering complete — {clusterResult} proposal{clusterResult !== 1 ? 's' : ''} written. Click "Suggest" on any chunk card to pre-fill labels.
        </div>
      )}

      {/* Stats bar */}
      {stats && (
        <div style={{
          display: 'flex',
          gap: 20,
          padding: '12px 16px',
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
          marginBottom: 20,
          boxShadow: shadow.card,
        }}>
          {[
            { label: 'Pending review', value: stats.pending, color: C.blue },
            { label: 'Reviewed',       value: stats.reviewed, color: C.success },
            { label: 'Avg entropy',    value: stats.avg_entropy.toFixed(2), color: entropyColor(stats.avg_entropy) },
            { label: 'Drift events',   value: stats.drift_events, color: stats.drift_events > 0 ? C.danger : C.textMuted },
          ].map(({ label, value, color }) => (
            <div key={label}>
              <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
              <div style={{ fontSize: 11, color: C.textMuted, marginTop: 1 }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {loading ? (
        <p style={{ color: C.textMuted, fontSize: 13 }}>Loading…</p>
      ) : queue.length === 0 ? (
        <div style={{
          padding: '40px 0',
          textAlign: 'center',
          color: C.textMuted,
          fontSize: 13,
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
        }}>
          {stats?.pending === 0
            ? 'All caught up — no uncertain chunks to review.'
            : 'No chunks loaded. Try "Backfill scores" if you have indexed videos.'}
        </div>
      ) : (
        queue.map(item => (
          <ChunkCard key={item.chunk_id} item={item} onDone={handleDone} />
        ))
      )}

      <DriftPanel events={drift} />
    </div>
  )
}
