import React, { useEffect, useState, useCallback } from 'react'
import { C, radius, shadow } from '../theme'

function fetchMetrics() {
  return fetch('/api/metrics').then(r => {
    if (!r.ok) throw new Error(`API ${r.status}`)
    return r.json()
  })
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: radius.md,
      padding: '16px 20px',
      minWidth: 110,
      flex: 1,
    }}>
      <div style={{ fontSize: 11, color: C.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color: accent || C.white, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.textSecondary, marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

function SectionHeader({ children }) {
  return (
    <div style={{
      fontSize: 11,
      fontWeight: 700,
      color: C.textMuted,
      textTransform: 'uppercase',
      letterSpacing: '0.1em',
      marginBottom: 12,
    }}>
      {children}
    </div>
  )
}

function Section({ children, style }) {
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: radius.md,
      padding: '20px 24px',
      ...style,
    }}>
      {children}
    </div>
  )
}

function StageBar({ stage, ms, totalMs }) {
  const pct = totalMs > 0 ? Math.round((ms / totalMs) * 100) : 0
  const barColor = pct > 60 ? C.danger : pct > 30 ? '#e0a050' : C.blue
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
      <div style={{ width: 110, fontSize: 13, color: C.textSecondary, flexShrink: 0 }}>{stage}</div>
      <div style={{ flex: 1, background: C.bg, borderRadius: 3, height: 10, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
      <div style={{ width: 70, textAlign: 'right', fontSize: 13, color: C.textPrimary, fontVariantNumeric: 'tabular-nums' }}>
        {ms.toFixed(1)} ms
      </div>
      <div style={{ width: 36, textAlign: 'right', fontSize: 12, color: C.textMuted, fontVariantNumeric: 'tabular-nums' }}>
        {pct}%
      </div>
    </div>
  )
}

function VramBar({ label, gb, totalGb, color }) {
  const pct = totalGb > 0 ? Math.round((gb / totalGb) * 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
      <div style={{ width: 100, fontSize: 13, color: C.textSecondary, flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, background: C.bg, borderRadius: 3, height: 10, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <div style={{ width: 70, textAlign: 'right', fontSize: 13, color: C.textPrimary, fontVariantNumeric: 'tabular-nums' }}>
        {gb.toFixed(2)} GB
      </div>
      <div style={{ width: 36, textAlign: 'right', fontSize: 12, color: C.textMuted }}>
        {pct}%
      </div>
    </div>
  )
}

function ConformalSection({ conformal, onCalibrate, calibrating }) {
  const enabled = conformal?.enabled
  const calibrated = conformal?.calibrated

  return (
    <Section>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <SectionHeader>Conformal Reranking</SectionHeader>
        {enabled && (
          <button
            onClick={onCalibrate}
            disabled={calibrating}
            style={{
              background: calibrating ? C.surfaceHigh : C.purple,
              border: 'none',
              borderRadius: radius.sm,
              color: calibrating ? C.textMuted : C.white,
              padding: '5px 14px',
              fontSize: 13,
              cursor: calibrating ? 'default' : 'pointer',
              fontWeight: 600,
            }}
          >
            {calibrating ? 'Calibrating…' : 'Calibrate'}
          </button>
        )}
      </div>

      {!enabled ? (
        <div style={{ fontSize: 13, color: C.textMuted }}>
          Disabled — set <code style={{ color: C.brown }}>reranker.conformal_coverage: 0.90</code> in config.yaml to enable.
        </div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Status',          calibrated ? 'Calibrated' : 'Uncalibrated (using plain reranker)'],
              ['Target coverage', conformal.target_coverage != null ? `${(conformal.target_coverage * 100).toFixed(0)}%` : '—'],
              ['q̂ (threshold)',   conformal.q_hat != null ? conformal.q_hat.toFixed(4) : '—'],
            ].map(([k, v]) => (
              <tr key={k} style={{ borderBottom: `1px solid ${C.bg}` }}>
                <td style={{ padding: '7px 0', color: C.textSecondary }}>{k}</td>
                <td style={{ padding: '7px 0', color: calibrated ? C.white : C.textMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  )
}

export default function PerformanceMonitor() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [calibrating, setCalibrating] = useState(false)
  const [calibrateMsg, setCalibrateMsg] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    fetchMetrics()
      .then(d => {
        setData(d)
        setError(null)
        setLastUpdated(new Date())
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const handleCalibrate = useCallback(() => {
    setCalibrating(true)
    setCalibrateMsg(null)
    fetch('/api/conformal/calibrate', { method: 'POST' })
      .then(r => r.json().then(body => ({ ok: r.ok, body })))
      .then(({ ok, body }) => {
        if (ok) {
          setCalibrateMsg(`Calibrated: q̂=${body.q_hat?.toFixed(4)}, coverage=${(body.empirical_coverage * 100).toFixed(1)}% on ${body.n_pairs} pairs`)
          load()
        } else {
          setCalibrateMsg(`Error: ${body.detail}`)
        }
      })
      .catch(e => setCalibrateMsg(`Error: ${e.message}`))
      .finally(() => setCalibrating(false))
  }, [load])

  useEffect(() => {
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [load])

  if (loading && !data) {
    return (
      <div style={{ color: C.textSecondary, padding: 40, textAlign: 'center' }}>
        Loading metrics…
      </div>
    )
  }

  if (error && !data) {
    return (
      <div style={{ color: C.danger, padding: 40, textAlign: 'center' }}>
        {error}
      </div>
    )
  }

  const lat = data?.latency || {}
  const last = data?.last_search || {}
  const vram = data?.vram || {}
  const idx = data?.index || {}
  const conformal = data?.conformal || {}

  const stages = Object.entries(last.stages_ms || {}).sort((a, b) => b[1] - a[1])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 900 }}>

      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: C.white }}>Performance Monitor</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {lastUpdated && (
            <span style={{ fontSize: 12, color: C.textMuted }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={load}
            disabled={loading}
            style={{
              background: C.surfaceHigh,
              border: `1px solid ${C.border}`,
              borderRadius: radius.sm,
              color: loading ? C.textMuted : C.textPrimary,
              padding: '5px 14px',
              fontSize: 13,
              cursor: loading ? 'default' : 'pointer',
            }}
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Latency stat cards */}
      <div>
        <SectionHeader>Historical Latency — {lat.n_queries ?? 0} queries logged</SectionHeader>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <StatCard label="P50" value={lat.n_queries ? `${lat.p50_ms}ms` : '—'} sub="median" accent={C.blue} />
          <StatCard label="P95" value={lat.n_queries ? `${lat.p95_ms}ms` : '—'} sub="95th pct" accent={lat.p95_ms > 500 ? C.danger : '#e0a050'} />
          <StatCard label="P99" value={lat.n_queries ? `${lat.p99_ms}ms` : '—'} sub="99th pct" accent={lat.p99_ms > 1000 ? C.danger : C.textPrimary} />
          <StatCard label="Mean" value={lat.n_queries ? `${lat.mean_ms}ms` : '—'} sub="average" />
          <StatCard label="Range" value={lat.n_queries ? `${lat.min_ms}–${lat.max_ms}` : '—'} sub="min – max ms" />
        </div>
      </div>

      {/* Last search breakdown */}
      <Section>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <SectionHeader>Last Search — Stage Breakdown</SectionHeader>
          {last.total_ms != null && (
            <div style={{ fontSize: 20, fontWeight: 700, color: C.white, fontVariantNumeric: 'tabular-nums' }}>
              {last.total_ms} ms total
            </div>
          )}
        </div>
        {stages.length === 0 ? (
          <div style={{ color: C.textMuted, fontSize: 14 }}>No search recorded yet — run a query first.</div>
        ) : (
          stages.map(([stage, ms]) => (
            <StageBar key={stage} stage={stage} ms={ms} totalMs={last.total_ms || 1} />
          ))
        )}
      </Section>

      {/* Index stats + VRAM side by side */}
      <div style={{ display: 'flex', gap: 16 }}>

        {/* Index stats */}
        <Section style={{ flex: 1 }}>
          <SectionHeader>Index Statistics</SectionHeader>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <tbody>
              {[
                ['Chunks indexed', (idx.total_chunks ?? 0).toLocaleString()],
                ['Total videos', (idx.total_videos ?? 0).toLocaleString()],
                ['Indexed videos', (idx.indexed_videos ?? 0).toLocaleString()],
                ['Uncertain (unreviewed)', (idx.uncertain_chunks_pending ?? 0).toLocaleString()],
                ['Impressions logged', (idx.total_impressions_logged ?? 0).toLocaleString()],
                ['Clicks logged', (idx.total_clicks_logged ?? 0).toLocaleString()],
                ['CTR', idx.total_impressions_logged ? `${(idx.ctr * 100).toFixed(1)}%` : '—'],
              ].map(([k, v]) => (
                <tr key={k} style={{ borderBottom: `1px solid ${C.bg}` }}>
                  <td style={{ padding: '7px 0', color: C.textSecondary }}>{k}</td>
                  <td style={{ padding: '7px 0', color: C.white, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        {/* VRAM */}
        <Section style={{ flex: 1 }}>
          <SectionHeader>GPU Memory</SectionHeader>
          {!vram.available ? (
            <div style={{ color: C.textMuted, fontSize: 14, paddingTop: 8 }}>
              No CUDA device available (CPU mode).
            </div>
          ) : (
            <>
              <div style={{ fontSize: 14, color: C.textSecondary, marginBottom: 14 }}>
                {vram.device} &mdash; {vram.total_gb} GB total
              </div>

              {/* GPU compute utilisation */}
              {vram.gpu_util_pct != null && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
                    <div style={{ width: 100, fontSize: 13, color: C.textSecondary, flexShrink: 0 }}>GPU util</div>
                    <div style={{ flex: 1, background: C.bg, borderRadius: 3, height: 10, overflow: 'hidden' }}>
                      <div style={{
                        width: `${vram.gpu_util_pct}%`, height: '100%', borderRadius: 3,
                        background: vram.gpu_util_pct > 80 ? C.danger : vram.gpu_util_pct > 40 ? '#e0a050' : C.success,
                        transition: 'width 0.4s ease',
                      }} />
                    </div>
                    <div style={{ width: 70, textAlign: 'right', fontSize: 13, color: C.textPrimary, fontVariantNumeric: 'tabular-nums' }}>
                      {vram.gpu_util_pct}%
                    </div>
                    <div style={{ width: 36 }} />
                  </div>
                </div>
              )}

              {/* System-wide VRAM (all processes: whisper, Ollama, torch, etc.) */}
              <div style={{ fontSize: 11, fontWeight: 700, color: C.textMuted, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                All processes
              </div>
              <VramBar label="Used" gb={vram.system_used_gb ?? 0} totalGb={vram.total_gb} color={C.blue} />
              <VramBar label="Free" gb={vram.system_free_gb ?? 0} totalGb={vram.total_gb} color={C.success} />

              {/* PyTorch-only allocator */}
              {vram.pytorch_allocated_gb != null && (
                <>
                  <div style={{ fontSize: 11, fontWeight: 700, color: C.textMuted, textTransform: 'uppercase', letterSpacing: '0.07em', margin: '12px 0 6px' }}>
                    PyTorch (this process)
                  </div>
                  <VramBar label="Allocated" gb={vram.pytorch_allocated_gb}      totalGb={vram.total_gb} color={C.purple} />
                  <VramBar label="Reserved"  gb={vram.pytorch_reserved_gb}       totalGb={vram.total_gb} color={'#6060a0'} />
                  <VramBar label="Peak"      gb={vram.pytorch_peak_allocated_gb} totalGb={vram.total_gb} color={C.danger} />
                  {vram.pytorch_allocated_gb === 0 && (
                    <div style={{ marginTop: 8, fontSize: 12, color: C.textMuted }}>
                      PyTorch shows 0 — embedding &amp; reranker models are on CPU. Set <code style={{ color: C.brown }}>device: cuda</code> in config.yaml to move them to GPU.
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </Section>

      </div>

      {/* Conformal reranking */}
      <ConformalSection
        conformal={conformal}
        onCalibrate={handleCalibrate}
        calibrating={calibrating}
      />
      {calibrateMsg && (
        <div style={{
          fontSize: 13,
          color: calibrateMsg.startsWith('Error') ? C.danger : C.success,
          padding: '8px 12px',
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.sm,
        }}>
          {calibrateMsg}
        </div>
      )}

    </div>
  )
}
