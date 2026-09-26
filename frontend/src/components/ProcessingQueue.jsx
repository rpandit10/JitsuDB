import React, { useEffect } from 'react'
import ProgressBar from './ProgressBar'
import useQueue from '../hooks/useQueue'
import useSSE from '../hooks/useSSE'
import { clearQueue } from '../utils/api'
import { C, radius, shadow } from '../theme'

function formatDuration(seconds) {
  if (seconds == null || seconds < 0) return null
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

const STATUS_ORDER = { processing: 0, pending: 1, done: 3, failed: 2 }
const STATUS_COLOR = {
  pending:    C.textMuted,
  processing: C.blue,
  done:       C.success,
  failed:     C.danger,
}

function NavBtn({ onClick, disabled, children, variant = 'default' }) {
  const bg     = variant === 'primary' ? C.blue : C.surfaceHigh
  const border = variant === 'danger'  ? C.danger : 'transparent'
  const color  = variant === 'danger'  ? C.danger
               : variant === 'primary' ? C.white
               : C.textSecondary
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: '6px 14px',
        fontSize: 13,
        fontWeight: 500,
        background: disabled ? C.surface : bg,
        border: `1px solid ${disabled ? C.border : border}`,
        borderRadius: radius.sm,
        color: disabled ? C.textMuted : color,
        transition: 'all 0.15s',
      }}
    >
      {children}
    </button>
  )
}

function EtaBar({ label, value, etaSeconds, color }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.textMuted }}>
          {label}
        </span>
        <span style={{ fontSize: 11, color: etaSeconds != null ? color : C.textMuted }}>
          {etaSeconds != null ? `~${formatDuration(etaSeconds)} remaining` : '—'}
        </span>
      </div>
      <ProgressBar value={value} accentColor={color} />
    </div>
  )
}

export default function ProcessingQueue() {
  const { queue, loading, refresh, startPipeline, pausePipeline, resumePipeline, cancelPipeline, retryFailedPipeline } = useQueue()
  const progress = useSSE('/api/status')

  useEffect(() => {
    refresh()
  }, [progress?.video_id, progress?.is_running, refresh])

  async function handleClearQueue() {
    await clearQueue()
    refresh()
  }

  if (loading) return <p style={{ color: C.textMuted }}>Loading queue…</p>

  const isRunning = progress?.is_running
  const isPaused  = progress?.is_paused
  const isIdle    = progress?.is_idle

  const failedCount = queue.filter(i => i.status === 'failed').length

  const sortedQueue = [...queue].sort((a, b) => {
    const ao = STATUS_ORDER[a.status] ?? 1
    const bo = STATUS_ORDER[b.status] ?? 1
    if (ao !== bo) return ao - bo
    return (a.priority ?? 0) - (b.priority ?? 0)
  })

  return (
    <div style={{ maxWidth: 760 }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        <NavBtn variant="primary" onClick={startPipeline} disabled={isRunning && !isPaused}>Start</NavBtn>
        <NavBtn onClick={isPaused ? resumePipeline : pausePipeline} disabled={!isRunning}>
          {isPaused ? 'Resume' : 'Pause'}
        </NavBtn>
        <NavBtn onClick={cancelPipeline} disabled={!isRunning}>Cancel</NavBtn>
        {failedCount > 0 && (
          <NavBtn onClick={retryFailedPipeline} disabled={isRunning}>
            Retry Failed ({failedCount})
          </NavBtn>
        )}
        <NavBtn variant="danger" onClick={handleClearQueue} disabled={isRunning}>Clear Queue</NavBtn>
      </div>

      {/* Progress card */}
      <div style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: radius.md,
        padding: '18px 20px',
        marginBottom: 20,
        boxShadow: shadow.card,
      }}>
        {isRunning && !isIdle ? (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: C.textPrimary }}>
                {progress.video_title || '…'}
              </span>
              <span style={{
                fontSize: 11, fontWeight: 700, padding: '2px 9px',
                borderRadius: radius.pill, background: C.surfaceHigh,
                color: C.blue, letterSpacing: '0.07em', textTransform: 'uppercase',
              }}>
                {progress.stage || '…'}
              </span>
            </div>

            <EtaBar
              label="Current video"
              value={progress.video_progress ?? 0}
              etaSeconds={progress.video_eta_seconds}
              color={C.blue}
            />
            <EtaBar
              label="Entire queue"
              value={progress.overall_progress ?? 0}
              etaSeconds={progress.queue_eta_seconds}
              color={C.purple}
            />

            <p style={{ fontSize: 11, color: C.textMuted, marginTop: 4 }}>
              Elapsed: {formatDuration(progress.elapsed_seconds)}
            </p>
          </>
        ) : (
          <p style={{ color: C.textMuted, fontSize: 13, textAlign: 'center' }}>
            {isPaused
              ? 'Paused'
              : isIdle
                ? 'Watching queue — new videos will be picked up automatically'
                : 'Pipeline stopped — click Start to resume'}
          </p>
        )}
      </div>

      {/* Queue list */}
      {queue.length === 0 ? (
        <p style={{ color: C.textMuted, textAlign: 'center', padding: '32px 0' }}>Queue is empty.</p>
      ) : (
        <div style={{
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: radius.md,
          overflow: 'hidden',
          boxShadow: shadow.card,
        }}>
          {sortedQueue.map((item, i) => (
            <div
              key={item.video_id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 18px',
                borderBottom: i < sortedQueue.length - 1 ? `1px solid ${C.border}` : 'none',
                background: item.status === 'processing' ? C.surfaceHigh : 'transparent',
                opacity: item.status === 'done' ? 0.5 : 1,
              }}
            >
              <span style={{ fontSize: 13, color: C.textPrimary }}>
                {item.video?.title || item.video_id}
              </span>
              <span style={{
                fontSize: 11, fontWeight: 700,
                color: STATUS_COLOR[item.status] || C.textMuted,
                textTransform: 'uppercase', letterSpacing: '0.07em',
              }}>
                {item.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
