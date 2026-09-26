import React, { useState } from 'react'
import VideoPlayer from './VideoPlayer'
import { recordClick } from '../utils/api'
import { C, radius, shadow } from '../theme'

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function ResultCard({ result, sessionId = null }) {
  const [showPlayer, setShowPlayer] = useState(false)
  const [thumbError, setThumbError] = useState(false)
  const { chunk, video, snippet, relevance_score, visual_score, retrieved_by } = result

  function handlePlay() {
    if (sessionId) recordClick(sessionId, chunk.id)
    setShowPlayer(true)
  }

  return (
    <>
      <div style={{
        display: 'flex',
        gap: 16,
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: radius.md,
        marginBottom: 12,
        overflow: 'hidden',
        boxShadow: shadow.card,
        transition: 'border-color 0.15s',
      }}>
        {/* Thumbnail */}
        <div
          onClick={handlePlay}
          style={{
            width: 200,
            flexShrink: 0,
            background: C.black,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          {!thumbError ? (
            <img
              src={`/api/thumbnail/${chunk.id}`}
              alt=""
              onError={() => setThumbError(true)}
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            />
          ) : (
            <div style={{
              width: '100%',
              aspectRatio: '16/9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 28,
              color: C.textMuted,
            }}>
              ▶
            </div>
          )}
          {/* Play overlay */}
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(0,0,0,0)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }} />
        </div>

        {/* Details */}
        <div style={{ flex: 1, padding: '14px 16px 14px 0', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
            <span style={{ fontWeight: 700, fontSize: 15, color: C.textPrimary }}>
              {video.title}
            </span>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
              {retrieved_by === 'visual' && (
                <span title={`Visual match — CLIP similarity ${Math.round(visual_score * 100)}%`} style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: C.purple,
                  background: `${C.purple}22`,
                  border: `1px solid ${C.purple}66`,
                  borderRadius: radius.sm,
                  padding: '2px 7px',
                }}>
                  visual
                </span>
              )}
              {retrieved_by === 'both' && (
                <span title={`Text + visual match — visual ${Math.round(visual_score * 100)}%`} style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: C.brownLight,
                  background: `${C.brown}22`,
                  border: `1px solid ${C.brown}66`,
                  borderRadius: radius.sm,
                  padding: '2px 7px',
                }}>
                  text+visual
                </span>
              )}
              <span style={{
                fontSize: 11,
                fontWeight: 600,
                color: C.textMuted,
                background: C.surfaceHigh,
                border: `1px solid ${C.border}`,
                borderRadius: radius.sm,
                padding: '2px 7px',
              }}>
                {Math.round(relevance_score * 100)}%
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 12, marginBottom: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 12, color: C.purpleLight, fontWeight: 500 }}>
              {video.instructor}
            </span>
            <span style={{ fontSize: 12, color: C.brownLight, fontWeight: 500 }}>
              {formatTime(chunk.start_time)} – {formatTime(chunk.end_time)}
            </span>
            <span style={{
              fontSize: 11,
              padding: '1px 7px',
              borderRadius: radius.pill,
              border: `1px solid ${C.border}`,
              color: C.textMuted,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}>
              {video.gi_nogi}
            </span>
          </div>

          <p style={{
            fontSize: 13,
            color: C.textSecondary,
            lineHeight: 1.6,
            margin: 0,
            overflow: 'hidden',
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
          }}>
            {snippet}
          </p>

          <button
            onClick={handlePlay}
            style={{
              marginTop: 12,
              padding: '5px 14px',
              fontSize: 12,
              fontWeight: 600,
              background: 'transparent',
              border: `1px solid ${C.blue}`,
              color: C.blue,
              borderRadius: radius.sm,
              letterSpacing: '0.04em',
            }}
          >
            ▶ Preview
          </button>
        </div>
      </div>

      {showPlayer && (
        <VideoPlayer
          chunkId={chunk.id}
          startTime={chunk.start_time}
          onClose={() => setShowPlayer(false)}
        />
      )}
    </>
  )
}
