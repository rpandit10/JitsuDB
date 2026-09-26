import React from 'react'
import { C, radius, shadow } from '../theme'

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Split answer into sentences matching the backend's boundary detection
function splitSentences(text) {
  const parts = []
  let last = 0
  const re = /(?<=[.?!])\s+/g
  let m
  while ((m = re.exec(text)) !== null) {
    parts.push(text.slice(last, m.index + 1).trimStart())
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last).trimStart())
  return parts.filter(Boolean)
}

// Render answer text, converting [1], [2] markers into styled citation badges
// and appending ⚠ after sentences that are not entailed by source chunks.
function AnswerText({ text, isLoading, faithfulness = {} }) {
  const hasFaithfulness = Object.keys(faithfulness).length > 0

  if (hasFaithfulness) {
    const sentences = splitSentences(text)
    return (
      <span>
        {sentences.map((sentence, idx) => {
          const verdict = faithfulness[idx]
          const warn = verdict === 'NEUTRAL' || verdict === 'CONTRADICTED'
          const parts = sentence.split(/(\[\d+\])/g)
          return (
            <span key={idx}>
              {parts.map((part, j) => {
                const match = part.match(/^\[(\d+)\]$/)
                if (match) {
                  return (
                    <sup key={j} style={{
                      display: 'inline-block',
                      fontSize: 9,
                      fontWeight: 700,
                      color: C.purple,
                      background: `${C.purple}22`,
                      border: `1px solid ${C.purple}55`,
                      borderRadius: radius.sm,
                      padding: '0 4px',
                      margin: '0 1px',
                      verticalAlign: 'super',
                      lineHeight: 1.6,
                    }}>{match[1]}</sup>
                  )
                }
                return <span key={j}>{part}</span>
              })}
              {warn && (
                <span
                  title={verdict === 'CONTRADICTED' ? 'Contradicts source' : 'Not directly supported by source'}
                  style={{
                    marginLeft: 3,
                    fontSize: 11,
                    cursor: 'help',
                    opacity: 0.85,
                  }}
                >⚠</span>
              )}{' '}
            </span>
          )
        })}
        {isLoading && (
          <span style={{
            display: 'inline-block',
            width: 2,
            height: '0.9em',
            background: C.purple,
            marginLeft: 2,
            verticalAlign: 'middle',
            animation: 'cursor-blink 0.8s step-end infinite',
          }} />
        )}
      </span>
    )
  }

  // No faithfulness data — plain rendering
  const parts = text.split(/(\[\d+\])/g)
  return (
    <span>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/)
        if (match) {
          return (
            <sup key={i} style={{
              display: 'inline-block',
              fontSize: 9,
              fontWeight: 700,
              color: C.purple,
              background: `${C.purple}22`,
              border: `1px solid ${C.purple}55`,
              borderRadius: radius.sm,
              padding: '0 4px',
              margin: '0 1px',
              verticalAlign: 'super',
              lineHeight: 1.6,
            }}>
              {match[1]}
            </sup>
          )
        }
        return <span key={i}>{part}</span>
      })}
      {isLoading && (
        <span style={{
          display: 'inline-block',
          width: 2,
          height: '0.9em',
          background: C.purple,
          marginLeft: 2,
          verticalAlign: 'middle',
          animation: 'cursor-blink 0.8s step-end infinite',
        }} />
      )}
    </span>
  )
}

export default function AnswerCard({ answer, citations, isLoading, error, onCitationClick, query, faithfulness = {} }) {
  const visible = isLoading || answer || error

  if (!visible) return null

  return (
    <>
      <style>{`
        @keyframes cursor-blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes shimmer {
          0% { background-position: -200% center }
          100% { background-position: 200% center }
        }
      `}</style>
      <div style={{
        background: C.surface,
        border: `1px solid ${C.purple}55`,
        borderRadius: radius.md,
        marginBottom: 20,
        overflow: 'hidden',
        boxShadow: `0 0 24px ${C.purple}18`,
      }}>
        {/* Header */}
        <div style={{
          padding: '10px 16px',
          borderBottom: `1px solid ${C.border}`,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: `${C.purple}11`,
        }}>
          <span style={{ fontSize: 14 }}>✦</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <span style={{
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: C.purpleLight,
            }}>
              Summary
            </span>
            {query && (
              <span style={{
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '0.08em',
                color: C.purpleLight,
                fontStyle: 'italic',
              }}>
                {query}
              </span>
            )}
          </div>
          {isLoading && (
            <span style={{
              marginLeft: 'auto',
              fontSize: 11,
              color: C.textMuted,
              background: `${C.purple}22`,
              padding: '2px 8px',
              borderRadius: radius.pill,
              border: `1px solid ${C.purple}44`,
            }}>
              Synthesizing…
            </span>
          )}
        </div>

        {/* Answer body */}
        <div style={{ padding: '14px 16px' }}>
          {error ? (
            <p style={{ margin: 0, color: C.danger, fontSize: 13 }}>{error}</p>
          ) : (
            <p style={{
              margin: 0,
              fontSize: 14,
              lineHeight: 1.75,
              color: answer ? C.textPrimary : C.textMuted,
            }}>
              {answer
                ? <AnswerText text={answer} isLoading={isLoading} faithfulness={faithfulness} />
                : isLoading
                  ? <span style={{
                      display: 'inline-block',
                      width: '60%',
                      height: 14,
                      borderRadius: 4,
                      background: `linear-gradient(90deg, ${C.surfaceHigh} 25%, ${C.border} 50%, ${C.surfaceHigh} 75%)`,
                      backgroundSize: '200% auto',
                      animation: 'shimmer 1.2s linear infinite',
                    }} />
                  : null
              }
            </p>
          )}
        </div>

        {/* Citations */}
        {citations.length > 0 && (
          <div style={{
            padding: '10px 16px',
            borderTop: `1px solid ${C.border}`,
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
          }}>
            <span style={{ fontSize: 11, color: C.textMuted, alignSelf: 'center', marginRight: 4 }}>
              Sources:
            </span>
            {citations.map(c => (
              <button
                key={c.index}
                onClick={() => onCitationClick && onCitationClick(c.chunk_id)}
                title={`${c.video_title} at ${formatTime(c.start_time)}`}
                style={{
                  fontSize: 11,
                  padding: '2px 8px',
                  borderRadius: radius.pill,
                  border: `1px solid ${C.purple}55`,
                  color: C.purpleLight,
                  background: `${C.purple}15`,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  maxWidth: 200,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                [{c.index}] {c.video_title.replace('Attacking The Kimura — ', '')} {formatTime(c.start_time)}
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
