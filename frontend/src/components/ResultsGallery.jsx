import React from 'react'
import ResultCard from './ResultCard'
import SearchLoader from './SearchLoader'
import { C } from '../theme'

export default function ResultsGallery({ results = [], loading = false, resultRefs, sessionId = null }) {
  if (loading) return <SearchLoader />

  if (results.length === 0) return (
    <p style={{ color: C.textMuted, padding: '32px 0', textAlign: 'center' }}>
      No results. Try a different query or adjust the filters.
    </p>
  )

  return (
    <div>
      <p style={{ fontSize: 12, color: C.textMuted, marginBottom: 14 }}>
        {results.length} result{results.length !== 1 ? 's' : ''}
      </p>
      {results.map(result => (
        <div
          key={result.chunk.id}
          ref={el => { if (resultRefs) resultRefs.current[result.chunk.id] = el }}
        >
          <ResultCard result={result} sessionId={sessionId} />
        </div>
      ))}
    </div>
  )
}
