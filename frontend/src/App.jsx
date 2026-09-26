import React, { useState, useRef } from 'react'
import SearchBar from './components/SearchBar'
import FilterPanel from './components/FilterPanel'
import ResultsGallery from './components/ResultsGallery'
import AnswerCard from './components/AnswerCard'
import ProcessingQueue from './components/ProcessingQueue'
import AddInstructional from './components/AddInstructional'
import LabelReview from './components/LabelReview'
import PerformanceMonitor from './components/PerformanceMonitor'
import TechFlowchart from './components/TechFlowchart'
import YouTubeIngest from './components/YouTubeIngest'
import useSearch from './hooks/useSearch'
import useAnswer from './hooks/useAnswer'
import { C } from './theme'

function mergeFilters(panelFilters, chipFilters) {
  return {
    positions:  [...new Set([...(panelFilters.positions  || []), ...(chipFilters.positions  || [])])],
    intents:    [...new Set([...(panelFilters.intents    || []), ...(chipFilters.intents    || [])])],
    techniques: [...new Set([...(panelFilters.techniques || []), ...(chipFilters.techniques || [])])],
    context:    panelFilters.context    || chipFilters.context    || null,
    gi_nogi:    panelFilters.gi_nogi    || chipFilters.gi_nogi    || null,
    instructor: panelFilters.instructor || null,
  }
}

function getOrCreateSessionId() {
  const key = 'jitsudb_session_id'
  let id = sessionStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem(key, id)
  }
  return id
}

const SESSION_ID = getOrCreateSessionId()

export default function App() {
  const [view, setView] = useState('search')
  const [panelFilters, setPanelFilters] = useState({
    positions: [], intents: [], techniques: [],
    context: null, gi_nogi: null, instructor: null,
  })
  const [lastQuery, setLastQuery]   = useState('')
  const [lastChips, setLastChips]   = useState({})
  const { results, detectedFilters, loading, search } = useSearch()
  const { answer, citations, isLoading: answerLoading, error: answerError, faithfulness, ask, clear: clearAnswer } = useAnswer()
  const resultRefs = useRef({})

  function handleSearch(query, chipFilters) {
    setLastQuery(query)
    setLastChips(chipFilters)
    search(query, mergeFilters(panelFilters, chipFilters), SESSION_ID)
    if (query.trim()) {
      ask(query)
    } else {
      clearAnswer()
    }
  }

  function handleFilterChange(updated) {
    setPanelFilters(updated)
    if (lastQuery) search(lastQuery, mergeFilters(updated, lastChips), SESSION_ID)
  }

  function scrollToChunk(chunkId) {
    const el = resultRefs.current[chunkId]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <div style={{ minHeight: '100vh', background: C.bg }}>
      {/* Nav */}
      <nav style={{
        background: C.nav,
        borderBottom: `1px solid ${C.border}`,
        padding: '0 24px',
        height: 52,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        <span style={{
          fontWeight: 800,
          fontSize: 18,
          letterSpacing: '-0.5px',
          color: C.white,
          marginRight: 24,
        }}>
          Jitsu<span style={{ color: C.blue }}>DB</span>
        </span>

        {['search', 'queue', 'add', 'youtube', 'labels', 'metrics', 'arch'].map(v => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              background: view === v ? C.surfaceHigh : 'transparent',
              border: 'none',
              padding: '6px 14px',
              borderRadius: 6,
              fontSize: 14,
              fontWeight: 500,
              color: view === v ? C.white : C.textSecondary,
              transition: 'all 0.15s',
              cursor: 'pointer',
            }}
          >
            {v === 'search' ? 'Search'
              : v === 'queue' ? 'Processing Queue'
              : v === 'add' ? 'Add Instructional'
              : v === 'youtube' ? 'YouTube'
              : v === 'labels' ? 'Label Review'
              : v === 'metrics' ? 'Performance'
              : 'Architecture'}
          </button>
        ))}
      </nav>

      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '24px 24px' }}>
        {view === 'search' && (
          <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
            <aside style={{ width: 220, flexShrink: 0 }}>
              <FilterPanel filters={panelFilters} onChange={handleFilterChange} />
            </aside>
            <main style={{ flex: 1, minWidth: 0 }}>
              <SearchBar onSearch={handleSearch} detectedFilters={detectedFilters} />
              <AnswerCard
                answer={answer}
                citations={citations}
                isLoading={answerLoading}
                error={answerError}
                onCitationClick={scrollToChunk}
                query={lastQuery}
                faithfulness={faithfulness}
              />
              <ResultsGallery results={results} loading={loading} resultRefs={resultRefs} sessionId={SESSION_ID} />
            </main>
          </div>
        )}

        {view === 'queue' && <ProcessingQueue />}
        {view === 'add' && <AddInstructional />}
        {view === 'youtube' && <YouTubeIngest />}
        {view === 'labels' && <LabelReview />}
        {view === 'metrics' && <PerformanceMonitor />}
        {view === 'arch'    && <TechFlowchart />}

      </div>
    </div>
  )
}
