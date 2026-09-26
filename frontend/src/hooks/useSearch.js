import { useState, useCallback } from 'react'
import { searchVideos } from '../utils/api'

// Hook for executing searches against /api/search.
export default function useSearch() {
  const [results, setResults] = useState([])
  const [detectedFilters, setDetectedFilters] = useState({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const search = useCallback(async (query, filters = {}, sessionId = null) => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await searchVideos(query, filters, sessionId)
      setResults(data)
      // Aggregate detected_filters across all results for chip display
      const merged = {}
      data.forEach(r => {
        Object.entries(r.detected_filters || {}).forEach(([k, vs]) => {
          merged[k] = [...new Set([...(merged[k] || []), ...vs])]
        })
      })
      setDetectedFilters(merged)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { results, detectedFilters, loading, error, search }
}
