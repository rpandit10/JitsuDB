import { useState, useEffect } from 'react'

/**
 * Subscribe to a Server-Sent Events endpoint and return the latest parsed payload.
 *
 * @param {string} url - SSE endpoint path (e.g. '/api/status')
 * @returns {object|null} Latest parsed JSON data, or null before first event
 */
export default function useSSE(url) {
  const [data, setData] = useState(null)

  useEffect(() => {
    const source = new EventSource(url)

    source.onmessage = (event) => {
      try {
        setData(JSON.parse(event.data))
      } catch {
        // Ignore malformed events
      }
    }

    source.onerror = () => {
      // Don't close — native EventSource auto-reconnects after errors
      console.warn('SSE connection lost, reconnecting...')
    }

    return () => source.close()
  }, [url])

  return data
}
