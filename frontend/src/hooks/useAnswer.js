import { useState, useRef, useCallback } from 'react'

export default function useAnswer() {
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [faithfulness, setFaithfulness] = useState({})
  const esRef = useRef(null)

  const ask = useCallback((query) => {
    if (esRef.current) {
      esRef.current.close()
    }
    setAnswer('')
    setCitations([])
    setError(null)
    setFaithfulness({})
    setIsLoading(true)

    const es = new EventSource(`/api/answer?q=${encodeURIComponent(query)}`)
    esRef.current = es

    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'citations') {
        setCitations(data.citations)
      } else if (data.type === 'token') {
        setAnswer(prev => prev + data.content)
      } else if (data.type === 'faithfulness_update') {
        setFaithfulness(prev => ({ ...prev, [data.sentence_idx]: data.verdict }))
      } else if (data.type === 'done') {
        setIsLoading(false)
        es.close()
        esRef.current = null
      } else if (data.type === 'error') {
        setError(data.message)
        setIsLoading(false)
        es.close()
        esRef.current = null
      }
    }

    es.onerror = () => {
      setIsLoading(false)
      es.close()
      esRef.current = null
    }
  }, [])

  const clear = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setAnswer('')
    setCitations([])
    setError(null)
    setFaithfulness({})
    setIsLoading(false)
  }, [])

  return { answer, citations, isLoading, error, faithfulness, ask, clear }
}
