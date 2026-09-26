import { useState, useEffect, useCallback } from 'react'
import { fetchQueue, fetchVideos, postPipelineAction, putQueuePriority, retryFailed } from '../utils/api'

/**
 * Hook for managing the processing queue.
 * Queue items are enriched with their Video objects for display.
 */
export default function useQueue() {
  const [queue, setQueue] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const [queueData, videosData] = await Promise.all([fetchQueue(), fetchVideos()])
      const videoMap = Object.fromEntries(videosData.map(v => [v.id, v]))
      setQueue(queueData.map(item => ({ ...item, video: videoMap[item.video_id] || null })))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const startPipeline = () => postPipelineAction('start').then(refresh)
  const pausePipeline = () => postPipelineAction('pause').then(refresh)
  const resumePipeline = () => postPipelineAction('resume').then(refresh)
  const cancelPipeline = () => postPipelineAction('cancel').then(refresh)
  const retryFailedPipeline = () => retryFailed().then(refresh)

  const updatePriority = (videoId, priority) =>
    putQueuePriority(videoId, priority).then(refresh)

  return { queue, loading, refresh, startPipeline, pausePipeline, resumePipeline, cancelPipeline, retryFailedPipeline, updatePriority }
}
