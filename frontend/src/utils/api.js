const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}

export function searchVideos(query, filters = {}, sessionId = null) {
  const params = new URLSearchParams({ q: query })
  if (filters.positions?.length) filters.positions.forEach(p => params.append('position', p))
  if (filters.intents?.length) filters.intents.forEach(i => params.append('intent', i))
  if (filters.techniques?.length) filters.techniques.forEach(t => params.append('technique', t))
  if (filters.context) params.set('context', filters.context)
  if (filters.gi_nogi) params.set('gi_nogi', filters.gi_nogi)
  if (filters.instructor) params.set('instructor', filters.instructor)
  if (sessionId) params.set('session_id', sessionId)
  return request(`/search?${params}`)
}

export function fetchVideos() {
  return request('/videos')
}

export function fetchQueue() {
  return request('/queue')
}

export function enqueueVideos(videoIds, priority = 0) {
  return request('/queue', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_ids: videoIds, priority }),
  })
}

export function putQueuePriority(videoId, priority) {
  return request(`/queue/${videoId}/priority`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ priority }),
  })
}

export function postPipelineAction(action) {
  return request(`/process/${action}`, { method: 'POST' })
}

export function clearQueue() {
  return fetch('/api/queue', { method: 'DELETE' })
}

export function retryFailed() {
  return request('/process/retry-failed', { method: 'POST' })
}

export function fetchAvailableFolders() {
  return request('/instructionals/available')
}

export function seedFolder(folderPath, autoStart = true) {
  return request('/instructionals/seed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_path: folderPath, auto_start: autoStart }),
  })
}

export function fetchLabelStats() {
  return request('/labels/stats')
}

export function fetchReviewQueue(limit = 20) {
  return request(`/labels/review?limit=${limit}`)
}

export function fetchDriftLog(limit = 50) {
  return request('/labels/drift?limit=' + limit)
}

export function saveLabel(chunkId, correction) {
  return request(`/labels/${chunkId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(correction),
  })
}

export function triggerBackfill() {
  return request('/labels/backfill', { method: 'POST' })
}

export function triggerClustering() {
  return request('/labels/cluster', { method: 'POST' })
}

export function fetchProposal(chunkId) {
  return request(`/labels/propose/${chunkId}`)
}

export function submitYouTube(url, gi_nogi, instructor, primary_category) {
  return request('/youtube', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, gi_nogi, instructor, primary_category }),
  })
}

export function fetchYouTubeStatus(downloadId) {
  return request(`/youtube/${downloadId}`)
}

export function fetchYouTubeJobs() {
  return request('/youtube')
}

export function fetchRecommendations(refresh = false) {
  return request(`/youtube/recommendations${refresh ? '?refresh=true' : ''}`)
}

export function confirmRecommendation(rec, gi_nogi, instructor, primary_category) {
  return request('/youtube/recommendations/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      yt_video_id: rec.yt_video_id,
      url: rec.url,
      title: rec.title,
      uploader: rec.uploader,
      gi_nogi,
      instructor,
      primary_category,
    }),
  })
}

export function dismissRecommendation(rec) {
  return request('/youtube/recommendations/dismiss', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ yt_video_id: rec.yt_video_id, title: rec.title, channel: rec.uploader }),
  })
}

export function recordClick(sessionId, chunkId) {
  return request('/feedback/click', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, chunk_id: chunkId }),
  }).catch(() => {})
}
