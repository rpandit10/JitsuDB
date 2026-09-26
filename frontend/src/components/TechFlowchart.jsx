import React from 'react'
import { C } from '../theme'

// palette
const COLORS = {
  ingest:   { bg: '#1e3a2a', border: '#3d7a55', text: '#7dd4a0' },
  storage:  { bg: '#1e2a3a', border: '#3d5f8a', text: '#7ab8e8' },
  search:   { bg: '#2a1e3a', border: '#6a3d8a', text: '#c090f0' },
  active:   { bg: '#3a2a1e', border: '#8a6a3d', text: '#e8c07a' },
  model:    { bg: '#2a1e1e', border: '#8a3d3d', text: '#f07a7a' },
  output:   { bg: '#1e2a1e', border: '#3d7a3d', text: '#88e088' },
}

// primitives
function Node({ label, color, mono = false, small = false }) {
  return (
    <div style={{
      background: color.bg,
      border: `1px solid ${color.border}`,
      borderRadius: 8,
      padding: small ? '4px 10px' : '6px 14px',
      fontSize: small ? 11 : 12,
      fontFamily: mono ? 'monospace' : 'inherit',
      color: color.text,
      whiteSpace: 'nowrap',
      textAlign: 'center',
      fontWeight: 600,
    }}>
      {label}
    </div>
  )
}

function Arrow({ horizontal = false, label = '' }) {
  if (horizontal) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <div style={{ flex: 1, height: 1, background: C.border }} />
        {label && <span style={{ fontSize: 10, color: C.textMuted, whiteSpace: 'nowrap' }}>{label}</span>}
        <div style={{ color: C.textMuted, fontSize: 12 }}>→</div>
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
      <div style={{ width: 1, height: label ? 8 : 14, background: C.border }} />
      {label && <span style={{ fontSize: 10, color: C.textMuted }}>{label}</span>}
      <div style={{ color: C.textMuted, fontSize: 12, lineHeight: 1 }}>↓</div>
    </div>
  )
}

function ColHeader({ children, color }) {
  return (
    <div style={{
      fontSize: 10,
      fontWeight: 700,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: color.text,
      padding: '4px 0',
      marginBottom: 8,
      borderBottom: `1px solid ${color.border}`,
    }}>
      {children}
    </div>
  )
}

function Col({ children, style = {} }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 4,
      minWidth: 200,
      ...style,
    }}>
      {children}
    </div>
  )
}

function Section({ title, children, style = {} }) {
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: 12,
      padding: 20,
      ...style,
    }}>
      <div style={{
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: C.textMuted,
        marginBottom: 16,
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

// groups
function IngestCol() {
  const c = COLORS.ingest
  const m = COLORS.model
  return (
    <Col>
      <ColHeader color={c}>Ingest Pipeline</ColHeader>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <Node label="Local Video File" color={c} small />
        <span style={{ fontSize: 10, color: C.textMuted }}>or</span>
        <Node label="YouTube URL" color={c} small />
      </div>
      <Arrow label="yt-dlp" />
      <Node label="yt-dlp" color={m} mono />
      <Arrow />
      <Node label="FFmpeg · audio extract" color={c} />
      <Arrow />
      <Node label="faster-whisper medium int8 CUDA" color={m} mono />
      <Arrow />
      <Node label="Glossary corrector" color={c} />
      <Arrow />
      <Node label="Chunker  45 s / 15 s overlap" color={c} />
      <Arrow />
      <Node label="NLI classifier  (taxonomy)" color={c} />
      <Arrow />
      <Node label="Entropy scoring → uncertain_chunks" color={COLORS.active} />
      <Arrow />
      <Node label="nomic-embed-text-v1.5  768d" color={m} mono />
      <Arrow label="search_document:" />
      <Node label="CLIP-ViT-B-32  512d" color={m} mono />
      <Arrow />
      <Node label="Temporal graph  chunk_edges  ±2 hops" color={c} />
    </Col>
  )
}

function StorageCol() {
  const c = COLORS.storage
  return (
    <Col>
      <ColHeader color={c}>Storage</ColHeader>
      <Node label="SQLite · WAL" color={c} />
      <div style={{ fontSize: 10, color: C.textMuted, textAlign: 'center', lineHeight: 1.6 }}>
        videos · chunks · taxonomy_labels<br />
        processing_queue · word_timestamps<br />
        uncertain_chunks · retrieval_log<br />
        drift_log · chunk_label_proposals<br />
        chunk_edges
      </div>
      <div style={{ height: 24 }} />
      <Node label="ChromaDB · cosine  (text)" color={c} mono />
      <div style={{ fontSize: 10, color: C.textMuted }}>jitsudb_chunks_v2  768d</div>
      <div style={{ height: 8 }} />
      <Node label="ChromaDB · cosine  (visual)" color={c} mono />
      <div style={{ fontSize: 10, color: C.textMuted }}>jitsudb_visual  512d</div>
      <div style={{ height: 8 }} />
      <Node label="ChromaDB · cosine  (ft)" color={c} mono />
      <div style={{ fontSize: 10, color: C.textMuted }}>text_ft  768d (post fine-tune)</div>
      <div style={{ height: 8 }} />
      <Node label="ColBERT token index  (memmap)" color={c} mono />
      <div style={{ fontSize: 10, color: C.textMuted }}>data/colbert/  per-token 768d</div>
    </Col>
  )
}

function SearchCol() {
  const c = COLORS.search
  const m = COLORS.model
  const o = COLORS.output
  return (
    <Col>
      <ColHeader color={c}>Search Pipeline</ColHeader>
      <Node label="User Query" color={o} />
      <Arrow />
      <Node label="Query preprocessor  abbrev + auto-filters" color={c} />
      <Arrow />
      <Node label="HyDE  N=3 hypotheticals → centroid" color={c} />
      <Arrow label="Ollama llama3.2:3b" />
      <Node label="nomic-embed  query  768d" color={m} mono />
      <Arrow label="search_query:" />
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
        <Col style={{ minWidth: 0 }}>
          <Node label="ANN text  ChromaDB" color={c} small />
          <Arrow />
          <Node label="top-30 candidates" color={c} small />
        </Col>
        <Col style={{ minWidth: 0 }}>
          <Node label="CLIP text encode" color={m} small />
          <Arrow />
          <Node label="ANN visual  ChromaDB" color={c} small />
        </Col>
      </div>
      <Arrow />
      <Node label="Hybrid score fusion  α=0.7 text / 0.3 visual" color={c} />
      <Arrow />
      <Node label="CrossEncoder  ms-marco-MiniLM-L-6-v2" color={m} mono />
      <Arrow />
      <Node label="Conformal prediction  threshold" color={c} />
      <Arrow />
      <Node label="Beam expansion  ±2 hops  temporal graph" color={c} />
      <Arrow />
      <Node label="ColBERT MaxSim  late interaction rescoring" color={c} />
      <Arrow />
      <Node label="Click reranker  cosine implicit feedback" color={c} />
      <Arrow />
      <Node label="SearchResult  ≤30 ranked chunks" color={o} />
      <Arrow />
      <Node label="RAG synthesis  Ollama llama3.2:3b" color={m} mono />
      <Arrow />
      <Node label="Faithfulness NLI  entailment check" color={c} />
      <Arrow />
      <Node label="SSE stream  /api/answer  /api/status" color={o} mono />
    </Col>
  )
}

function ActiveLearningRow() {
  const c = COLORS.active
  const m = COLORS.model
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      flexWrap: 'wrap',
      justifyContent: 'center',
    }}>
      <Node label="uncertain_chunks  high-entropy" color={c} />
      <Arrow horizontal />
      <Node label="HDBSCAN  clustering" color={m} mono />
      <Arrow horizontal />
      <Node label="chunk_label_proposals  majority vote" color={c} />
      <Arrow horizontal />
      <Node label="Label Review UI  human confirm" color={COLORS.output} />
      <Arrow horizontal />
      <Node label="MNRL fine-tune  click pairs" color={m} mono />
      <Arrow horizontal />
      <Node label="reembed  text_ft  collection" color={c} mono />
    </div>
  )
}

const COLOR_KEY = [
  { key: 'ingest',  label: 'Ingest stages',           desc: 'Pipeline steps that process raw input into storable form' },
  { key: 'storage', label: 'Storage / databases',      desc: 'Persistent stores — SQLite, ChromaDB, ColBERT memmap index' },
  { key: 'search',  label: 'Search pipeline',          desc: 'Query-time logic — ranking, fusion, conformal filtering' },
  { key: 'active',  label: 'Active learning',          desc: 'Feedback loop — uncertain chunks, clustering, fine-tuning' },
  { key: 'model',   label: 'ML models & libraries',   desc: 'Loaded checkpoints or external tools with specific versions' },
  { key: 'output',  label: 'User-facing I/O',          desc: 'Query inputs and ranked results returned to the UI' },
]

// main
export default function TechFlowchart() {
  return (
    <div style={{ color: C.textPrimary }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: C.white }}>
          System Architecture
        </h2>
        <p style={{ margin: '6px 0 0', fontSize: 13, color: C.textSecondary }}>
          End-to-end technical stack — ingest (local + YouTube) → storage → retrieval (HyDE · ColBERT · conformal) → active learning
        </p>
      </div>

      {/* Color key */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: 24,
        padding: '12px 16px',
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
      }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.textMuted, alignSelf: 'center', marginRight: 4 }}>
          Color key
        </span>
        {COLOR_KEY.map(({ key, label, desc }) => {
          const color = COLORS[key]
          return (
            <div
              key={key}
              title={desc}
              style={{
                background: color.bg,
                border: `1px solid ${color.border}`,
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                color: color.text,
                cursor: 'default',
              }}
            >
              {label}
            </div>
          )
        })}
      </div>

      {/* Top 3 columns */}
      <Section title="Ingest · Storage · Search">
        <div style={{
          display: 'flex',
          gap: 40,
          justifyContent: 'center',
          alignItems: 'flex-start',
          overflowX: 'auto',
          paddingBottom: 8,
        }}>
          <IngestCol />

          {/* center connectors */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 40, alignItems: 'center' }}>
            <Arrow horizontal />
            <div style={{ fontSize: 10, color: C.textMuted, textAlign: 'center', lineHeight: 1.8 }}>
              upsert<br />embeddings
            </div>
          </div>

          <StorageCol />

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 40, alignItems: 'center' }}>
            <Arrow horizontal />
            <div style={{ fontSize: 10, color: C.textMuted, textAlign: 'center', lineHeight: 1.8 }}>
              ANN<br />query
            </div>
          </div>

          <SearchCol />
        </div>
      </Section>

      <div style={{ height: 20 }} />

      {/* Active learning row */}
      <Section title="Active Learning Loop ">
        <ActiveLearningRow />
      </Section>

      <div style={{ height: 20 }} />

      {/* Key models legend */}
      <Section title="Key Models  &  Libraries">
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 10,
          justifyContent: 'center',
        }}>
          {[
            ['faster-whisper',       'Transcription',          COLORS.model],
            ['nomic-embed-text-v1.5','Text embeddings 768d',   COLORS.model],
            ['CLIP-ViT-B-32',        'Visual embeddings 512d', COLORS.model],
            ['ms-marco-MiniLM-L-6-v2','CrossEncoder rerank',   COLORS.model],
            ['llama3.2:3b (Ollama)', 'RAG · HyDE',            COLORS.model],
            ['HDBSCAN (sklearn≥1.3)','Cluster proposals',      COLORS.active],
            ['ChromaDB',             'Vector store (cosine)',  COLORS.storage],
            ['SQLite WAL',           'Structured metadata',    COLORS.storage],
            ['FastAPI + uvicorn',    'HTTP + SSE backend',     COLORS.search],
            ['React + Vite',         'Frontend',               COLORS.output],
            ['sentence-transformers','MNRL fine-tuning',       COLORS.active],
            ['FFmpeg',               'Audio extraction',       COLORS.ingest],
            ['yt-dlp',               'YouTube download',       COLORS.ingest],
            ['ColBERT (memmap)',      'Late-interaction MaxSim',COLORS.search],
          ].map(([name, role, color]) => (
            <div key={name} style={{
              background: color.bg,
              border: `1px solid ${color.border}`,
              borderRadius: 8,
              padding: '6px 12px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
              minWidth: 160,
            }}>
              <span style={{ fontSize: 12, fontWeight: 700, fontFamily: 'monospace', color: color.text }}>{name}</span>
              <span style={{ fontSize: 10, color: C.textMuted }}>{role}</span>
            </div>
          ))}
        </div>
      </Section>

      <div style={{ height: 20 }} />

      {/* Project summary */}
      <Section title="What JitsuDB Does">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 860 }}>
          {[
            {
              heading: 'Video ingestion',
              body: 'JitsuDB accepts BJJ instructional videos either as local files or YouTube URLs. yt-dlp downloads the video; FFmpeg strips the audio track. faster-whisper (medium, int8 on CUDA) transcribes each video with a BJJ-specific glossary corrector to fix common mis-transcriptions of technique names and terminology.',
            },
            {
              heading: 'Chunking & indexing',
              body: 'Transcripts are split into 45-second overlapping chunks (15 s overlap, sentence-snapping to avoid mid-sentence cuts). Each chunk gets text embeddings from nomic-embed-text-v1.5 (768d) and visual embeddings from CLIP-ViT-B-32 (512d, thumbnail frame). Embeddings are stored in two ChromaDB collections. An NLI classifier tags every chunk with positional context (top/bottom/standing/neutral) and technique taxonomy labels. A temporal graph records continuity scores between adjacent chunks so that search can expand results ±2 hops along the video timeline.',
            },
            {
              heading: 'Semantic search',
              body: 'Queries go through a preprocessor that expands abbreviations and extracts structured filters (gi/nogi, instructor). HyDE generates up to three hypothetical technique descriptions via Ollama llama3.2:3b and averages their embeddings to improve recall on short or abstract queries. Candidates are retrieved from both text and visual ChromaDB indexes, fused (70 % text / 30 % visual), then re-ranked by a CrossEncoder (ms-marco-MiniLM-L-6-v2). A conformal prediction threshold trims results to a calibrated coverage level. ColBERT MaxSim late-interaction rescoring and a cosine implicit-feedback click reranker apply final ordering.',
            },
            {
              heading: 'RAG answer synthesis',
              body: 'The top-ranked chunks are assembled into a context window and sent to Ollama llama3.2:3b to generate a streamed natural-language answer. An optional faithfulness NLI check filters out sentences not entailed by the retrieved context. The answer streams back to the UI via SSE.',
            },
            {
              heading: 'Active learning loop',
              body: 'Chunks where the NLI classifier was uncertain (high Shannon entropy) are flagged in uncertain_chunks. HDBSCAN clusters those chunks; majority-vote labeling produces chunk_label_proposals for human review in the Label Review UI. Confirmed pairs feed a Metric Learning with Neighborhood Relations (MNRL) fine-tuning run that updates the embedding model and re-embeds the corpus into a separate text_ft ChromaDB collection.',
            },
            {
              heading: 'YouTube discovery (Discover tab)',
              body: 'A recommendation scraper runs yt-dlp keyword searches (configurable tags) and optionally scrapes trusted channel URLs. It filters out any video already seen (confirmed or dismissed) via the youtube_seen table and surfaces new candidates. Users can confirm a recommendation — which queues it for download and ingest — or dismiss it so it never reappears.',
            },
            {
              heading: 'Observability',
              body: 'Every search is logged in retrieval_log with latency, rank, and click signals. The Performance tab shows P50/P95/P99 latency, per-stage search timing, system-wide VRAM usage via pynvml (covering CTranslate2, Ollama, and PyTorch), PyTorch allocator stats, conformal q-hat, and index counts (chunks, videos, CTR).',
            },
          ].map(({ heading, body }) => (
            <div key={heading}>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.textPrimary, marginBottom: 4 }}>
                {heading}
              </div>
              <div style={{ fontSize: 12, color: C.textSecondary, lineHeight: 1.7 }}>
                {body}
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  )
}
