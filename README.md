# JitsuDB

<!-- One-paragraph pitch: what it is, who it's for, why it exists. -->

<!-- Suggested: a screenshot or short GIF of the search UI here. -->

## Features

<!-- Bullet the things a reader can't guess from the name. Candidates:
     - semantic + visual (CLIP) hybrid search over video transcripts
     - grounded answers with citations via a local LLM
     - BJJ-specific taxonomy classification and transcript correction
     - active-learning loop: label review, click feedback, drift detection
     - runs entirely offline on your own machine -->

## How it works

<!-- The pipeline, end to end. The TechFlowchart component in the UI already
     lays this out; mirror it here in prose or a diagram:
     video -> audio -> transcribe -> correct -> chunk -> classify -> embed
     -> thumbnail -> visual embed -> graph edges

     Then the query path: preprocess -> HyDE -> retrieve -> rerank
     -> ColBERT rescore -> beam-walk expansion -> answer synthesis -->

## Requirements

<!-- ffmpeg, Python 3.11+, Ollama, an NVIDIA GPU (and what degrades without one).
     Say which parts are optional. -->

## Installation

<!-- uv / pip install steps. Note the pyproject extras: [dev], [eval]. -->

## Configuration

<!-- Copy config.example.yaml to config.yaml, then set paths.video_library.
     Call out the settings most people will actually change:
     transcription.model_size, the device fields, rag.model,
     and the feature toggles (reranker, visual_embedding, hyde, colbert). -->

## Library layout

<!-- The folder structure seed_catalog.py expects:
       <library>/<NN-Category>/<Instructor - Series Title>/<part>.mp4
     And the optional Index.md catalogue at the library root, which supplies
     canonical instructor/series names and gi/nogi. See Index.example.md. -->

## Usage

<!-- Start the backend, start the frontend, then: seed a folder, watch the
     processing queue, search. Mention the scripts/ helpers worth knowing. -->

## Evaluation

<!-- What eval/ does and what the metrics mean. Note that the golden set is
     library-specific, so users build their own from the .example.jsonl files. -->

## Project layout

<!-- Short tour of the top-level directories: jitsudb/, frontend/, scripts/,
     eval/, tests/. One line each. -->

## Notes and limitations

<!-- Worth stating plainly:
     - the taxonomy is BJJ-specific and hand-built
     - you supply your own video files; none are distributed
     - the server binds loopback only and has no authentication -->

## License

<!-- Pick one. -->
