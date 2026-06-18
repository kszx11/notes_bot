# notes_bot

A local notes RAG assistant that:
- indexes `.md` and `.txt` files from a configured notes directory
- stores embeddings in ChromaDB when available, otherwise a built-in SQLite vector store
- answers questions using only retrieved source snippets
- supports file discovery/search commands for indexed content

## Features

- Incremental background indexing
- Manual reindex command
- Grounded answers with citation validation
- Chat history persistence
- File search by filename and/or file text

## Project Layout

```text
notes_bot/
  run_chat.py
  config.yaml
  src/notes_bot/
    chat.py
    config.py
    indexer.py
    store.py
    manifest.py
    scanner.py
    chunker.py
    prompt.py
    validate.py
    history.py
    hybrid.py
```

## Requirements

- Python 3.10+ (tested with Python 3.14)
- OpenAI API key
- Python packages:
  - `openai`
  - `pyyaml`

## Setup

From project root:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional:

```bash
python -m pip install chromadb
```

If `chromadb` is not installed, the app falls back to a built-in SQLite-backed vector store.

Set your API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Configure

Edit `config.yaml`:

- `doc_root`: single folder containing notes to index
- `doc_roots`: optional list of multiple note roots; each item can be a path or `{name, path}`
- `include_ext`: file extensions to index
- `data_dir`, `index_dir`, `manifest_path`, `chat_history_path`: writable local paths
- `embedding_model`, `chat_model`: OpenAI models
- `scan_interval_minutes`: cadence for background indexing passes
- `background_index_start_delay_seconds`: delay before the first background indexing pass after startup
- `retry_error_after_minutes`: how long background indexing waits before retrying files previously marked `error`
- `answer_context_before_lines`, `answer_context_after_lines`: expand grounded answers with surrounding note lines around each hit
- `adjacent_chunk_window`: merge neighboring retrieved chunks from the same note before showing passages or answering
- `max_file_size_mb`: skip files larger than this size during indexing (prevents OOM on huge text files)
- `max_chunks_per_file`: cap chunks per file to bound indexing memory/runtime

Current default paths are project-local:

- `doc_root: ./notes`
- writable state under `./data`

Example multi-root config:

```yaml
doc_roots:
  - name: work
    path: /home/you/work-notes
  - name: personal
    path: /home/you/personal-notes
```

When multiple roots are configured, indexed file paths are exposed as `<root>/<path>`, for example `work/runbooks/db.md`.

## Run

From project root:

```bash
python run_chat.py
```

Optional flags:

```bash
python run_chat.py --config config.yaml --no-background-index
```

## MCP Server

This repo also includes a full MCP server over stdio:

```bash
python run_mcp.py
```

Optional flags:

```bash
python run_mcp.py --config config.yaml --no-background-index
```

Exposed MCP tools:

- `list_indexed_files`
- `list_file_status`
- `find_files`
- `route_query`
- `search_notes`
- `get_note`
- `reindex_status`
- `reindex_now`
- `answer_from_notes`
- `list_large_files`

`reindex_now` is asynchronous by default for MCP clients. Call `reindex_status` to poll progress, or pass `{"wait": true}` if you explicitly want a blocking reindex run.

Exposed MCP resources:

- `notes://manifest`
- `notes://config`
- `notes://reindex/status`

Exposed MCP prompts:

- `grounded_note_answer`

Example Codex MCP entry:

```toml
[mcp_servers.notes_bot]
command = 'C:\Users\kspringall\code\notes_bot\venv\Scripts\python.exe'
args = ['C:\Users\kspringall\code\notes_bot\run_mcp.py', '--config', 'C:\Users\kspringall\code\notes_bot\config.yaml']
```

## Commands

- `/clear` clear chat context/history
- `/reindex` run incremental indexing now
- `/indexed` list indexed files from manifest
- `/status` show manifest status summary and recent non-indexed files
- `/find <term>` search indexed files by filename or text
- `/findfresh <term>` search indexed files but exclude stale entries
- `/findname <term>` search filename only
- `/findtext <term>` search text content only
- `/search <term>` search ranked note passages and show excerpts
- `/searchfresh <term>` search ranked passages from fresh indexed content only
- `/open <n>` open surrounding context for a result from the last `/search`
- `/open <path[:start-end]>` open a note or line range with surrounding context
- `/exit` quit

## Natural-Language Queries Supported

- `What files have been indexed?`
- `What files mention wazuh?`
- `What files mention wazuh in filename?`
- `Find files mentioning docker in text`

## Answer Behavior

For normal Q&A, the assistant is instructed to use only provided sources.  
If it cannot ground an answer in retrieved notes, fallback is:

`I can't find that in your notes.`

## Troubleshooting

- `SyntaxError: from __future__ imports must occur at the beginning...`
  - Ensure `src/notes_bot/chat.py` starts with `from __future__ import annotations`.

- `ModuleNotFoundError` for `openai`, `yaml`, or `chromadb`
  - Install dependencies in your active venv with `python -m pip install -r requirements.txt`.

- `attempt to write a readonly database` (Chroma/SQLite)
  - Use writable paths in `config.yaml` for `data_dir`, `index_dir`, and `manifest_path`.

- `can't open file .../src/notes_bot/run_chat.py`
  - Run the root script: `python3 run_chat.py` (from project root).

- `Configured doc root does not exist`
  - Create the missing directory, or update `doc_root` / `doc_roots` in `config.yaml`.

## Notes

- First run may show no indexed files until background indexing runs or you execute `/reindex`.
- Background indexing now waits briefly after startup before its first pass, and failed files are retried with backoff during automatic scans.
- Use `/search` when you want ranked excerpts before asking a full question.
- Use `/searchfresh` or MCP `freshness_mode: indexed_only` when you want to exclude stale indexed content.
- Use `/open` to expand a retrieved passage into surrounding note context.
- Use `/status` when you want to see skipped, stale, or failed files.
- Search prefers fresh indexed content over stale content, and answers will note when cited evidence is stale.
- Bare keyword-style queries in the CLI now default to passage search; question-style queries default to grounded answers.
- Retrieval merges adjacent chunks from long notes before answer synthesis, which improves results on large `.txt` source documents.
