# notes_bot

A local notes RAG assistant that:
- indexes `.md` and `.txt` files from a configured notes directory
- stores embeddings in ChromaDB
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
  - `chromadb`
  - `pyyaml`

## Setup

From project root:

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install openai chromadb pyyaml
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

## Configure

Edit `config.yaml`:

- `doc_root`: folder containing notes to index
- `include_ext`: file extensions to index
- `data_dir`, `index_dir`, `manifest_path`, `chat_history_path`: writable local paths
- `embedding_model`, `chat_model`: OpenAI models
- `max_file_size_mb`: skip files larger than this size during indexing (prevents OOM on huge text files)
- `max_chunks_per_file`: cap chunks per file to bound indexing memory/runtime

Current default data paths are project-local under:

`./data`

The checked-in `config.yaml` is currently set to:

- `doc_root: Y:\TextSync`
- project-local writable state under `./data`

## Run

From project root:

```bash
venv\Scripts\python.exe run_chat.py
```

## MCP Server

This repo also includes a full MCP server over stdio:

```bash
venv\Scripts\python.exe run_mcp.py
```

Optional flags:

```bash
venv\Scripts\python.exe run_mcp.py --config config.yaml --no-background-index
```

Exposed MCP tools:

- `list_indexed_files`
- `find_files`
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
- `/find <term>` search indexed files by filename or text
- `/findname <term>` search filename only
- `/findtext <term>` search text content only
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
  - Install dependencies in your active venv.

- `attempt to write a readonly database` (Chroma/SQLite)
  - Use writable paths in `config.yaml` for `data_dir`, `index_dir`, and `manifest_path`.

- `can't open file .../src/notes_bot/run_chat.py`
  - Run the root script: `python3 run_chat.py` (from project root).

## Notes

- First run may show no indexed files until background indexing runs or you execute `/reindex`.
- Generic Q&A uses top-k retrieval and may not be exhaustive for file discovery; use `/find*` commands for that.
