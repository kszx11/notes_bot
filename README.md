# notes_bot

A local notes RAG assistant that:
- indexes `.md` and `.txt` files from a configured notes directory
- stores embeddings in ChromaDB
- supports natural-language note search across file names and note content
- supports general world-knowledge chat when the prompt is not about your notes

## Features

- Incremental background indexing
- Manual reindex command
- Natural-language routing between note search and general chat
- Chat history persistence
- Ranked search results with file paths and snippets
- Direct note opening by filename or prior search-result number
- Chunk metadata includes file/title context to improve vague searches after reindexing

## Project Layout

```text
notes_bot/
  run_chat.py
  run_eval_candidates.py
  run_eval_promote.py
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
- `route_query`
- `query_notes`
- `chat`
- `eval_candidates`
- `eval_promote`
- `list_large_files`

Legacy MCP compatibility tools still exposed:

- `find_files`
- `search_notes`
- `answer_from_notes`

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
- `/reindex-force` rebuild the entire index now
- `/reindex-status` show current indexer status
- `/indexed` list indexed files from manifest
- `/exit` quit

## Eval Runner

Run local search evals against your current index:

```bash
python run_eval.py
```

Optional arguments:

```bash
python run_eval.py data/eval_queries.json 8
```

Generate candidate eval cases from real `notes_search` usage:

```bash
python run_eval_candidates.py
```

Optional arguments:

```bash
python run_eval_candidates.py data/eval_candidates.json 40
```

Promote reviewed candidates into the main eval file:

```bash
python run_eval_promote.py 1 2 5-8
```

Promote all candidates:

```bash
python run_eval_promote.py all
```

The starter eval file lives at:

`data/eval_queries.json`

Edit it with real queries you care about and the expected `rel_path` values you want to see in the top results.

Optional eval fields per case:

- `expected_query_type`: `filename_focus`, `snippet_focus`, or `mixed`
- `expected_snippet_terms`: list of lowercase terms you expect to appear in a matching snippet

Example:

```json
{
  "query": "where did I mention pipelines and QA",
  "expected_paths": ["Pipelines and QA.txt"],
  "expected_query_type": "snippet_focus",
  "expected_snippet_terms": ["pipelines", "qa"]
}
```

Candidate generation workflow:

- normal note searches are logged to `data/search_queries.jsonl`
- `python run_eval_candidates.py` reads that log
- it skips queries already present in `data/eval_queries.json`
- it writes a deduped seed file you can review and promote into the main eval set
- `python run_eval_promote.py 1 2 5-8` promotes selected candidate indexes into `data/eval_queries.json`
- promoted candidates are removed from `data/eval_candidates.json`

## Natural-Language Queries Supported

- `where did I mention wazuh?`
- `find my note about backups`
- `show me the contents of Twingate.md`
- `show me the first one`
- `notes with docker in the title`
- `what files have been indexed?`
- `what is the difference between TCP and UDP?`

## Query Behavior

- Queries clearly about your files or prior notes are routed to natural-language note search.
- Explicit filename requests such as `show me the contents of Twingate.md` open the indexed note directly.
- Follow-up requests such as `show me the first one` open the corresponding item from the most recent search result list.
- Other prompts are routed to general chat using the configured chat model.
- Note search returns ranked file and snippet matches instead of requiring `/find*` commands.
- Reindex after upgrading so new heading/title metadata is stored and used in ranking.

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
- Use `/reindex-force` when you want to rebuild every indexed file, even if the files themselves did not change.
- Use `/reindex-status` to check live progress while a long background or manual reindex is running.
- Filename-focused searches are boosted directly from the indexed manifest so title/path queries do not depend entirely on semantic retrieval.
- Use `python run_eval.py` after ranking changes so you can compare hit rates instead of tuning by feel.
- Query embeddings are cached locally in `data/query_embedding_cache.sqlite`, so repeated searches and eval runs reuse prior embeddings instead of calling the API again for the same query text.
- Natural-language note searches are logged locally in `data/search_queries.jsonl`, which you can convert into draft eval cases with `python run_eval_candidates.py`.
