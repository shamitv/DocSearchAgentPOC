# Wikipedia Elasticsearch Indexer

This document describes how the current version of `indexer.py` in the `wiki_es_indexer` module works. It covers configuration, parsing, processing modes, and profiling.

## 1. Configuration & Initialization

- Environment variables (loaded via `EnvLoader` in `utils.py`):
  - `ES_HOST`, `ES_PORT`: Elasticsearch connection details.
  - `ES_SEARCH_INDEX`: Target index name (required).
  - `INDEXER_MAX_WORKERS`: Maximum number of worker threads/processes (default 20).
- Logging configured through `LoggerConfig` in `utils.py`.
- `ElasticsearchHandler` handles bulk/async indexing to Elasticsearch and clean shutdown.

## 2. Wikipedia Article Parsing

The `WikipediaArticleParser` class uses `mwparserfromhell` to convert raw wikitext to plain text, handling templates:

1. **Template Processing**: Recursively extract template parameters as `name: value` lines.
2. **Markup Stripping**: Remove all remaining wiki markup.

Methods:
- `_process_templates(wikicode)`: Walks and replaces templates.
- `parse_article_text(raw_text)`: Parses and returns plain text or empty string on error.

## 3. Page Extraction & Indexing

### 3.1 `extract_page(page)`

- Skips redirects and non-article namespaces (`namespace > 0`).
- Returns a dict with `page_id`, `title`, `raw_text`, `revision_id`, and `timestamp`, or `None` if skipped.

### 3.2 `process_page_for_mp(page_data, parser)`

- Tracks metrics: total pages, parsing time, max parse time/title, skipped count.
- Parses wikitext, logs slow parses (>1s).
- Builds document with fields: `title`, `text`, `metadata`, `indexed_on`.
- Indexes via `es_handler.index_document` and logs slow indexing (>0.5s).
- Periodically logs metrics every 1,000 pages.

### 3.3 `process_page_str(page_xml, parser)`

- XML‐string variant for streaming mode: parses a `<page>` snippet via `xml.etree.ElementTree`.
- Applies same skip rules and delegates to `process_page_for_mp`.

## 4. Dump Processing Modes

### 4.1 Batch + ProcessPool (`process_dump`)

```bash
python indexer.py <dump_file>
```

- Uses `IndexedBzip2File` for random-access decompression.
- Loads `mwxml.Dump` iterator.
- Spawns a `ProcessPoolExecutor` (up to CPU×2 or 20 workers).
- Reads pages in batches (default size 20), extracts page data, and submits `process_page_for_mp` tasks.
- Shuts down the handler and logs a performance summary (file open, processing, shutdown).

### 4.2 Streaming + ThreadPool (`process_dump_stream`)

```bash
python indexer.py --stream <dump_file>
```

- Wraps the bzip2 file in a `TextIOWrapper`.
- Reads line-by-line to collect `<page>...</page>` blocks.
- Submits each block to a `ThreadPoolExecutor`.
- Ideal for low-memory or lower-latency indexing.

## 5. Profiling

- The `@profile` decorator wraps any function with a `cProfile` session.
- Prints top stats to console and optionally dumps raw stats to a `.prof` file.
- Used on `process_dump` by default, saving to `indexer_profile.prof`.

## 6. CLI Entry Point

At the end of `indexer.py`:

```bash
Usage: python indexer.py [--stream] <dump_file>
```

- No flags: batch + multiprocess mode.
- `--stream`: streaming + multithread mode.