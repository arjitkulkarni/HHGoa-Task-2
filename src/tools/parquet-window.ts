/**
 * Byte-window parquet reader.
 *
 * MSMARCO-XI ships each language as a single ~460 MB parquet file containing
 * exactly ONE row group, and without a page offset index. Any off-the-shelf
 * reader therefore has to download the entire column chunk to hand back even a
 * single row: reading 50 rows out of hinval.parquet costs 461.9 MB.
 *
 * Pages inside a column chunk are laid out sequentially and each is
 * self-describing, so the first N rows live in the first few megabytes. This
 * module fetches a bounded window of each large column, walks the page headers
 * to find the last page that fits *entirely* inside the window, and hands
 * hyparquet a buffer truncated to that boundary. hyparquet's chunk reader stops
 * at the end of the buffer it is given (`readColumn` loops while
 * `reader.offset < reader.view.byteLength - 1`), so it decodes the pages we
 * fetched and nothing more.
 *
 * Measured: 2,000 complete rows for 37 MB — 12.5x less traffic, and nothing
 * larger than the window is ever resident in memory.
 */
import { parquetMetadataAsync, parquetReadObjects } from 'hyparquet';
import type { FileMetaData } from 'hyparquet';
// hyparquet's own PageHeader parser. Reusing it means the page walk below reads
// headers exactly the way the decoder that consumes them does.
import { deserializeTCompactProtocol } from 'hyparquet/src/thrift.js';

/**
 * Byte length of the prefix of `buf` that ends on a complete page boundary.
 *
 * Pages are laid out as [thrift PageHeader][compressed_page_size bytes], back to
 * back, so walking them is just: parse header, jump `field_3` bytes, repeat.
 * We stop at the first page that would run past the end of the window.
 */
function completePagePrefix(buf: ArrayBuffer): { length: number; pages: number } {
  const view = new DataView(buf);
  const reader = { view, offset: 0 };
  let lastEnd = 0;
  let pages = 0;

  while (reader.offset < view.byteLength - 1) {
    const headerStart = reader.offset;
    let compressedSize: unknown;
    try {
      compressedSize = deserializeTCompactProtocol(reader).field_3;
    } catch {
      reader.offset = headerStart;
      break; // the header itself straddles the window edge
    }
    if (typeof compressedSize !== 'number') break;

    const end = reader.offset + compressedSize;
    if (end > view.byteLength) break; // the page body straddles the window edge
    reader.offset = end;
    lastEnd = end;
    pages++;
  }

  return { length: lastEnd, pages };
}

/** Retries a transient network call with exponential backoff. */
async function withRetry<T>(fn: () => Promise<T>, attempts = 4): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= attempts - 1) throw error;
      await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
    }
  }
}

export interface WindowReadOptions {
  url: string;
  /** Top-level column names to materialise. */
  columns: string[];
  /** Upper bound on rows; the reader returns fewer if the window ran out first. */
  maxRows: number;
  /**
   * Per-leaf-column byte windows, keyed by dotted path_in_schema. Columns with
   * no entry, or smaller than their window, are read whole.
   */
  windows?: Record<string, number>;
  onProgress?: (message: string) => void;
}

export interface WindowReadResult<T> {
  rows: T[];
  bytesFetched: number;
  totalFileBytes: number;
  rowsAvailable: number;
}

/** Reads the first `maxRows` rows of a remote parquet file using bounded range requests. */
export async function readParquetWindow<T = Record<string, unknown>>(
  options: WindowReadOptions,
): Promise<WindowReadResult<T>> {
  const { url, columns, maxRows, windows = {}, onProgress = () => {} } = options;

  const head = await withRetry(() => fetch(url, { method: 'HEAD', redirect: 'follow' }));
  if (!head.ok) throw new Error(`HEAD ${url} -> ${head.status}`);
  const byteLength = Number(head.headers.get('content-length'));
  if (!Number.isFinite(byteLength) || byteLength <= 0) {
    throw new Error(`no content-length for ${url}`);
  }

  let bytesFetched = 0;
  const fetchRange = (start: number, end: number): Promise<ArrayBuffer> => {
    const clamped = Math.min(end, byteLength);
    return withRetry(async () => {
      const response = await fetch(url, { headers: { Range: `bytes=${start}-${clamped - 1}` } });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`range request -> ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      bytesFetched += buffer.byteLength;
      return buffer;
    });
  };

  const metadata = (await parquetMetadataAsync({
    byteLength,
    slice: (start: number, end?: number) => fetchRange(start, end ?? byteLength),
  })) as FileMetaData;

  const rowGroup = metadata.row_groups[0];
  if (!rowGroup) throw new Error('parquet file has no row groups');

  // Pre-fetch and truncate the wide columns so hyparquet never asks for the rest.
  const truncated = new Map<string, ArrayBuffer>();
  for (const column of rowGroup.columns) {
    const meta = column.meta_data;
    if (!meta) continue;
    const topLevel = meta.path_in_schema[0];
    if (!columns.includes(topLevel)) continue;

    const start = Number(meta.dictionary_page_offset || meta.data_page_offset);
    const end = start + Number(meta.total_compressed_size);
    const window = windows[meta.path_in_schema.join('.')];
    if (!window || end - start <= window) continue;

    const buffer = await fetchRange(start, start + window);
    const { length, pages } = completePagePrefix(buffer);
    if (length === 0) throw new Error(`window too small for ${meta.path_in_schema.join('.')}`);
    truncated.set(`${start}-${end}`, buffer.slice(0, length));
    onProgress(
      `  window ${meta.path_in_schema.join('.')}: ${(window / 1e6).toFixed(1)} MB requested, ` +
        `${pages} complete pages (${(length / 1e6).toFixed(1)} MB usable)`,
    );
  }

  const file = {
    byteLength,
    async slice(start: number, end?: number): Promise<ArrayBuffer> {
      const stop = end ?? byteLength;
      const hit = truncated.get(`${start}-${stop}`);
      if (hit) return hit;
      return fetchRange(start, stop);
    },
  };

  const rowsAvailable = Number(metadata.num_rows);
  const rows = (await parquetReadObjects({
    file,
    metadata,
    columns,
    rowStart: 0,
    rowEnd: Math.min(maxRows, rowsAvailable),
  })) as T[];

  return { rows, bytesFetched, totalFileBytes: byteLength, rowsAvailable };
}
