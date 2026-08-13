/**
 * On-disk index format.
 *
 * Five files, all flat and all mmap-friendly. Loading is a set of buffer reads
 * plus one JSON parse — no per-record object allocation for the 63k chunks,
 * which is what keeps the resident set small enough to run in a 512 MB
 * container:
 *
 *   meta.json      build provenance, dimensions, per-strategy statistics
 *   passages.json  the only copy of the corpus text (chunks are offsets into it)
 *   chunks.bin     Int32Array, 5 fields per chunk, struct-of-arrays
 *   vec_bin.bin    1 bit per dimension  — the coarse search space
 *   vec_i8.bin     8 bits per dimension — the rescoring space
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { StrategyName } from '../types.ts';

export const CHUNK_FIELDS = 5; // passageId, start, end, strategyMask, tokenCount

export interface IndexMeta {
  version: 1;
  builtAt: string;
  model: string;
  dims: number;
  passageCount: number;
  chunkCount: number;
  /** Number of chunks carrying each strategy's bit (chunks can carry several). */
  strategyCounts: Record<StrategyName, number>;
  /** How many distinct spans each strategy proposed before deduplication. */
  strategyProposed: Record<StrategyName, number>;
  source: string;
  buildSeconds: number;
  bytes: { passages: number; chunks: number; binary: number; int8: number; total: number };
}

export interface LoadedIndex {
  meta: IndexMeta;
  passageText: string[];
  passageAlt: string[];
  /** Struct-of-arrays chunk table; index with chunkId * CHUNK_FIELDS. */
  chunks: Int32Array;
  binary: Uint8Array;
  int8: Int8Array;
  binaryBytesPerVector: number;
}

const FILES = {
  meta: 'meta.json',
  passages: 'passages.json',
  chunks: 'chunks.bin',
  binary: 'vec_bin.bin',
  int8: 'vec_i8.bin',
} as const;

export async function saveIndex(
  dir: string,
  data: {
    meta: Omit<IndexMeta, 'bytes'>;
    passageText: string[];
    passageAlt: string[];
    chunks: Int32Array;
    binary: Uint8Array;
    int8: Int8Array;
  },
): Promise<IndexMeta> {
  await mkdir(dir, { recursive: true });

  const passagesJson = JSON.stringify({ text: data.passageText, alt: data.passageAlt });
  const bytes = {
    passages: Buffer.byteLength(passagesJson),
    chunks: data.chunks.byteLength,
    binary: data.binary.byteLength,
    int8: data.int8.byteLength,
    total: 0,
  };
  bytes.total = bytes.passages + bytes.chunks + bytes.binary + bytes.int8;

  const meta: IndexMeta = { ...data.meta, bytes };

  await Promise.all([
    writeFile(resolve(dir, FILES.meta), `${JSON.stringify(meta, null, 2)}\n`),
    writeFile(resolve(dir, FILES.passages), passagesJson),
    writeFile(resolve(dir, FILES.chunks), Buffer.from(data.chunks.buffer, 0, data.chunks.byteLength)),
    writeFile(resolve(dir, FILES.binary), Buffer.from(data.binary.buffer, 0, data.binary.byteLength)),
    writeFile(resolve(dir, FILES.int8), Buffer.from(data.int8.buffer, 0, data.int8.byteLength)),
  ]);

  return meta;
}

export async function loadIndex(dir: string): Promise<LoadedIndex> {
  const [metaRaw, passagesRaw, chunksBuf, binaryBuf, int8Buf] = await Promise.all([
    readFile(resolve(dir, FILES.meta), 'utf8'),
    readFile(resolve(dir, FILES.passages), 'utf8'),
    readFile(resolve(dir, FILES.chunks)),
    readFile(resolve(dir, FILES.binary)),
    readFile(resolve(dir, FILES.int8)),
  ]);

  const meta = JSON.parse(metaRaw) as IndexMeta;
  const passages = JSON.parse(passagesRaw) as { text: string[]; alt: string[] };

  const chunks = new Int32Array(chunksBuf.buffer, chunksBuf.byteOffset, chunksBuf.byteLength / 4);
  const binary = new Uint8Array(binaryBuf.buffer, binaryBuf.byteOffset, binaryBuf.byteLength);
  const int8 = new Int8Array(int8Buf.buffer, int8Buf.byteOffset, int8Buf.byteLength);

  if (chunks.length !== meta.chunkCount * CHUNK_FIELDS) {
    throw new Error(`chunks.bin has ${chunks.length / CHUNK_FIELDS} rows, meta says ${meta.chunkCount}`);
  }
  if (int8.length !== meta.chunkCount * meta.dims) {
    throw new Error(`vec_i8.bin has ${int8.length / meta.dims} vectors, meta says ${meta.chunkCount}`);
  }

  return {
    meta,
    passageText: passages.text,
    passageAlt: passages.alt,
    chunks,
    binary,
    int8,
    binaryBytesPerVector: meta.dims >> 3,
  };
}

/** Field accessors — the chunk table is a flat Int32Array, not objects. */
export const chunkPassageId = (chunks: Int32Array, id: number): number => chunks[id * CHUNK_FIELDS];
export const chunkStart = (chunks: Int32Array, id: number): number => chunks[id * CHUNK_FIELDS + 1];
export const chunkEnd = (chunks: Int32Array, id: number): number => chunks[id * CHUNK_FIELDS + 2];
export const chunkMask = (chunks: Int32Array, id: number): number => chunks[id * CHUNK_FIELDS + 3];
export const chunkTokens = (chunks: Int32Array, id: number): number => chunks[id * CHUNK_FIELDS + 4];
