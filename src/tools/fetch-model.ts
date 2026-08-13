/**
 * Vendors the encoder into ./models so the server never touches the network at
 * boot and a container image is reproducible.
 *
 * Usage: node src/tools/fetch-model.ts
 */
import { mkdir, writeFile, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { EMBED_MODEL } from '../core/embed.ts';
import { RERANK_MODEL } from '../core/retrieve/rerank.ts';

const MODELS = [EMBED_MODEL, RERANK_MODEL];

const FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'onnx/model_quantized.onnx',
];

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  let total = 0;

  for (const model of MODELS) {
    const root = resolve(process.cwd(), 'models', model);
    console.log(`\n▸ ${model}`);

    for (const file of FILES) {
      const target = resolve(root, file);
      if (await exists(target)) {
        console.log(`  = ${file} (already present)`);
        continue;
      }
      const url = `https://huggingface.co/${model}/resolve/main/${file}`;
      const response = await fetch(url);
      if (!response.ok) {
        // Some of these are optional depending on the repo.
        if (response.status === 404 && file !== 'tokenizer.json' && !file.endsWith('.onnx')) {
          console.log(`  - ${file} (not published, skipping)`);
          continue;
        }
        throw new Error(`${url} -> ${response.status}`);
      }
      const buffer = Buffer.from(await response.arrayBuffer());
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, buffer);
      total += buffer.byteLength;
      console.log(`  + ${file.padEnd(28)} ${(buffer.byteLength / 1e6).toFixed(1)} MB`);
    }
  }

  console.log(`\n  ${(total / 1e6).toFixed(1)} MB fetched → models/\n`);
}

await main();
