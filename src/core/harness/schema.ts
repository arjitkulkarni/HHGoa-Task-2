/**
 * A very small runtime validator.
 *
 * Everything crossing a trust boundary in this system is validated here: the
 * HTTP request bodies, Sarvam's transcription response, and the tool-call
 * arguments the LLM emits. Those are three different kinds of "someone else's
 * JSON" and all three have to fail loudly rather than propagate an undefined
 * into the retrieval stage.
 *
 * Hand-rolled rather than a dependency because the surface needed is this small
 * and the error messages should name the exact path that failed — a retrying
 * harness is only useful if the retry knows what was wrong.
 */

export class ValidationError extends Error {
  readonly path: string;
  constructor(path: string, message: string) {
    super(path ? `${path}: ${message}` : message);
    this.name = 'ValidationError';
    this.path = path;
  }
}

export interface Validator<T> {
  parse: (value: unknown, path?: string) => T;
  optional: () => Validator<T | undefined>;
  withDefault: (fallback: T) => Validator<T>;
}

/**
 * `Validator<T>` is invariant in T (because of `withDefault`), so a shape typed
 * as `Record<string, Validator<unknown>>` would reject `Validator<string>`.
 * Matching on the parse signature instead keeps the shape covariant, which is
 * what lets `s.object({ name: s.string() })` infer `{ name: string }`.
 */
type AnyValidator = { parse: (value: unknown, path?: string) => unknown };
type Infer<V> = V extends { parse: (value: unknown, path?: string) => infer T } ? T : never;

function make<T>(parse: (value: unknown, path: string) => T): Validator<T> {
  const validator: Validator<T> = {
    parse: (value, path = '') => parse(value, path),
    optional: () =>
      make<T | undefined>((value, path) => (value === undefined || value === null ? undefined : parse(value, path))),
    withDefault: (fallback) =>
      make<T>((value, path) => (value === undefined || value === null ? fallback : parse(value, path))),
  };
  return validator;
}

export const s = {
  string: (options: { min?: number; max?: number; trim?: boolean } = {}) =>
    make<string>((value, path) => {
      if (typeof value !== 'string') throw new ValidationError(path, `expected string, got ${typeName(value)}`);
      const out = options.trim === false ? value : value.trim();
      if (options.min !== undefined && out.length < options.min) {
        throw new ValidationError(path, `expected at least ${options.min} characters`);
      }
      if (options.max !== undefined && out.length > options.max) {
        throw new ValidationError(path, `expected at most ${options.max} characters`);
      }
      return out;
    }),

  number: (options: { min?: number; max?: number; int?: boolean } = {}) =>
    make<number>((value, path) => {
      const out = typeof value === 'string' ? Number(value) : value;
      if (typeof out !== 'number' || !Number.isFinite(out)) {
        throw new ValidationError(path, `expected a finite number, got ${typeName(value)}`);
      }
      if (options.int && !Number.isInteger(out)) throw new ValidationError(path, 'expected an integer');
      if (options.min !== undefined && out < options.min) throw new ValidationError(path, `expected >= ${options.min}`);
      if (options.max !== undefined && out > options.max) throw new ValidationError(path, `expected <= ${options.max}`);
      return out;
    }),

  boolean: () =>
    make<boolean>((value, path) => {
      if (typeof value === 'boolean') return value;
      if (value === 'true') return true;
      if (value === 'false') return false;
      throw new ValidationError(path, `expected boolean, got ${typeName(value)}`);
    }),

  enum: <const T extends readonly string[]>(values: T) =>
    make<T[number]>((value, path) => {
      if (typeof value !== 'string' || !values.includes(value)) {
        throw new ValidationError(path, `expected one of ${values.join(' | ')}, got ${JSON.stringify(value)}`);
      }
      return value as T[number];
    }),

  array: <T>(item: Validator<T>, options: { min?: number; max?: number } = {}) =>
    make<T[]>((value, path) => {
      if (!Array.isArray(value)) throw new ValidationError(path, `expected array, got ${typeName(value)}`);
      if (options.min !== undefined && value.length < options.min) {
        throw new ValidationError(path, `expected at least ${options.min} items`);
      }
      if (options.max !== undefined && value.length > options.max) {
        throw new ValidationError(path, `expected at most ${options.max} items`);
      }
      return value.map((entry, i) => item.parse(entry, `${path}[${i}]`));
    }),

  object: <T extends Record<string, AnyValidator>>(shape: T) =>
    make<{ [K in keyof T]: Infer<T[K]> }>((value, path) => {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new ValidationError(path, `expected object, got ${typeName(value)}`);
      }
      const source = value as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(shape)) {
        out[key] = shape[key].parse(source[key], path ? `${path}.${key}` : key);
      }
      return out as { [K in keyof T]: Infer<T[K]> };
    }),

  /** Accepts anything — used where a field is passed through untouched. */
  unknown: () => make<unknown>((value) => value),
};

function typeName(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

/** Parses JSON and validates in one step, with a single error type for both. */
export function parseJson<T>(raw: string, validator: Validator<T>, label = ''): T {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new ValidationError(label, `invalid JSON (${(error as Error).message})`);
  }
  return validator.parse(parsed, label);
}
