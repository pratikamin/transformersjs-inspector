/**
 * Browser glue for the Export button. Nothing here reads `document`, `navigator` or `URL`
 * at module load: each function takes the object it needs (defaulting to the globals) so
 * the unit tests can pass their own.
 */

/** The `URL` statics used for a Blob download. */
export interface ObjectUrlApi {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
}

/** Delay before the object URL is revoked; long enough for the browser to open the download. */
export const REVOKE_DELAY_MS = 1000;

function objectUrlApiOf(doc: Document): ObjectUrlApi | null {
  const win = doc.defaultView as (Window & { URL?: Partial<ObjectUrlApi> }) | null;
  const api = win?.URL ?? (globalThis as { URL?: Partial<ObjectUrlApi> }).URL;
  if (typeof api?.createObjectURL !== 'function' || typeof api.revokeObjectURL !== 'function') return null;
  return api as ObjectUrlApi;
}

/**
 * Offers `text` as a JSON file named `filename` through a Blob URL on a temporary
 * `<a download>`. Returns `false`, without throwing, when object URLs are unavailable or
 * the anchor path throws; the caller then falls back to the clipboard. The URL is revoked
 * after `REVOKE_DELAY_MS`.
 */
export function downloadJson(text: string, filename: string, doc: Document = document): boolean {
  const api = objectUrlApiOf(doc);
  if (!api) return false;
  let url: string | null = null;
  try {
    url = api.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = doc.createElement('a');
    a.href = url;
    a.download = filename;
    a.rel = 'noopener';
    a.click();
    const pending = url;
    setTimeout(() => api.revokeObjectURL(pending), REVOKE_DELAY_MS);
    return true;
  } catch {
    if (url !== null) {
      try {
        api.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
    }
    return false;
  }
}

/** Writes `text` to the clipboard; `false` when there is no clipboard or the write is rejected. */
export async function copyText(text: string, nav: Navigator | undefined = globalThis.navigator): Promise<boolean> {
  const clipboard = nav?.clipboard;
  if (!clipboard || typeof clipboard.writeText !== 'function') return false;
  try {
    await clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
