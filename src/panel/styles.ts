/**
 * Panel stylesheet. Everything lives in the shadow root: `:host { all: initial }` walls the
 * panel off from the host page's cascade, and the constructed `CSSStyleSheet` (with a
 * `<style>` fallback) means no inline style attribute is ever written. The one
 * dynamic style, the top-k `.bar` width, is set through CSSOM (`el.style.width`), which
 * CSP `style-src` does not block.
 *
 * Every colour is a `--tjsi-*` custom property declared on `:host` (light values); the two
 * blocks at the end of the sheet swap in `DARK_VARS`.
 */

/**
 * Dark token set. Applied twice at the end of `PANEL_CSS`: under `prefers-color-scheme: dark`
 * unless the host carries `data-theme="light"`, and unconditionally for `data-theme="dark"`,
 * so the explicit option wins over the media query in both directions.
 */
const DARK_VARS =
  '--tjsi-fg: #e6edf3; --tjsi-muted: #8b949e; --tjsi-bg: #0d1117; --tjsi-bg-alt: #161b22; ' +
  '--tjsi-bg-detail: #10151c; --tjsi-border: #30363d; --tjsi-border-soft: #21262d; ' +
  '--tjsi-accent: #388bfd; --tjsi-on-accent: #ffffff; --tjsi-ok: #3fb950; --tjsi-warn: #d29922; ' +
  '--tjsi-err: #f85149; --tjsi-picked: #12361f; --tjsi-ws: #1f3a5a; --tjsi-shadow: rgba(0,0,0,.6); ' +
  'color-scheme: dark;';

export const PANEL_CSS = `
:host {
  all: initial;
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 2147483647;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --tjsi-fg: #1f2328;
  --tjsi-muted: #57606a;
  --tjsi-bg: #ffffff;
  --tjsi-bg-alt: #f6f8fa;
  --tjsi-bg-detail: #fbfcfd;
  --tjsi-border: #d0d7de;
  --tjsi-border-soft: #eaeef2;
  --tjsi-accent: #0969da;
  --tjsi-on-accent: #ffffff;
  --tjsi-ok: #1a7f37;
  --tjsi-warn: #bf8700;
  --tjsi-err: #cf222e;
  --tjsi-picked: #dafbe1;
  --tjsi-ws: #ddf4ff;
  --tjsi-shadow: rgba(31, 35, 40, 0.18);
  color: var(--tjsi-fg);
  color-scheme: light;
}
*, *::before, *::after { box-sizing: border-box; }
.panel {
  display: flex;
  flex-direction: column;
  width: 560px;
  max-width: calc(100vw - 32px);
  max-height: calc(100vh - 32px);
  background: var(--tjsi-bg);
  border: 1px solid var(--tjsi-border);
  border-radius: 8px;
  box-shadow: 0 8px 24px var(--tjsi-shadow);
  overflow: hidden;
}
.panel.closed { width: auto; }
.panel.closed .body { display: none; }

.header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: var(--tjsi-bg-alt);
  border-bottom: 1px solid var(--tjsi-border);
  cursor: pointer;
  user-select: none;
}
.panel.closed .header { border-bottom: none; }
.title { font-weight: 600; white-space: nowrap; }
.badge {
  min-width: 20px;
  padding: 0 6px;
  border-radius: 10px;
  background: var(--tjsi-accent);
  color: var(--tjsi-on-accent);
  text-align: center;
  font-weight: 600;
}
.spacer { flex: 1; }
.btn {
  font: inherit;
  padding: 2px 8px;
  border: 1px solid var(--tjsi-border);
  border-radius: 4px;
  background: var(--tjsi-bg);
  color: inherit;
  cursor: pointer;
}
.btn:hover { background: var(--tjsi-bg-alt); }
.btn:disabled { opacity: 0.55; cursor: progress; }
.chev { width: 1em; text-align: center; color: var(--tjsi-muted); }

.body { overflow: auto; }
.rows { list-style: none; margin: 0; padding: 0; }
.empty { padding: 16px; color: var(--tjsi-muted); text-align: center; }
.row { border-bottom: 1px solid var(--tjsi-border-soft); }
.row:last-child { border-bottom: none; }

.summary {
  display: grid;
  grid-template-columns: 3em minmax(8em, max-content) 1fr auto 1em;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  cursor: pointer;
}
.summary:hover { background: var(--tjsi-bg-alt); }
.row.expanded .summary { background: var(--tjsi-bg-alt); }
.n { color: var(--tjsi-muted); }
.label { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.excerpt { color: var(--tjsi-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ms { color: var(--tjsi-muted); white-space: nowrap; text-align: right; }
.dot { width: 8px; height: 8px; border-radius: 50%; justify-self: center; }
.dot.pending { background: var(--tjsi-warn); }
.dot.ok { background: var(--tjsi-ok); }
.dot.err { background: var(--tjsi-err); }

.details { padding: 4px 10px 10px 10px; background: var(--tjsi-bg-detail); }
.section { margin-top: 8px; }
.section > h3 {
  margin: 0 0 4px 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--tjsi-muted);
}
.section h4 { margin: 6px 0 2px 0; font-size: 11px; font-weight: 600; color: var(--tjsi-muted); }
.meta { color: var(--tjsi-muted); margin-bottom: 4px; }
.error { color: var(--tjsi-err); white-space: pre-wrap; }
.muted { color: var(--tjsi-muted); }
pre {
  margin: 0;
  padding: 6px 8px;
  max-height: 240px;
  overflow: auto;
  background: var(--tjsi-bg);
  border: 1px solid var(--tjsi-border);
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-word;
  font: inherit;
}

.tokens { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 4px; }
.chip {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  padding: 1px 5px;
  border: 1px solid var(--tjsi-border);
  border-radius: 4px;
  background: var(--tjsi-bg);
}
.chip-id { font-size: 10px; color: var(--tjsi-muted); }
.chip-str { white-space: pre; }

.run { margin-bottom: 8px; }
table.tensors, table.topk { width: 100%; border-collapse: collapse; background: var(--tjsi-bg); }
table.tensors th, table.tensors td, table.topk th, table.topk td {
  padding: 2px 6px;
  border: 1px solid var(--tjsi-border-soft);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
table.tensors th, table.topk th { background: var(--tjsi-bg-alt); font-weight: 600; color: var(--tjsi-muted); }
table.tensors td.head { white-space: normal; word-break: break-all; color: var(--tjsi-muted); min-width: 14em; }
table.tensors td.num, table.topk td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* max-width: 0 keeps the wide colspan values cell from stretching the table and squeezing
   the other columns; the list scrolls inside the cell instead. */
tr.values td { white-space: normal; background: var(--tjsi-bg-detail); max-width: 0; }
.values-list {
  max-height: 200px;
  overflow: auto;
  overflow-wrap: anywhere;
  word-break: break-all;
  font-variant-numeric: tabular-nums;
  color: var(--tjsi-fg);
}

/* Generation section: per-step blocks and top-k rows with probability bars. */
.step { margin-bottom: 8px; }
.step-head { font-weight: 600; color: var(--tjsi-fg); }
table.topk td.tok { white-space: pre; font-weight: 600; }
table.topk tr.picked td { background: var(--tjsi-picked); }
table.topk td.prob { width: 45%; }
.prob { display: flex; align-items: center; gap: 6px; }
.prob-text { min-width: 5.5em; text-align: right; font-variant-numeric: tabular-nums; }
.bar { height: 8px; width: 0; max-width: 100%; background: var(--tjsi-accent); border-radius: 2px; }

@media (prefers-color-scheme: dark) { :host(:not([data-theme="light"])) { ${DARK_VARS} } }
:host([data-theme="dark"]) { ${DARK_VARS} }
`;

type ConstructableShadow = ShadowRoot & { adoptedStyleSheets?: CSSStyleSheet[] };

function supportsConstructedSheets(shadow: ConstructableShadow): boolean {
  return (
    typeof CSSStyleSheet === 'function' &&
    typeof (CSSStyleSheet.prototype as { replaceSync?: unknown }).replaceSync === 'function' &&
    'adoptedStyleSheets' in shadow
  );
}

/**
 * Attaches `PANEL_CSS` to `shadow`: a constructed `CSSStyleSheet` via `replaceSync` where
 * supported, otherwise a `<style>` element prepended to the shadow root.
 */
export function adoptStyles(shadow: ShadowRoot, css: string = PANEL_CSS): void {
  const target = shadow as ConstructableShadow;
  if (supportsConstructedSheets(target)) {
    try {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(css);
      target.adoptedStyleSheets = [...(target.adoptedStyleSheets ?? []), sheet];
      return;
    } catch {
      // Fall through to the <style> element.
    }
  }
  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  shadow.prepend(styleEl);
}
