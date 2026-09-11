/**
 * Panel stylesheet. Everything lives in the shadow root: `:host { all: initial }` walls the
 * panel off from the host page's cascade, and the constructed `CSSStyleSheet` (with a
 * `<style>` fallback) means no inline style attribute is ever written. The one
 * dynamic style, the top-k `.bar` width, is set through CSSOM (`el.style.width`), which
 * CSP `style-src` does not block.
 */

export const PANEL_CSS = `
:host {
  all: initial;
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 2147483647;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #1f2328;
  color-scheme: light;
}
*, *::before, *::after { box-sizing: border-box; }
.panel {
  display: flex;
  flex-direction: column;
  width: 560px;
  max-width: calc(100vw - 32px);
  max-height: calc(100vh - 32px);
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(31, 35, 40, 0.18);
  overflow: hidden;
}
.panel.closed { width: auto; }
.panel.closed .body { display: none; }

.header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: #f6f8fa;
  border-bottom: 1px solid #d0d7de;
  cursor: pointer;
  user-select: none;
}
.panel.closed .header { border-bottom: none; }
.title { font-weight: 600; white-space: nowrap; }
.badge {
  min-width: 20px;
  padding: 0 6px;
  border-radius: 10px;
  background: #0969da;
  color: #ffffff;
  text-align: center;
  font-weight: 600;
}
.spacer { flex: 1; }
.btn {
  font: inherit;
  padding: 2px 8px;
  border: 1px solid #d0d7de;
  border-radius: 4px;
  background: #ffffff;
  color: inherit;
  cursor: pointer;
}
.btn:hover { background: #f3f4f6; }
.btn:disabled { opacity: 0.55; cursor: progress; }
.chev { width: 1em; text-align: center; color: #57606a; }

.body { overflow: auto; }
.rows { list-style: none; margin: 0; padding: 0; }
.empty { padding: 16px; color: #57606a; text-align: center; }
.row { border-bottom: 1px solid #eaeef2; }
.row:last-child { border-bottom: none; }

.summary {
  display: grid;
  grid-template-columns: 3em minmax(8em, max-content) 1fr auto 1em;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  cursor: pointer;
}
.summary:hover { background: #f6f8fa; }
.row.expanded .summary { background: #f6f8fa; }
.n { color: #57606a; }
.label { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.excerpt { color: #57606a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ms { color: #57606a; white-space: nowrap; text-align: right; }
.dot { width: 8px; height: 8px; border-radius: 50%; justify-self: center; }
.dot.pending { background: #bf8700; }
.dot.ok { background: #1a7f37; }
.dot.err { background: #cf222e; }

.details { padding: 4px 10px 10px 10px; background: #fbfcfd; }
.section { margin-top: 8px; }
.section > h3 {
  margin: 0 0 4px 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #57606a;
}
.section h4 { margin: 6px 0 2px 0; font-size: 11px; font-weight: 600; color: #57606a; }
.meta { color: #57606a; margin-bottom: 4px; }
.error { color: #cf222e; white-space: pre-wrap; }
.muted { color: #57606a; }
pre {
  margin: 0;
  padding: 6px 8px;
  max-height: 240px;
  overflow: auto;
  background: #ffffff;
  border: 1px solid #d0d7de;
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
  border: 1px solid #d0d7de;
  border-radius: 4px;
  background: #ffffff;
}
.chip-id { font-size: 10px; color: #57606a; }
.chip-str { white-space: pre; }
.ws { background: #ddf4ff; border-radius: 2px; }

.run { margin-bottom: 8px; }
table.tensors, table.topk { width: 100%; border-collapse: collapse; background: #ffffff; }
table.tensors th, table.tensors td, table.topk th, table.topk td {
  padding: 2px 6px;
  border: 1px solid #eaeef2;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
table.tensors th, table.topk th { background: #f6f8fa; font-weight: 600; color: #57606a; }
table.tensors td.head { white-space: normal; word-break: break-all; color: #57606a; min-width: 14em; }
table.tensors td.num, table.topk td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* max-width: 0 keeps the wide colspan values cell from stretching the table and squeezing
   the other columns; the list scrolls inside the cell instead. */
tr.values td { white-space: normal; background: #fbfcfd; max-width: 0; }
.values-list {
  max-height: 200px;
  overflow: auto;
  overflow-wrap: anywhere;
  word-break: break-all;
  font-variant-numeric: tabular-nums;
  color: #1f2328;
}

/* Generation section: per-step blocks and top-k rows with probability bars. */
.step { margin-bottom: 8px; }
.step-head { font-weight: 600; color: #1f2328; }
table.topk td.tok { white-space: pre; font-weight: 600; }
table.topk tr.picked td { background: #dafbe1; }
table.topk td.prob { width: 45%; }
.prob { display: flex; align-items: center; gap: 6px; }
.prob-text { min-width: 5.5em; text-align: right; font-variant-numeric: tabular-nums; }
.bar { height: 8px; width: 0; max-width: 100%; background: #0969da; border-radius: 2px; }
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
