"""Turn the annotation sheets into readable HTML pages, one per annotator.

The CSVs keep the raw LaTeX, which is hard to read. This page renders the math (MathJax),
shows code blocks as code, and highlights every \\boxed{...} answer in yellow, so the boxed
part stays easy to spot. Labels can be entered on the page: they are saved in the browser as
you go, and "Download labels" gives a CSV (item_id, label, annotator, notes) to hand back.

Usage (math/ directory; no GPU, only the Python standard library):
    python make_annotation_html.py $SCRATCH/rlvr/annotation_b1
Reads sheet_<annotator>.csv and items_full.jsonl from that directory and writes
annotate_<annotator>.html next to them. Open the HTML in a browser (needs internet for MathJax).
"""
import csv
import glob
import json
import os
import sys

TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation __NAME__</title>
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true},
  options: {skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'input']}
};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
<style>
  :root { --bg:#fafafa; --card:#fff; --ink:#1d1d1f; --muted:#666; --line:#ddd; --code:#f3f3f3;
          --box:#fff2a8; --boxline:#d4a300; --accent:#2563eb; --done:#e8f5e9; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151515; --card:#1f1f1f; --ink:#eee; --muted:#aaa; --line:#333; --code:#2a2a2a;
            --box:#5a4a00; --boxline:#e0b800; --accent:#7aa2ff; --done:#1e3323; }
  }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  header input { padding:4px 8px; }
  button { padding:6px 12px; cursor:pointer; }
  main { max-width:960px; margin:0 auto; padding:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px;
          margin-bottom:18px; }
  .card.done { background:var(--done); }
  .meta { color:var(--muted); font-size:13px; }
  .row { margin:8px 0; }
  .lab { font-weight:600; }
  .text { white-space:pre-wrap; overflow-wrap:anywhere; }
  pre { background:var(--code); padding:8px 10px; border-radius:6px; overflow-x:auto; font-size:13px;
        white-space:pre-wrap; }
  mjx-container { overflow-x:auto; max-width:100%; }
  mjx-menclose { background:var(--box); outline:2px solid var(--boxline); }
  .rawbox { background:var(--box); outline:2px solid var(--boxline); }
  .answers { display:flex; flex-wrap:wrap; gap:16px; }
  .answers div { min-width:200px; }
  .labels label { margin-right:14px; font-weight:600; }
  .prelabel { color:var(--muted); }
  textarea { width:100%; min-height:38px; }
  details summary { cursor:pointer; color:var(--accent); }
  .all { color:#c2410c; font-weight:600; }
</style>
</head>
<body>
<header>
  <strong>__NAME__</strong>
  <span>annotator: <input id="who" placeholder="your name"></span>
  <span id="progress"></span>
  <button id="dl">Download labels</button>
  <span class="meta">标签：A 写对了没框 / B 没给最终答案 / C 真做错 / D 框了且等价但判分器没认出（规则见 Google Doc 的“标注规则”）</span>
</header>
<main id="main"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const items = JSON.parse(document.getElementById('data').textContent);
const KEY = 'rlvr-annotation-__NAME__';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} progress(); }

function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
// Highlight \boxed{...} that sits outside math delimiters (MathJax only boxes it inside math).
function markRawBoxed(html) {
  const MATH = /\$\$[\s\S]*?\$\$|\$[^$]*?\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g;
  const BOX = /\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g;
  let out = '', last = 0, m;
  while ((m = MATH.exec(html)) !== null) {   // leave math alone: MathJax boxes it itself
    out += html.slice(last, m.index).replace(BOX, s => '<span class="rawbox">' + s + '</span>') + m[0];
    last = m.index + m[0].length;
  }
  return out + html.slice(last).replace(BOX, s => '<span class="rawbox">' + s + '</span>');
}
function render(text) {
  const parts = String(text ?? '').split('```');
  return parts.map((p, i) => {
    if (i % 2 === 1) {
      const body = p.replace(/^[a-zA-Z]*\n/, '');
      return '<pre><code>' + esc(body) + '</code></pre>';
    }
    return '<div class="text">' + esc(p) + '</div>';
  }).join('');
}
function card(it) {
  const s = saved[it.item_id] || {};
  const radios = ['A','B','C','D'].map(l =>
    `<label><input type="radio" name="l_${it.item_id}" value="${l}" ${s.label===l?'checked':''}> ${l}</label>`).join('');
  return `<div class="card ${s.label?'done':''}" id="c_${it.item_id}">
    <div class="meta">${esc(it.item_id)} · MATH500 idx ${esc(it.idx)} · level ${esc(it.level)}
      ${it.assignee==='ALL' ? ' · <span class="all">三人共同标注</span>' : ''}</div>
    <div class="row"><span class="lab">题目</span><div class="text">${esc(it.question)}</div></div>
    <div class="answers row">
      <div><span class="lab">标准答案：</span> \\(${esc(it.ground_truth)}\\)</div>
      <div><span class="lab">程序抽出的答案：</span> <code>${esc(it.extracted)}</code></div>
    </div>
    <div class="row"><span class="lab">第一段代码之前（最后一部分）</span>${render(it.before_first_code_block)}</div>
    <div class="row"><span class="lab">回答结尾</span>${render(it.response_tail)}</div>
    <details class="row"><summary>展开完整回答</summary>${render(it.full)}</details>
    <div class="row prelabel">脚本建议：${esc(it.prelabel)}（${esc(it.prelabel_reason)}）——仅供参考</div>
    <div class="row labels">${radios}</div>
    <textarea placeholder="notes（拿不准的原因；B 类方法正确写 method correct）">${esc(s.notes||'')}</textarea>
  </div>`;
}
const main = document.getElementById('main');
main.innerHTML = items.map(card).join('');
main.querySelectorAll('.text').forEach(el => { el.innerHTML = markRawBoxed(el.innerHTML); });
items.forEach(it => {
  const c = document.getElementById('c_' + it.item_id);
  c.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', () => {
    saved[it.item_id] = Object.assign(saved[it.item_id] || {}, {label: r.value});
    c.classList.add('done'); save();
  }));
  c.querySelector('textarea').addEventListener('input', e => {
    saved[it.item_id] = Object.assign(saved[it.item_id] || {}, {notes: e.target.value}); save();
  });
});
const who = document.getElementById('who');
who.value = saved.__who || '';
who.addEventListener('input', () => { saved.__who = who.value; save(); });
function progress() {
  const n = items.filter(it => (saved[it.item_id] || {}).label).length;
  document.getElementById('progress').textContent = `已标 ${n} / ${items.length}`;
}
progress();
document.getElementById('dl').addEventListener('click', () => {
  const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = [['item_id','label','annotator','notes'].join(',')].concat(items.map(it => {
    const s = saved[it.item_id] || {};
    return [it.item_id, s.label || '', saved.__who || '', s.notes || ''].map(q).join(',');
  }));
  const blob = new Blob(['﻿' + rows.join('\n')], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'labels___NAME__.csv'; a.click();
});
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_annotation_html.py <annotation out_dir>")
    d = sys.argv[1]
    full = {}
    with open(os.path.join(d, "items_full.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                full[x["item_id"]] = x["response"]
    sheets = [p for p in glob.glob(os.path.join(d, "sheet_*.csv"))
              if not p.endswith("sheet_all.csv")]
    if not sheets:
        sys.exit(f"No sheet_<annotator>.csv in {d}")
    for p in sorted(sheets):
        name = os.path.basename(p)[len("sheet_"):-len(".csv")]
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["full"] = full.get(r["item_id"], "")
        data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        html = TEMPLATE.replace("__NAME__", name).replace("__DATA__", data)
        out = os.path.join(d, f"annotate_{name}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote {out} ({len(rows)} items)")


if __name__ == "__main__":
    main()
