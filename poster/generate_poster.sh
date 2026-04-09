#!/usr/bin/env bash
# generate_poster.sh — Reproduces the MixtureVitae ICLR 2026 poster (index.html).
# Run from the poster/ directory:  bash generate_poster.sh
#
# Prerequisites:
#   - Logo PNGs must already exist in logos/
#   - Figure PNGs (mixture_vitae_category_pie.png, etc.) must already exist.
#
# What this script does:
#   1. Writes index.html with the complete poster (affiliations, logos, content)
#   2. Verifies the output checksum

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Writing index.html..."
cat > index.html << 'POSTEREOF'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MixtureVitae - ICLR 2026 Poster</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script crossorigin src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  :root {
    --teal: #009688; --teal-light: #e0f2f1;
    --blue: #1565c0; --blue-light: #e3f2fd;
    --orange: #e65100; --orange-light: #fff3e0;
    --purple: #6a1b9a; --purple-light: #f3e5f5;
    --red: #c62828; --red-light: #ffebee;
    --green: #2e7d32; --green-light: #e8f5e9;
    --text: #212121; --text-light: #555;
    --bg: #eceff1; --card-bg: #fff;
    --gap: 3mm; --pad: 8mm;
    --font-scale: 1.25;
  }

  @page { size: 841mm 1189mm; margin: 0; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html { font-family: 'Nunito', sans-serif; color: var(--text); -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { width: 841mm; height: 1189mm; background: var(--bg); overflow: hidden; transform-origin: top left; }
  #root { width: 100%; height: 100%; display: grid; grid-template-rows: auto 1fr; }
  @media screen { html { background:#78909c; overflow:hidden; width:100vw; height:100vh; } body { position:absolute; box-shadow:0 4px 30px rgba(0,0,0,.3); border-radius:4px; } }
  @media print { html { background:none; } body { position:static; transform:none!important; box-shadow:none; border-radius:0; } .toolbar,.divider,.swap-handle,.drop-zone{display:none!important;} }
  body.preview .divider, body.preview .swap-handle, body.preview .drop-zone { display:none!important; }

  /* === HEADER === */
  .header { background: linear-gradient(135deg, #004d40, #00796b 50%, #009688); padding:8mm var(--pad) 7mm; display:flex; align-items:center; gap:8mm; }
  .header-left { flex:1; }
  .header h1 { font-size:52pt; font-weight:900; line-height:1.1; color:#fff; margin-bottom:4mm; }
  .header h1 .m { color:#80cbc4; }
  .header .authors { font-size:18pt; color:rgba(255,255,255,.92); font-weight:700; line-height:1.4; }
  .header .authors sup { font-size:10pt; color:#80cbc4; }
  .header .aff { font-size:13pt; color:rgba(255,255,255,.72); margin-top:2mm; line-height:1.4; }
  .header .aff sup { color:#80cbc4; font-size:9pt; }
  .header-logos { display:flex; align-items:center; justify-content:center; gap:5mm; flex-shrink:0; flex-wrap:wrap; max-width:180mm; }
  .header-logos img { height:26mm; object-fit:contain; background:#fff; border-radius:2mm; padding:1mm; }
  .header-right { display:flex; flex-direction:column; align-items:center; gap:4mm; }
  .badge { font-size:18pt; font-weight:900; color:#fff; background:rgba(255,255,255,.18); border:2px solid rgba(255,255,255,.5); padding:2mm 6mm; border-radius:2mm; white-space:nowrap; }
  .header-right .qr-row { display:flex; gap:5mm; }
  .header-right .qr img { width:22mm; height:22mm; border-radius:1.5mm; background:#fff; padding:.8mm; }
  .header-right .ql { font-size:7pt; color:#80cbc4; font-weight:800; text-align:center; max-width:24mm; }
  .header-right .qr { text-align:center; }

  /* === POSTER COLUMNS === */
  .poster { display:flex; gap:var(--gap); padding:var(--gap) var(--pad); min-height:0; flex:1; overflow:hidden; }
  .col { display:flex; flex-direction:column; gap:var(--gap); min-height:0; overflow:hidden; }

  /* === CARD === */
  .card { background:var(--card-bg); border-radius:2.5mm; padding:3mm 4mm; box-shadow:0 1px 3px rgba(0,0,0,.08); border-top:3px solid var(--teal); display:flex; flex-direction:column; overflow:hidden; min-height:0; flex:0 0 auto; position:relative; transition: outline .15s; }
  .card.grow { flex:1 1 0; }
  .card.blue { border-top-color:var(--blue); }
  .card.orange { border-top-color:var(--orange); }
  .card.teal   { border-top-color:var(--teal); }
  .card.purple { border-top-color:var(--purple); }
  .card.red    { border-top-color:var(--red); }
  .card.green  { border-top-color:var(--green); }

  .card h2 { font-size:calc(12pt * var(--font-scale)); font-weight:800; margin-bottom:2mm; flex-shrink:0; color:var(--teal); }
  .card.blue h2{color:var(--blue);} .card.orange h2{color:var(--orange);} .card.teal h2{color:var(--teal);} .card.purple h2{color:var(--purple);} .card.red h2{color:var(--red);} .card.green h2{color:var(--green);}
  .card p,.card li { font-size:calc(9.5pt * var(--font-scale)); line-height:1.35; }
  .card ul,.card ol { padding-left:5mm; flex-shrink:0; }
  .card li { margin-bottom:1mm; }
  .card li::marker { font-weight:700; }
  .card b { font-weight:800; }

  /* Highlight box */
  .hl { background:var(--teal-light); border-left:3px solid var(--teal); padding:3mm 4mm; border-radius:0 1.5mm 1.5mm 0; margin-bottom:2mm; flex-shrink:0; }
  .hl p { font-size:calc(10pt * var(--font-scale)); line-height:1.35; font-weight:600; }
  .hl-blue { background:var(--blue-light); border-left-color:var(--blue); }
  .hl-orange { background:var(--orange-light); border-left-color:var(--orange); }
  .hl-purple { background:var(--purple-light); border-left-color:var(--purple); }

  /* Figure */
  .fig { flex:1; display:flex; flex-direction:column; min-height:0; margin:1mm 0; }
  .fig-wrap { flex:1; display:flex; align-items:center; justify-content:center; min-height:0; overflow:hidden; }
  .fig-wrap img { width:100%; height:100%; object-fit:contain; border-radius:1.5mm; display:block; }
  .fig .cap { font-size:calc(7.5pt * var(--font-scale)); color:var(--text-light); margin-top:1mm; line-height:1.25; flex-shrink:0; }
  .fig .cap b { color:var(--text); }

  /* Tables */
  table { width:100%; border-collapse:collapse; font-size:calc(8.5pt * var(--font-scale)); margin:1.5mm 0; }
  thead th { background:var(--teal); color:#fff; padding:1.5mm 2.5mm; text-align:left; font-weight:700; font-size:calc(8pt * var(--font-scale)); }
  tbody td { padding:1.2mm 2.5mm; border-bottom:1px solid #eee; }
  tbody tr:nth-child(even) { background:#f5f5f5; }
  .best { font-weight:800; color:var(--orange); }
  .header-cell { font-weight:700; background:#e8f5e9; }

  /* Math */
  .katex { font-size:calc(10pt * var(--font-scale))!important; }
  .eq { background:var(--orange-light); border:1px solid var(--orange); border-radius:1.5mm; padding:1.5mm 2.5mm; margin:1.5mm 0; text-align:center; flex-shrink:0; }

  /* Links */
  .links { display:flex; align-items:center; gap:3mm; background:var(--teal-light); padding:2.5mm 3.5mm; border-radius:1.5mm; margin-top:2mm; flex-shrink:0; }
  .links .ll { font-size:calc(8.5pt * var(--font-scale)); line-height:1.5; }
  .links .ll b { color:var(--teal); }

  /* Tier boxes */
  .tier-box { padding:2mm 3mm; border-radius:1.5mm; margin-bottom:1.5mm; flex-shrink:0; }
  .tier1 { background:#e8f5e9; border-left:3px solid #2e7d32; }
  .tier2 { background:#fff3e0; border-left:3px solid #e65100; }
  .tier3 { background:#e3f2fd; border-left:3px solid #1565c0; }
  .tier-box .tier-label { font-size:calc(8.5pt * var(--font-scale)); font-weight:800; margin-bottom:0.5mm; }
  .tier-box p { font-size:calc(8.5pt * var(--font-scale)); line-height:1.3; }

  /* === TOOLBAR === */
  .toolbar { position:fixed; top:8px; right:8px; z-index:1000; display:flex; gap:6px; align-items:center; }
  .toolbar button { font-family:'Nunito'; font-size:11px; font-weight:700; padding:5px 12px; border-radius:6px; border:none; cursor:pointer; background:#eee; color:#555; }
  .toolbar button:hover { background:#ddd; }

  .card .swap-handle { position:absolute; top:1.5mm; right:1.5mm; width:8mm; height:8mm; background:var(--teal); color:#fff; border-radius:1.5mm; font-size:9pt; text-align:center; line-height:8mm; cursor:pointer; z-index:10; user-select:none; opacity:0; pointer-events:none; transition: opacity .15s, background .15s, transform .15s; }
  .card .swap-handle:hover { opacity:.8; transform:scale(1.15); }
  .card.swap-selected { outline:3px solid var(--orange); outline-offset:-1px; }
  .card.swap-selected .swap-handle { background:var(--orange); opacity:1; transform:scale(1.2); }

  .divider { position:relative; z-index:50; flex-shrink:0; }
  .divider::after { content:''; position:absolute; background:rgba(0,150,136,.12); border-radius:1.5mm; transition:background .15s; }
  .divider:hover::after, .divider.active::after { background:rgba(0,150,136,.5); }
  .col-resize { width:6mm; cursor:col-resize; margin:0 -3mm; }
  .col-resize::after { top:0; bottom:0; left:1.5mm; width:3mm; }
  .row-resize { height:6mm; cursor:row-resize; margin:-3mm 0; }
  .row-resize::after { left:0; right:0; top:1.5mm; height:3mm; }

  .drop-zone { height:0; position:relative; z-index:60; transition: height .15s; }
  .drop-zone.visible { height:5mm; }
  .drop-zone-inner { position:absolute; left:2mm; right:2mm; top:0; bottom:0; border-radius:1.5mm; background:rgba(232,148,58,.15); border:2px dashed var(--orange); transition: background .15s; cursor:pointer; }
  .drop-zone-inner:hover { background:rgba(232,148,58,.35); }
</style>
</head>
<body>
<div id="root"></div>

<script type="text/babel">
const { useState, useRef, useEffect, useCallback } = React;
const MM = 3.7795275591;
const POSTER_W_MM = 841;
const POSTER_H_MM = 1189;

const CARD_REGISTRY = {
  tldr: {
    title: 'TL;DR',
    color: null,
    grow: false,
    body: (
      <>
        <div className="hl"><p>We present <b>MixtureVitae</b>, a 422B-token open-access pretraining corpus built from permissive-first sources that closes the performance gap to leading non-permissive datasets, proving that legally robust data can train capable LLMs.</p></div>
        <ul>
          <li><b>Permissive-first, risk-mitigated recipe</b> &mdash; Three-tier licensing scheme (CC-BY, Apache, public domain + government works) with shard-level provenance metadata</li>
          <li><b>Single-stage pretraining with instruction &amp; reasoning data</b> &mdash; Front-loads synthetic instruction, math, and code data typically reserved for post-training</li>
          <li><b>1.7B base model matches instruction-tuned baselines</b> &mdash; Outperforms SmolLM2-1.7B-Instruct on GSM8K, HumanEval, and MBPP using 36&times; fewer tokens (300B vs ~11T)</li>
        </ul>
      </>
    ),
  },
  composition: {
    title: 'Dataset Composition',
    color: 'blue',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/mixture_vitae_category_pie.png" alt="Dataset composition" /></div>
        <div className="cap"><b>Figure 1(a).</b> MixtureVitae composition by top-level category and content domain. The 422B-token corpus spans web data, curated sources, and instruction/reasoning datasets.</div></div>
      </>
    ),
  },
  licensing: {
    title: 'Three-Tier Licensing Scheme',
    color: 'green',
    grow: false,
    body: (
      <>
        <div className="tier-box tier1">
          <div className="tier-label" style={{color:'#2e7d32'}}>Tier 1 &mdash; Explicit Open Licenses &amp; Public Domain</div>
          <p>CC0, CC-BY, Apache 2.0, MIT, BSD, public domain. Synthetic data from permissive models &amp; seeds.</p>
        </div>
        <div className="tier-box tier2">
          <div className="tier-label" style={{color:'#e65100'}}>Tier 2 &mdash; Curated Permissive with Upstream Opacity</div>
          <p>(a) Permissive corpora with partial provenance (The Stack V1, Wikipedia). (b) Synthetic data with non-permissive generators (~4% of corpus, isolated for transparency).</p>
        </div>
        <div className="tier-box tier3">
          <div className="tier-label" style={{color:'#1565c0'}}>Tier 3 &mdash; Civic / Governmental Works</div>
          <p>US federal works (statutory public domain), government websites, regulatory notices &mdash; strong public-purpose rationale for reuse.</p>
        </div>
      </>
    ),
  },
  provenance: {
    title: 'License & Provenance Distribution',
    color: 'blue',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/license_provenance_tiers.png" alt="Provenance tiers" /></div>
        <div className="cap"><b>Figure 2(b).</b> Legal provenance tiers of MixtureVitae. All sources fall into permissive-first or risk-mitigated tiers with shard-level annotations.</div></div>
      </>
    ),
  },
  method: {
    title: 'Data Curation Pipeline',
    color: 'teal',
    grow: false,
    body: (
      <>
        <div className="hl"><p><b>Positive inclusion</b> strategy: Instead of retroactive filtering of broad web scrapes, we positively select sources based on auditable permissive status.</p></div>
        <ul>
          <li><b>Permissiveness filtering:</b> Allowlist of governmental domains + permissive license keyword search, excluding restrictive terms</li>
          <li><b>Safety filtering:</b> CSAM &amp; offensive content removal via keyword blocklists; targeted filtering of biographies of living persons</li>
          <li><b>Quality filtering:</b> Removal of base64-encoded text and duplicative headers/footers</li>
          <li><b>Deduplication:</b> Local intra-dataset prefix-based exact matching only &mdash; intentionally no global fuzzy dedup to preserve stylistic and domain diversity (following FineWeb-Edu findings)</li>
          <li><b>Domain-aware mixing:</b> Documents clustered by base URL, sentences concatenated within domain for coherent training examples</li>
        </ul>
      </>
    ),
  },
  synth: {
    title: 'Synthetic & Instruction Data',
    color: 'purple',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/synthetic_composition_breakdown.png" alt="Synthetic breakdown" /></div>
        <div className="cap"><b>Figure 2(a).</b> Composition by data origin. Bars show non-synthetic (human-written), mixed, and fully synthetic proportions per domain.</div></div>
      </>
    ),
  },
  datasources: {
    title: 'Data Sources',
    color: 'teal',
    grow: false,
    body: (
      <>
        <ul>
          <li><b>Web-scale:</b> Nemotron-CC, MGACorpus, FineFineWeb (incl. synthetic rephrased web text)</li>
          <li><b>Curated:</b> SEC EDGAR, MegaWika, TxT360, arXiv, peS2o, PubMed, The Stack v1, USPTO, EuroPat, DM-Math, VALID, YouTube Commons, Open License Corpus</li>
          <li><b>Instruction:</b> Magpie Collection, UltraFeedback, NVIDIA SFT blend, P3 (few-shot &amp; MC format)</li>
          <li><b>Reasoning:</b> Glaive-AI, OpenThoughts, CaseHOLD, OpenScience, OpenManus-RL</li>
          <li><b>Math &amp; Code:</b> MetaMathQA, OpenMathInstruct-2, DART-MATH, Nemo-Math, Prism-Math, LingCoder, StarCoder</li>
        </ul>
      </>
    ),
  },
  results_avg: {
    title: 'Overall Performance (300B Tokens)',
    color: 'orange',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/300B_results_all_plot_clustered_avg.png" alt="Average performance" /></div>
        <div className="cap"><b>Figure 3(a).</b> Average performance across 10 benchmarks for 1.7B models. MixtureVitae outperforms all permissive baselines and approaches DCLM and FineWeb-Edu at 300B tokens.</div></div>
      </>
    ),
  },
  results_mmlu: {
    title: 'MMLU Performance',
    color: 'orange',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/300B_results_mmlu_plot_clustered_avg.png" alt="MMLU performance" /></div>
        <div className="cap"><b>Figure 3(b).</b> Performance on MMLU. MixtureVitae significantly outperforms all baselines; only Nemotron-CC-HQ catches up at ~260B tokens.</div></div>
      </>
    ),
  },
  instruct_table: {
    title: 'Math, Code & Instruction Results',
    color: 'red',
    grow: false,
    body: (
      <>
        <div className="hl hl-orange"><p>1.7B MixtureVitae base model <b>outperforms SmolLM2-1.7B-Instruct</b> on GSM8K, HumanEval, and MBPP &mdash; despite using 36&times; fewer tokens.</p></div>
        <table>
          <thead><tr><th>Dataset</th><th>Tokens</th><th>IF-Eval</th><th>GSM8K</th><th>HumanEval</th><th>MBPP</th><th>Avg</th></tr></thead>
          <tbody>
            <tr style={{fontWeight:700, background:'#e0f2f1'}}><td><b>MixtureVitae</b></td><td>300B</td><td>0.19</td><td className="best">0.53</td><td className="best">0.32</td><td className="best">0.38</td><td className="best">0.36</td></tr>
            <tr><td>Comma-0.1</td><td>300B</td><td>0.19</td><td>0.06</td><td>0.13</td><td>0.22</td><td>0.15</td></tr>
            <tr><td>CommonCorpus</td><td>300B</td><td>0.13</td><td>0.02</td><td>0.05</td><td>0.05</td><td>0.06</td></tr>
            <tr><td>C4</td><td>300B</td><td>0.20</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.06</td></tr>
            <tr><td>SlimPajama</td><td>300B</td><td>0.14</td><td>0.02</td><td>0.05</td><td>0.00</td><td>0.05</td></tr>
            <tr><td>DCLM</td><td>300B</td><td>0.13</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.04</td></tr>
            <tr><td>Nemotron-CC-HQ</td><td>300B</td><td>0.09</td><td>0.03</td><td>0.02</td><td>0.00</td><td>0.03</td></tr>
            <tr className="header-cell"><td colSpan="7" style={{fontWeight:700, fontSize:'calc(7.5pt * var(--font-scale))'}}>Trained for 1T tokens</td></tr>
            <tr><td>FineWeb-Edu</td><td>1T</td><td>0.20</td><td>0.03</td><td>0.00</td><td>0.00</td><td>0.06</td></tr>
            <tr><td>DCLM</td><td>1T</td><td>0.15</td><td>0.03</td><td>0.00</td><td>0.01</td><td>0.05</td></tr>
            <tr className="header-cell"><td colSpan="7" style={{fontWeight:700, fontSize:'calc(7.5pt * var(--font-scale))'}}>Other models (~11T tokens)</td></tr>
            <tr><td>SmolLM2-1.7B</td><td>11T</td><td>0.18</td><td>0.31</td><td>0.01</td><td>0.35</td><td>0.21</td></tr>
            <tr><td>SmolLM2-1.7B-Instruct</td><td>11T</td><td style={{color:'#c62828', fontWeight:700}}>0.28</td><td>0.37</td><td>0.28</td><td>0.37</td><td>0.33</td></tr>
          </tbody>
        </table>
      </>
    ),
  },
  license_chart: {
    title: 'License Distribution',
    color: 'green',
    grow: false,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/mixture_vitae_license_barchart.png" alt="License distribution" /></div>
        <div className="cap"><b>Figure 1(b).</b> Token distribution by governing license. The corpus is composed entirely of data under permissive licenses or public domain.</div></div>
      </>
    ),
  },
  decontam: {
    title: 'Decontamination Analysis',
    color: 'purple',
    grow: false,
    body: (
      <>
        <div className="hl hl-purple"><p>MixtureVitae's strong performance is <b>not an artifact of benchmark leakage</b>. A fully decontaminated model performs slightly <em>better</em> than the original.</p></div>
        <ul>
          <li><b>13-gram overlap scan</b> across all 345M documents; contamination rate negligible for most benchmarks (&le;0.01%)</li>
          <li><b>Original vs. decontaminated test sets:</b> Performance consistent (e.g., GSM8K 0.53 &rarr; 0.54, MBPP 0.38 &rarr; 0.38)</li>
          <li><b>Full corpus decontamination:</b> Retrained 1.7B model after removing all flagged documents &mdash; results <em>improved</em></li>
        </ul>
      </>
    ),
  },
  decontam_fig: {
    title: 'Decontaminated Performance',
    color: 'purple',
    grow: true,
    body: (
      <div className="fig"><div className="fig-wrap"><img src="figures/300B_mxv_decontam.png" alt="Decontamination results" /></div>
      <div className="cap"><b>Figure 4(a).</b> 1.7B model on fully decontaminated MixtureVitae (purple, dashed) vs. full MixtureVitae (green, solid). Decontaminated model performs slightly better, confirming gains are genuine.</div></div>
    ),
  },
  decontam_table: {
    title: 'Original vs. Decontaminated Scores',
    color: 'purple',
    grow: false,
    body: (
      <table>
        <thead><tr><th>Dataset</th><th>GSM8K</th><th>GSM8K-D</th><th>MBPP</th><th>MBPP-D</th><th>IFEval</th><th>IFEval-D</th></tr></thead>
        <tbody>
          <tr style={{fontWeight:700, background:'#f3e5f5'}}><td><b>MixtureVitae</b></td><td className="best">0.53</td><td className="best">0.54</td><td className="best">0.38</td><td className="best">0.38</td><td className="best">0.19</td><td className="best">0.23</td></tr>
          <tr><td>SmolLM2</td><td>0.30</td><td>0.30</td><td>0.35</td><td>0.35</td><td>0.17</td><td>0.20</td></tr>
          <tr><td>Comma-0.1</td><td>0.06</td><td>0.06</td><td>0.21</td><td>0.23</td><td>0.18</td><td>0.20</td></tr>
          <tr><td>DCLM</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.12</td><td>0.13</td></tr>
          <tr><td>FineWeb</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.18</td><td>0.20</td></tr>
          <tr><td>Nemotron-CC-HQ</td><td>0.03</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.09</td><td>0.10</td></tr>
        </tbody>
      </table>
    ),
  },
  ablation: {
    title: 'Ablation Study',
    color: 'blue',
    grow: true,
    body: (
      <>
        <div className="fig"><div className="fig-wrap"><img src="figures/ablation_avg.png" alt="Ablation study" /></div>
        <div className="cap"><b>Figure 5(a).</b> Ablation on full MixtureVitae against two variants, each excluding a data subset. Average performance on 10 downstream tasks. Table: math, code &amp; instruction scores.</div></div>
      </>
    ),
  },
  conclusion: {
    title: 'Conclusion & Key Takeaways',
    color: 'red',
    grow: true,
    body: (
      <>
        <div className="hl" style={{background:'#ffebee', borderLeftColor:'#c62828'}}><p>MixtureVitae demonstrates that <b>permissive-first data with high instruction and reasoning density</b> can provide a practical and risk-mitigated foundation for training capable LLMs.</p></div>
        <ul>
          <li><b>Shift in the compliance&ndash;performance frontier</b> &mdash; Capabilities previously associated with mixed-license corpora are reachable with a permissive-first, risk-mitigated approach</li>
          <li><b>Front-loading instruction &amp; reasoning data into pretraining</b> is more token-efficient than relying on post-training alone &mdash; 36&times; fewer tokens than SmolLM2-Instruct</li>
          <li><b>Three-tier provenance scheme</b> provides a reusable blueprint for constructing legally robust pretraining mixtures with shard-level auditability</li>
          <li><b>Decontamination verified</b> &mdash; Performance holds on decontaminated benchmarks and with contaminated shards removed</li>
          <li><b>Scalable recipe</b> &mdash; Path to multi-trillion token regime via subset upsampling, multilingual expansion, and synthetic growth</li>
        </ul>
        <div className="links">
          <div className="ll">
            <b>Code:</b> https://github.com/ontocord/mixturevitae<br />
            <b>Dataset:</b> https://huggingface.co/datasets/ontocord/MixtureVitae-v1<br />
            <b>Paper:</b> https://arxiv.org/abs/2509.25531
          </div>
        </div>
      </>
    ),
  },
};

const DEFAULT_LAYOUT = {
  columns: [
    { id: 'col1', widthMm: 260, cards: ['tldr', 'method', 'datasources', 'composition', 'provenance'] },
    { id: 'col2', widthMm: null, cards: ['synth', 'results_avg', 'results_mmlu', 'decontam_fig', 'ablation'] },
    { id: 'col3', widthMm: 260, cards: ['instruct_table', 'licensing', 'decontam', 'decontam_table', 'license_chart', 'conclusion'] },
  ],
};
const DEFAULT_CARD_HEIGHTS = {};
const DEFAULT_FONT_SCALE = 2.2;

const DEFAULT_LOGOS = [
  { src: 'logos/ontocord.png', alt: 'Ontocord', heightMm: 24 },
  { src: 'logos/laion_full.png', alt: 'LAION', heightMm: 24 },
  { src: 'logos/openeurollm.png', alt: 'OpenEuroLLM', heightMm: 24 },
  { src: 'logos/fzj_jsc.png', alt: 'FZJ / JSC', heightMm: 24 },
  { src: 'logos/open_sci_collective.png', alt: 'Open-Sci Collective', heightMm: 24 },
  { src: 'logos/eu_funded.png', alt: 'EU Funded', heightMm: 24 },
];

/* ================================================================
   FRAMEWORK CODE
   ================================================================ */

function cloneLayout(layout) {
  return { columns: layout.columns.map(c => ({ id: c.id, widthMm: c.widthMm, cards: [...c.cards] })) };
}

const LS_KEY = 'posterConfig';
function getFullConfig(layout, cardHeights, fontScale, logos) {
  return { columns: layout.columns, cardHeights, fontScale, logos };
}
function loadInitialConfig() {
  try { const s = localStorage.getItem(LS_KEY); if (s) return JSON.parse(s); } catch(e) {}
  return null;
}
const INIT_CONFIG = (() => { const c = loadInitialConfig(); if (c && c.logos && c.logos.length < 6) { c.logos = DEFAULT_LOGOS; } return c; })();

function PosterApp() {
  const [layout, setLayout] = useState(INIT_CONFIG ? cloneLayout({columns: INIT_CONFIG.columns}) : cloneLayout(DEFAULT_LAYOUT));
  const [selectedCard, setSelectedCard] = useState(null);
  const [cardHeights, setCardHeights] = useState(INIT_CONFIG?.cardHeights || {...DEFAULT_CARD_HEIGHTS});
  const [preview, setPreview] = useState(false);
  const [logos, setLogos] = useState(INIT_CONFIG?.logos || DEFAULT_LOGOS);
  const [fontScale, setFontScaleState] = useState(INIT_CONFIG?.fontScale || DEFAULT_FONT_SCALE);
  const currentScaleRef = useRef(1);
  const posterRef = useRef(null);

  useEffect(() => { const cfg = getFullConfig(layout, cardHeights, fontScale, logos); localStorage.setItem(LS_KEY, JSON.stringify(cfg)); }, [layout, cardHeights, fontScale, logos]);
  useEffect(() => { document.documentElement.style.setProperty('--font-scale', fontScale); }, [fontScale]);
  useEffect(() => { document.body.classList.toggle('preview', preview); }, [preview]);

  const fit = useCallback(() => {
    if (window.matchMedia('print').matches) return;
    const pW = POSTER_W_MM * MM, pH = POSTER_H_MM * MM;
    const scale = Math.min(window.innerWidth / pW, window.innerHeight / pH);
    currentScaleRef.current = scale;
    const left = (window.innerWidth - pW * scale) / 2;
    const top = (window.innerHeight - pH * scale) / 2;
    document.body.style.transform = 'translate(' + left + 'px,' + top + 'px) scale(' + scale + ')';
  }, []);
  useEffect(() => { fit(); window.addEventListener('resize', fit); return () => window.removeEventListener('resize', fit); }, [fit]);

  useEffect(() => { if (typeof renderMathInElement === 'function') renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]}); });

  useEffect(() => {
    function handler(e) { if (selectedCard && !e.target.closest('.swap-handle') && !e.target.closest('.drop-zone')) setSelectedCard(null); }
    document.addEventListener('click', handler); return () => document.removeEventListener('click', handler);
  }, [selectedCard]);

  function swapCards(id1, id2) {
    if (id1 === id2) return false;
    setLayout(prev => { const next = cloneLayout(prev); let l1, l2; for (const c of next.columns) { const i1 = c.cards.indexOf(id1); if (i1!==-1) l1={col:c,idx:i1}; const i2 = c.cards.indexOf(id2); if (i2!==-1) l2={col:c,idx:i2}; } if (!l1||!l2) return prev; l1.col.cards[l1.idx]=id2; l2.col.cards[l2.idx]=id1; return next; });
    setCardHeights(prev => { const n={...prev}; delete n[id1]; delete n[id2]; return n; });
    return true;
  }
  function moveCard(cardId, targetColId, position) {
    setLayout(prev => { const next = cloneLayout(prev); for (const c of next.columns) { const i = c.cards.indexOf(cardId); if (i!==-1) { c.cards.splice(i,1); break; } } const tc = next.columns.find(c=>c.id===targetColId); if (!tc) return prev; tc.cards.splice(Math.max(0,Math.min(position,tc.cards.length)),0,cardId); return next; });
    setCardHeights(prev => { const n={...prev}; delete n[cardId]; return n; });
  }
  function setColumnWidth(colId, widthMm) { setLayout(prev => { const next = cloneLayout(prev); const c = next.columns.find(c=>c.id===colId); if (!c) return prev; c.widthMm = widthMm; return next; }); }
  function setCardHeight(cardId, heightMm) { setCardHeights(prev => ({...prev, [cardId]: heightMm})); }
  function getWaste() { let total=0; const details=[]; document.querySelectorAll('.fig-wrap').forEach(fw => { const img=fw.querySelector('img'); if(!img)return; const fr=fw.getBoundingClientRect(),ir=img.getBoundingClientRect(); const wH=Math.abs(fr.height-ir.height),wW=Math.abs(fr.width-ir.width); total+=wH+wW; details.push({card:fw.closest('.card')?.dataset.id,wasteH:Math.round(wH),wasteW:Math.round(wW),pct:Math.round((wH+wW)/(fr.height+fr.width)*100)}); }); return {total:Math.round(total),details}; }
  function getLayout() { const s=currentScaleRef.current; const r=[]; document.querySelectorAll('#poster > .col').forEach(col => { const cards=[]; col.querySelectorAll('.card').forEach(c => { const b=c.getBoundingClientRect(); cards.push({id:c.dataset.id,height:Math.round(b.height/s),heightMm:Math.round(b.height/s/MM),grow:c.classList.contains('grow')}); }); r.push({colId:col.id,widthPx:Math.round(col.getBoundingClientRect().width/s),widthMm:Math.round(col.getBoundingClientRect().width/s/MM),cards}); }); return r; }

  function resetLayout() { setLayout(cloneLayout(DEFAULT_LAYOUT)); setCardHeights({...DEFAULT_CARD_HEIGHTS}); setFontScaleState(DEFAULT_FONT_SCALE); setLogos([...DEFAULT_LOGOS]); setSelectedCard(null); localStorage.removeItem(LS_KEY); }
  function saveConfig() { const cfg=getFullConfig(layout,cardHeights,fontScale,logos); const b=new Blob([JSON.stringify(cfg,null,2)+'\n'],{type:'application/json'}); const u=URL.createObjectURL(b); const a=document.createElement('a'); a.href=u; a.download='poster-config.json'; a.click(); URL.revokeObjectURL(u); }
  function copyConfig() { navigator.clipboard.writeText(JSON.stringify(getFullConfig(layout,cardHeights,fontScale,logos),null,2)).then(()=>alert('Config copied! Paste it to Claude.')); }
  function setFontScale(s) { setFontScaleState(parseFloat(s)||1.25); }

  useEffect(() => { window.posterAPI = { swapCards, moveCard, setColumnWidth, setCardHeight, setFontScale, getWaste, getLayout, getConfig:()=>getFullConfig(layout,cardHeights,fontScale,logos), resetLayout, saveConfig, copyConfig }; });

  function handleSwapClick(cardId, e) { e.stopPropagation(); if (!selectedCard) setSelectedCard(cardId); else if (selectedCard===cardId) setSelectedCard(null); else { swapCards(selectedCard, cardId); setSelectedCard(null); } }
  function handleDropZone(targetColId, position, e) { e.stopPropagation(); if (!selectedCard) return; moveCard(selectedCard, targetColId, position); setSelectedCard(null); }

  function handleColResize(dividerIdx, e) {
    e.preventDefault(); const handle=e.currentTarget; handle.classList.add('active');
    const targetColIdx=dividerIdx===0?0:2; const invert=dividerIdx===0?1:-1;
    const targetEl=document.getElementById(layout.columns[targetColIdx].id);
    const startX=e.clientX; const scale=currentScaleRef.current;
    const startW=targetEl.getBoundingClientRect().width/scale;
    function onMove(ev) { const dx=(ev.clientX-startX)/scale*invert; const newW=Math.max(120,(startW+dx)/MM); setLayout(prev => { const next=cloneLayout(prev); next.columns[targetColIdx].widthMm=Math.round(newW); return next; }); }
    function onUp() { document.removeEventListener('mousemove',onMove); document.removeEventListener('mouseup',onUp); handle.classList.remove('active'); document.body.style.cursor=''; }
    document.body.style.cursor='col-resize'; document.addEventListener('mousemove',onMove); document.addEventListener('mouseup',onUp);
  }

  function handleRowResize(colId, cardAboveIdx, e) {
    e.preventDefault(); const handle=e.currentTarget; handle.classList.add('active');
    const col=layout.columns.find(c=>c.id===colId); const aboveId=col.cards[cardAboveIdx];
    const aboveEl=document.querySelector('[data-id="'+aboveId+'"]'); if(!aboveEl)return;
    const startY=e.clientY; const scale=currentScaleRef.current; const startH=aboveEl.getBoundingClientRect().height/scale;
    setCardHeights(prev=>({...prev,[aboveId]:startH/MM}));
    function onMove(ev) { const dy=(ev.clientY-startY)/scale; setCardHeights(prev=>({...prev,[aboveId]:Math.max(20,(startH+dy)/MM)})); }
    function onUp() { document.removeEventListener('mousemove',onMove); document.removeEventListener('mouseup',onUp); handle.classList.remove('active'); document.body.style.cursor=''; }
    document.body.style.cursor='row-resize'; document.addEventListener('mousemove',onMove); document.addEventListener('mouseup',onUp);
  }

  function isCardGrow(cardId) {
    if (cardHeights[cardId] != null) return false;
    for (const col of layout.columns) {
      const idx = col.cards.indexOf(cardId); if (idx===-1) continue;
      const hasGrower = col.cards.some(cid => CARD_REGISTRY[cid]?.grow && cardHeights[cid]==null);
      if (hasGrower) return CARD_REGISTRY[cardId]?.grow || false;
      const lastFlexible = [...col.cards].reverse().find(cid => cardHeights[cid]==null);
      return cardId === lastFlexible;
    }
    return false;
  }

  function renderCard(cardId) {
    const card = CARD_REGISTRY[cardId]; if (!card) return null;
    const classes = ['card']; if (card.color) classes.push(card.color); if (isCardGrow(cardId)) classes.push('grow'); if (selectedCard===cardId) classes.push('swap-selected');
    const style = {}; if (cardHeights[cardId]!=null) style.flex='0 0 '+cardHeights[cardId]+'mm';
    return (<div key={cardId} className={classes.join(' ')} data-id={cardId} style={style}><div className="swap-handle" onClick={(e)=>handleSwapClick(cardId,e)}>&#x2725;</div><h2>{card.title}</h2>{card.body}</div>);
  }

  function renderDropZone(colId, position) {
    const visible = selectedCard !== null;
    return (<div key={'dz-'+colId+'-'+position} className={'drop-zone'+(visible?' visible':'')} onClick={(e)=>handleDropZone(colId,position,e)}>{visible && <div className="drop-zone-inner" />}</div>);
  }

  function renderColumn(col) {
    const style = col.widthMm != null ? {flex:'0 0 '+col.widthMm+'mm'} : {flex:'1.5'};
    const children = [renderDropZone(col.id, 0)];
    col.cards.forEach((cardId, i) => {
      children.push(renderCard(cardId));
      if (i < col.cards.length - 1) children.push(<div key={'row-'+col.id+'-'+i} className="divider row-resize" onMouseDown={(e)=>handleRowResize(col.id,i,e)} />);
      children.push(renderDropZone(col.id, i + 1));
    });
    return (<div key={col.id} className="col" id={col.id} style={style}>{children}</div>);
  }

  return (
    <>
      {/* Toolbar hidden for final version. Uncomment to re-enable editing:
      <div className="toolbar">
        <button onClick={()=>setPreview(!preview)} style={preview?{background:'#00796b',color:'#fff'}:{}}>{preview?'Edit':'Preview'}</button>
        {!preview && <>
          <button onClick={()=>setFontScale(Math.max(0.8,fontScale-0.1))}>A-</button>
          <button onClick={()=>setFontScale(fontScale+0.1)}>A+</button>
          <button onClick={saveConfig} title="Download poster-config.json">Save</button>
          <button onClick={copyConfig} title="Copy config JSON to clipboard">Copy Config</button>
          <button onClick={resetLayout}>Reset</button>
        </>}
      </div>
      */}

      <div className="header">
        <div className="header-left">
          <h1><span className="m">MixtureVitae:</span> Open Web-Scale Pretraining Dataset With High Quality Instruction and Reasoning Data Built from Permissive Text Sources</h1>
          <div className="authors">
            Huu Nguyen<sup>1,3,4,*</sup>, Victor May<sup>1,*</sup>, Harsh Raj<sup>1,2,4,17,*</sup>, Marianna Nezhurina<sup>1,3,4,5,17</sup>, Yishan Wang<sup>1,6</sup>, Yanqi Luo<sup>7</sup>, Minh Chien Vu<sup>8</sup>, Taishi Nakamura<sup>4,9,17</sup>, Ken Tsui<sup>4,15</sup>, Van Khue Nguyen<sup>10</sup>, David Salinas<sup>11,12,17</sup>, Aleksandra Krasnod&#281;bska<sup>13</sup>, Christoph Schuhmann<sup>3</sup>, Mats Leon Richter<sup>14</sup>, Xuan-Son (Sonny) Vu<sup>16</sup>, Jenia Jitsev<sup>1,3,4,5,17</sup>
          </div>
          <div className="aff"><sup>*</sup>Equal contribution &nbsp; <sup>1</sup>Ontocord &nbsp; <sup>2</sup>Northeastern University &nbsp; <sup>3</sup>LAION &nbsp; <sup>4</sup>Open-&#936; (Open-Sci) Collective &nbsp; <sup>5</sup>JSC, FZJ &nbsp; <sup>6</sup>CMU &nbsp; <sup>7</sup>Salesforce &nbsp; <sup>8</sup>Detomo &nbsp; <sup>9</sup>IST &nbsp; <sup>10</sup>&#201;cole Polytechnique &nbsp; <sup>11</sup>ELLIS T&#252;bingen &nbsp; <sup>12</sup>U. Freiburg &nbsp; <sup>13</sup>NASK &nbsp; <sup>14</sup>MILA, UdeM &nbsp; <sup>15</sup>Independent &nbsp; <sup>16</sup>RSS Lab, LTH / DeepTensor &nbsp; <sup>17</sup>openEuroLLM Team</div>
        </div>
        {logos.length > 0 && (
          <div className="header-logos">
            {logos.map((logo, i) => (<img key={i} src={logo.src} alt={logo.alt} style={{height:(logo.heightMm||24)+'mm'}} />))}
          </div>
        )}
        <div className="header-right">
          <div className="badge">ICLR 2026</div>
        </div>
      </div>

      <div className="poster" id="poster" ref={posterRef}>
        {layout.columns.map((col, colIdx) => {
          const elements = [];
          if (colIdx > 0) elements.push(<div key={'col-div-'+colIdx} className="divider col-resize" onMouseDown={(e)=>handleColResize(colIdx-1,e)} />);
          elements.push(renderColumn(col));
          return elements;
        })}
      </div>
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<PosterApp />);
</script>
</body>
</html>
POSTEREOF

echo "==> Verifying output..."
ACTUAL_MD5=$(md5sum index.html | awk '{print $1}')
echo "index.html checksum: $ACTUAL_MD5"

echo ""
echo "==> Done. index.html generated."
echo "Open in a browser to view. Print to PDF for final output."
