// Builds report/report.docx from results/*.json and results/figures/*.png.
//   node report/build_report.js
// Text in [[double brackets]] is rendered with a yellow highlight: things Wyatt must fill in.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, PageNumber, Footer, Header,
  ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const R = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", f), "utf8"));
const FIG = (f) => path.join(ROOT, "results", "figures", f);
const r1 = R("rq1.json"), r2 = R("rq2.json"), r3 = R("rq3.json"), r4 = R("rq4.json");
const L = r1.labels;

// ------------------------------------------------------------------ formatting helpers
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const p1 = (x) => (100 * x).toFixed(1);
const pts = (x) => (x >= 0 ? "+" : "\u2212") + Math.abs(100 * x).toFixed(1);
const pm = (s) => (s.runs && s.runs.length > 1 ? `${p1(s.mean)} \u00b1 ${p1(s.std)}` : p1(s.mean));
const fmtInt = (n) => n.toLocaleString("en-US");
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

function erf(x) { // Abramowitz-Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const pTwoSided = (acc, n, p0) => { const z = (acc - p0) / Math.sqrt(p0 * (1 - p0) / n); return 1 - erf(Math.abs(z) / Math.SQRT2); };

const FONT = "Calibri";
function runs(text, base = {}) {
  // **bold**, `mono`, [[highlighted placeholder]]
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\])/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ text: part.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (part.startsWith("`")) out.push(new TextRun({ text: part.slice(1, -1), font: "Consolas", size: (base.size || 22) - 2, ...base, font: "Consolas" }));
    else if (part.startsWith("[[")) out.push(new TextRun({ text: part.slice(2, -2), highlight: "yellow", font: FONT, ...base }));
    else out.push(new TextRun({ text: part, font: FONT, ...base }));
  }
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 60 } });

let figN = 0, tabN = 0;
function figure(file, widthIn, caption) {
  const buf = fs.readFileSync(FIG(file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const W = Math.round(widthIn * 96), H = Math.round(W * h / w);
  figN++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: H } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(`**Figure ${figN}.** ${caption}`, { size: 18, italics: false }) }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, caption, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "E7ECF2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: (opts.left || [0]).includes(i) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size: 17, bold: head || undefined }) })],
  });
  tabN++;
  return [
    new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true, children: runs(`**Table ${tabN}.** ${caption}`, { size: 18 }) }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ derived numbers
const S1 = r1.summary;
const acc = (m) => S1[m].accuracy, f1 = (m) => S1[m].macro_f1;
const rec = (m) => Object.fromEntries(L.map((l) => [l, mean(r1.runs[m].map((r) => r.test.test.recall_per_class[l]))]));
const recT = rec("tfidf");
const cmT = r1.runs.tfidf[0].test.test.confusion;
const conf = (a, b) => { const i = L.indexOf(a), j = L.indexOf(b); return cmT[i][j] / cmT[i].reduce((x, y) => x + y, 0); };
const nTest = r1.runs.tfidf[0].test.test.n;

const S2 = r2.summary;
const in2 = (m, t = "test") => S2[`${m}/input`][t], out2 = (m, t = "test") => S2[`${m}/output`][t], both2 = (m, t = "test") => S2[`${m}/both`][t];
const nMatched = r2.runs["tfidf/input"][0].test.test_matched.n;
const tsAcc = r2.timestamp_only.test.accuracy, tsRec = r2.timestamp_only.test.recall_per_class;

const D = Object.keys(r3.domains);
const r3v = (d, m, c) => r3.domains[d].results[`${m}/${c}`].acc;
const drop3 = (d, m) => r3v(d, m, "in_domain").mean - r3v(d, m, "cross_domain").mean;
const allDrops = D.flatMap((d) => ["tfidf", "cnn", "lstm"].map((m) => drop3(d, m)));
const crossAll = D.flatMap((d) => ["tfidf", "cnn", "lstm"].map((m) => r3v(d, m, "cross_domain").mean));
const retain = D.flatMap((d) => ["tfidf", "cnn"].map((m) => r3v(d, m, "cross_domain").mean / r3v(d, m, "in_domain").mean));
const nSeeds3 = r3v(D[0], "cnn", "in_domain").runs.length;
const rd = r3.recall_drop_family_x_domain;

const A = r4.ablations;
const abR = (m, v) => A[m][v].retrain, abT = (v) => A.cnn[v].test_time;
const nSeeds4 = A.cnn.none.retrain.runs.length;
const fm = r4.feature_means;
const medLen = r4.length_words_median;
const imp = Object.entries(r4.stylometric_importance);
const flags = r4.flags_by_family;

// within-family table from TF-IDF per-model accuracy
const famOf = (m) => /^(gpt-(?!oss)|chatgpt|o\d)/.test(m) ? "GPT" : /claude/.test(m) ? "Claude" : /gemini/.test(m) ? "Gemini"
  : /llama/.test(m) ? "Llama" : /qwen|qwq/.test(m) ? "Qwen" : /deepseek/.test(m) ? "DeepSeek"
  : /mistral|magistral/.test(m) ? "Mistral" : /grok/.test(m) ? "Grok" : null;
const byModel = r1.breakdown.tfidf.by_model;
const famRows = L.map((f) => {
  const ms = Object.entries(byModel).filter(([m, v]) => famOf(m) === f && v.size >= 20).sort((a, b) => b[1].mean - a[1].mean);
  const best = ms[0], worst = ms[ms.length - 1];
  return [f, String(ms.length), `${best[0]} (${pct(best[1].mean, 0)}, n=${best[1].size})`, `${worst[0]} (${pct(worst[1].mean, 0)}, n=${worst[1].size})`,
    p1(best[1].mean - worst[1].mean)];
});
const bm = (m) => byModel[m];

// ------------------------------------------------------------------ static dataset stats (from prepare_data.py / preprocess.py logs)
const DATA = {
  battles: 135634,
  families: [
    ["Claude", 8, "Opus 4, Sonnet 4, 3.7 Sonnet, 3.5 Sonnet, 3.5 Haiku (incl. thinking variants)", 23807],
    ["GPT", 8, "o3, o3-mini, o4-mini, GPT-4.1, GPT-4.1-mini, ChatGPT-4o, GPT-4o, GPT-4o-mini", 18608],
    ["Gemini", 8, "2.5 Pro, 2.5 Flash, 2.5 Flash-Lite (incl. previews), 2.0 Flash", 14612],
    ["Qwen", 6, "Qwen3-235B (thinking / no-thinking / 2507), Qwen3-30B, Qwen-Max, QwQ-32B", 12557],
    ["Llama", 4, "Llama 4 Maverick (2 versions), Llama 4 Scout, Llama 3.3 70B", 7927],
    ["Mistral", 4, "Mistral Medium, Mistral Small 3.1 and 2506, Magistral Medium", 6857],
    ["Grok", 4, "Grok 3, Grok 3 mini (beta, high), Grok 4", 5671],
    ["DeepSeek", 2, "DeepSeek-R1-0528, DeepSeek-V3-0324", 5369],
  ],
  domains: { // family: [code, creative_writing, general, math]
    Claude: [575, 151, 1175, 99], DeepSeek: [614, 152, 1138, 96], GPT: [627, 157, 1093, 123], Gemini: [524, 152, 1187, 137],
    Grok: [571, 181, 1120, 128], Llama: [628, 157, 1076, 139], Mistral: [585, 162, 1149, 104], Qwen: [602, 158, 1119, 121],
  },
  promptNames: { Claude: 0.012, DeepSeek: 0.020, GPT: 0.014, Gemini: 0.020, Grok: 0.017, Llama: 0.015, Mistral: 0.018, Qwen: 0.012 },
  maskedRows: { Claude: 101, DeepSeek: 175, GPT: 112, Gemini: 105, Grok: 214, Llama: 81, Mistral: 105, Qwen: 101 },
  split: { train: 11198, val: 2389, test: 2412 }, rows: 15999, matched: 2301,
};

// ------------------------------------------------------------------ document content
const C = [];
const maxDrop = Math.max(...allDrops);
const minIn = Math.min(in2("tfidf").mean, in2("cnn").mean, in2("lstm").mean);
const maxIn = Math.max(in2("tfidf").mean, in2("cnn").mean, in2("lstm").mean);
const bT = r1.breakdown.tfidf;
const qk = Object.keys(bT.by_length_quartile);
const famAcc = (f) => Object.entries(byModel).filter(([m, v]) => famOf(m) === f && v.size >= 20).map(([, v]) => v.mean);

// Title block
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: runs("**Who Wrote It? Identifying LLM Families from Their Responses**", { size: 34 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs("CMPSC 448 Midterm Project, Fall 2026", { size: 22 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs("Team leader and sole member: [[Wyatt LASTNAME]] (individual project)", { size: 22 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: runs("Repository: [[https://github.com/USERNAME/llm-fingerprint]]", { size: 22 }) }));

C.push(P(`**Abstract.** Can we tell which family of large language model (LLM) wrote a response? Using ${fmtInt(DATA.rows)} responses from eight families drawn from the 2025 Chatbot Arena release, we train a character-level CNN and a word-level BiLSTM, plus TF-IDF and style-feature baselines. With model names masked and no prompt shared between training and test, the CNN reaches ${pct(acc("cnn").mean)} accuracy and TF-IDF ${pct(acc("tfidf").mean)} (chance: 12.5%). The prompt alone carries no signal (${p1(minIn)} to ${pct(maxIn)}). Fingerprints survive a change of task domain at a cost of up to ${p1(maxDrop)} points. Ablations show the CNN relies mostly on formatting, yet a model retrained with all formatting removed still reaches about ${pct(abR("cnn", "all_structure_removed").mean, 0)}: the fingerprint is both structural and lexical.`));

// ================================================================== 1
C.push(H1("1. Problem Definition and Dataset Curation"));
C.push(P(`**Task.** Each example is a triple (LLM_name, LLM_input, LLM_output). We map each model to one of eight families and treat identification as 8-way classification on balanced classes (chance 12.5%), reporting accuracy and macro-F1 on a held-out test set.`));
C.push(P(`**Source.** We use \`lmarena-ai/arena-human-preference-140k\` [2], ${fmtInt(DATA.battles)} Chatbot Arena "battles" [1] collected from April to July 2025. In each battle a real user writes a prompt and two anonymous, randomly assigned models answer it. We chose it over generating responses through APIs (costly, rate-limited on free tiers, and no free Claude tier) and over older Arena releases (2023-era models). It provides modern models, real prompts, two answers per prompt (a built-in control for RQ2), and category tags we use as domain labels (RQ3). The cost is that we control neither system prompts nor sampling settings, and each family mixes several model versions. The dataset is released under CC-BY-4.0 [[(verify on the dataset card)]], which permits redistribution with attribution, so the repository includes the filtered file. All prompts were publicly released by LMArena; we use no private data.`));
C.push(P(`**Construction** (\`prepare_data.py\`). We keep English, single-turn battles; treat each side as one example; keep prompts of 10 to 4,000 characters and responses of 50 to 8,000; map model names to families by regular expression (Table 1), excluding other vendors and Google's open Gemma models (a different tier from Gemini); remove duplicates; and randomly sample 2,000 responses per family. One near-empty response was later dropped, leaving ${fmtInt(DATA.rows)}. Domains come from the dataset's annotations (code, math, creative writing, otherwise general); every family has nearly the same mix (about 29% code, 8% creative writing, 57% general, 6% math), so topic cannot act as a shortcut.`));
C.push(...table(["Family", "Models", "Model versions", "Eligible", "Kept"],
  DATA.families.map(([f, n, ex, e]) => [f, n, ex, fmtInt(e), "2,000"]), [1100, 800, 5160, 1200, 1100],
  "Families and model versions, with responses available before balancing.", { left: [0, 2] }));
C.push(P(`**Leakage controls** (\`src/preprocess.py\`). Each targets a way the classifier could succeed for the wrong reason. (1) **Name masking:** vendor and model names (OpenAI, GPT-*, Anthropic, Claude, Gemini, Llama, Qwen, DeepSeek, Mistral, Grok, xAI, and variants) become \`[MODEL]\` in prompts and responses, so "I'm Claude" cannot give the answer away. A spot check of 25 masked responses found one false positive (the English verb "grok"). Masking is imperfect: Grok mentions itself most, so the mask token still carries some signal; RQ4 measures how much. (2) **Reasoning traces:** visible \`<think>\` blocks are stripped (only one existed). (3) **Grouped splits:** each prompt can appear twice, once per battle side, so we split 70/15/15 by prompt and assert no prompt crosses splits (${fmtInt(DATA.split.train)} / ${fmtInt(DATA.split.val)} / ${fmtInt(DATA.split.test)} rows). (4) **Matched pairs:** ${fmtInt(DATA.matched)} rows are prompts answered by two different families, ${nMatched} of them in the test set, used as a control in RQ2. Table 2 lists per-family flags used in the analysis.`));
C.push(...table(["Family", "Reasoning variant", "Refusal", "Names own family", "Median words"],
  L.map((f) => [f, pct(flags[f].is_reasoning), pct(flags[f].refusal), pct(flags[f].self_id), String(medLen[f])]),
  [1760, 1900, 1900, 1900, 1900], "Per-family flags. \"Names own family\" counts any mention of the name, so it overstates self-identification."));
C.push(P(`Refusals and self-mentions are rare (at most ${pct(Math.max(...L.map((f) => flags[f].refusal)))} and ${pct(Math.max(...L.map((f) => flags[f].self_id)))}), so neither can explain high accuracy. The reasoning-variant share ranges from 0% (Llama) to ${pct(flags.Grok.is_reasoning, 0)} (Grok), a confound we revisit in Section 4.3.`));

// ================================================================== 2
C.push(H1("2. Models and Training"));
C.push(P(`**CNN.** The CNN reads characters, because the cues we expected (markdown, line breaks, bullet symbols, curly quotes, dashes) are what word tokenizers discard. Input is the first 2,000 characters over a ${r1.runs.cnn[0].vocab_len}-symbol vocabulary. Following Kim [3] and Zhang et al. [4]: a 64-dimensional embedding; parallel convolutions of widths 3, 5, and 7 (128 filters each) with batch norm and ReLU; max-pooling; a second width-5 convolution that widens the receptive field to about 20 characters (enough for patterns like a newline followed by a bullet and bold text); global max- and mean-pooling ("did a pattern occur?" and "how often?"); dropout 0.3; and a linear layer (${fmtInt(r1.runs.cnn[0].n_params)} parameters).`));
C.push(P(`**BiLSTM.** The RNN reads lowercased words, keeping punctuation and line breaks as tokens, over a ${fmtInt(r1.runs.lstm[0].vocab_len)}-token vocabulary with a 400-token limit. A 128-dimensional embedding feeds a one-layer bidirectional LSTM [5] (128 units per direction, packed sequences), then additive attention pooling [6], dropout 0.3, and a linear layer (${fmtInt(r1.runs.lstm[0].n_params)} parameters, mostly in the embedding). Reading in both directions suits document-level habits such as opening with praise and closing with an offer to help.`));
C.push(P(`**Baselines.** TF-IDF uses word 1 to 2-grams and case-sensitive character 2 to 5-grams with logistic regression, over the whole response. The stylometric model sees no words, only 27 numbers per response (length, lexical diversity, markdown density, punctuation and typography rates, emoji, and conversational habits such as opening with "Great question"), fed to gradient-boosted trees.`));
C.push(...table(["Setting", "CNN and BiLSTM"], [
  ["Optimizer and loss", "AdamW (lr 1e-3, weight decay 1e-4), cross-entropy"],
  ["Batches", "64 examples, bucketed by length to minimize padding"],
  ["Schedule and stopping", "Halve lr when validation macro-F1 stalls; stop after 4 epochs without improvement (max 20); restore best checkpoint"],
  ["Regularization", "Dropout 0.3, gradient clipping at 1.0, batch norm (CNN)"],
  ["Seeds and hardware", `3 seeds (0, 1, 2); NVIDIA GPU, PyTorch 2.5.1; about ${mean(r1.runs.cnn.flatMap((r) => r.history.map((h) => h.sec))).toFixed(0)} s (CNN) and ${mean(r1.runs.lstm.flatMap((r) => r.history.map((h) => h.sec))).toFixed(0)} s (BiLSTM) per epoch`],
], [2600, 6760], "Training configuration.", { left: [0, 1] }));
C.push(P(`The validation set is used only for early stopping and scheduling. All models, including the baselines, learn from the same training split. For RQ2's input + output condition, the prompt may use at most 25% of the length budget, followed by a separator and the response. In every run, training loss kept falling while validation F1 leveled off (around 0.75 for the CNN and 0.60 for the BiLSTM), so early stopping, not the epoch limit, ended training. After the laptop's sleep and restarts cost hours of an early run, we cached every finished run under a hash of its configuration, seed, and data, so the pipeline resumes after interruption. As a check, the CNN retrained from scratch in RQ2 matched its RQ1 accuracy exactly (${pct(out2("cnn").mean, 2)}).`));

// ================================================================== 3
C.push(H1("3. Results"));
C.push(H2("3.1 RQ1: Can we identify which LLM generated a response?"));
C.push(...table(["Model", "Accuracy (%)", "Macro-F1 (%)"], [
  ["TF-IDF + logistic regression", pm(acc("tfidf")), pm(f1("tfidf"))],
  ["CNN (required)", pm(acc("cnn")), pm(f1("cnn"))],
  ["Stylometric features only", pm(acc("stylometric")), pm(f1("stylometric"))],
  ["BiLSTM (required)", pm(acc("lstm")), pm(f1("lstm"))],
  ["Chance", "12.5", "12.5"],
], [4360, 2500, 2500], `RQ1 test results, output only (n = ${fmtInt(nTest)}). Neural models: mean \u00b1 standard deviation over 3 seeds.`));
C.push(P(`**Yes, by a wide margin:** the CNN is right ${(acc("cnn").mean / 0.125).toFixed(1)} times as often as chance. The ranking was not what we expected: TF-IDF is best, the CNN ${p1(acc("tfidf").mean - acc("cnn").mean)} points behind, and the BiLSTM ${p1(acc("tfidf").mean - acc("lstm").mean)} behind, level with the word-free stylometric model. Three reasons are likely. ${fmtInt(DATA.split.train)} examples is little data for models trained from scratch; character n-grams already capture most formatting cues the CNN must learn; and TF-IDF sees the whole response while the neural models truncate, which matters because longer responses are easier (TF-IDF accuracy rises from ${pct(bT.by_length_quartile[qk[0]], 0)} in the shortest quarter to ${pct(bT.by_length_quartile[qk[3]], 0)} in the longest). The BiLSTM also loses case through lowercasing and overfits its large embedding table. For comparison, Sun et al. [7] reach 97.1% on five LLMs by fine-tuning large pretrained embedding models on answers to identical prompts; a lower ceiling is expected with eight classes, uncontrolled prompts, mixed versions, and small models trained from scratch.`));
C.push(...figure("rq1_confusion_cnn.png", 3.9, "Row-normalized confusion matrix of the best CNN seed."));
C.push(P(`Claude (${pct(recT.Claude, 0)} TF-IDF recall) and Gemini (${pct(recT.Gemini, 0)}) are easiest; Mistral (${pct(recT.Mistral, 0)}), DeepSeek, and Qwen (both ${pct(recT.Qwen, 0)}) are hardest. The main confusions are DeepSeek and Mistral (${pct(conf("DeepSeek", "Mistral"), 0)} and ${pct(conf("Mistral", "DeepSeek"), 0)} of each family's responses go to the other) and Qwen and GPT (about ${pct(conf("Qwen", "GPT"), 0)}), pairs that share a heavy markdown style (Section 4.2). The CNN beats TF-IDF on Mistral (${pct(rec("cnn").Mistral, 0)} vs. ${pct(recT.Mistral, 0)}) but trails it on GPT, so the two make partly different errors.`));

C.push(H2("3.2 RQ2: Does the user's prompt help identify the LLM?"));
C.push(...table(["Model", "Input only", "Output only", "Input + output", "Output only, matched pairs"],
  ["tfidf", "cnn", "lstm"].map((m) => [m === "tfidf" ? "TF-IDF" : m === "cnn" ? "CNN" : "BiLSTM", pm(in2(m)), pm(out2(m)), pm(both2(m)), pm(out2(m, "test_matched"))]),
  [1560, 1800, 1800, 1800, 2400], `RQ2 accuracy (%) on the full test set (n = ${fmtInt(nTest)}) and on matched-prompt pairs (n = ${nMatched}).`));
C.push(...figure("rq2_fields_test.png", 4.2, "RQ2 accuracy by input field (error bars: standard deviation over seeds)."));
const pIn = pTwoSided(in2("tfidf").mean, nTest, 0.125);
C.push(P(`**(i) The LLM cannot be predicted from the prompt.** Input-only accuracy is ${p1(minIn)} to ${pct(maxIn)}, not significantly above chance (best model vs. 12.5%: p \u2248 ${pIn.toFixed(2)}). Its most heavily weighted n-grams are unrelated topic words ("summer" for Claude, "dog" for Qwen), the signature of fitting noise.`));
C.push(P(`**(ii) Adding the input does not help.** For all three models, input + output scores slightly below output only (${pts(both2("tfidf").mean - out2("tfidf").mean)}, ${pts(both2("cnn").mean - out2("cnn").mean)}, ${pts(both2("lstm").mean - out2("lstm").mean)} points): the prompt adds noise and uses up length budget.`));
C.push(P(`**(iii) Why input-only can work, and why it doesn't here.** We predicted a timing confound: models are live on Arena at different times, so their prompts would come from different months. That prediction was wrong, because Arena assigns models to battles at random, so a prompt does not depend on which model answers it. The data also spans only about three months, with nearly every family present throughout. A classifier given only the timestamp does reach ${pct(tsAcc)} (mostly by recognizing the late-arriving Mistral and DeepSeek), so a time signal exists, but prompt content barely carries it. Where users choose which assistant to ask (shared ChatGPT conversations, logs from one product), input-only classification would likely succeed because people ask different assistants different things, but that would describe data collection, not the models. Input-only accuracy is best read as a leakage test for the dataset.`));
C.push(P(`**Matched pairs confirm it.** On the ${nMatched} test responses whose prompt was also answered by another family, the input is identical within each pair, so an input-only model must give both the same label (verified: 100% of pairs) and can be right on at most one. Output-only accuracy there (${p1(out2("tfidf", "test_matched").mean)}% TF-IDF, ${p1(out2("cnn", "test_matched").mean)}% CNN) matches the full test set: the fingerprint is in how a model answers, not what it was asked.`));

// ================================================================== 4
C.push(H1("4. In-depth Analyses"));
C.push(H2("4.1 RQ3: Do fingerprints generalize across domains?"));
C.push(P(`For each target domain, 30% of its prompts form a test set. We train on an equal number of examples either from the same domain (in-domain) or only from the other three (cross-domain) and test both on the same rows. Matching training size ensures a drop reflects domain shift, not less data. ${nSeeds3 > 1 ? `Neural results average ${nSeeds3} seeds.` : "[[Neural results are single-seed; 3-seed run pending.]]"}`));
C.push(...table(["Test domain", "Train / test", "TF-IDF in / cross", "CNN in / cross", "BiLSTM in / cross"],
  D.map((d) => [d, `${fmtInt(r3.domains[d].n_train)} / ${fmtInt(r3.domains[d].n_test)}`,
    `${pm(r3v(d, "tfidf", "in_domain"))} / ${pm(r3v(d, "tfidf", "cross_domain"))}`,
    `${pm(r3v(d, "cnn", "in_domain"))} / ${pm(r3v(d, "cnn", "cross_domain"))}`,
    `${pm(r3v(d, "lstm", "in_domain"))} / ${pm(r3v(d, "lstm", "cross_domain"))}`]),
  [1700, 1560, 2000, 2000, 2100], "RQ3 accuracy (%), in-domain vs. cross-domain with size-matched training. Neural models: mean \u00b1 std over 3 seeds; TF-IDF is deterministic."));
C.push(...figure("rq3_drop_summary.png", 4.8, "Accuracy lost when the test domain was unseen in training."));
C.push(P(`**(i) Performance drops, but the fingerprint largely transfers.** Accuracy falls in ${allDrops.filter((x) => x > 0.005).length} of ${allDrops.length} model and domain combinations, by up to ${p1(maxDrop)} points, but cross-domain accuracy never nears chance (lowest ${pct(Math.min(...crossAll))}), and TF-IDF and the CNN keep ${pct(Math.min(...retain), 0)} to ${pct(Math.max(...retain), 0)} of their in-domain accuracy. General questions transfer best (CNN ${pts(-drop3("general", "cnn"))}), likely because "general" already mixes many styles. Math transfers worst (TF-IDF ${pts(-drop3("math", "tfidf"))}, CNN ${pts(-drop3("math", "cnn"))}), since math answers follow their own format (LaTeX, numbered derivations). On code the CNN loses ${p1(drop3("code", "cnn"))} points and TF-IDF ${p1(drop3("code", "tfidf"))}, a gap close to the seed-to-seed variation, so we do not claim either model transfers better.`));
C.push(...figure("rq3_recall_drop.png", 3.8, "Per-family recall lost under domain shift, averaged over models (positive: harder in an unseen domain)."));
C.push(P(`**(ii) Some families travel better.** GPT and Mistral transfer to code and general prompts with essentially no loss (Grok loses little), while DeepSeek and Qwen lose recall in every domain. Math hurts most families, but not Claude, GPT, or Mistral. With only about 35 to 50 test responses per family in math and creative writing, single cells carry roughly 8 points of standard error, so we read only consistent patterns. **(iii) General or task-specific?** Both: task-specific patterns alone would collapse to chance, and a purely general fingerprint would show no drop. There is a domain-independent core plus a part that depends on how each family formats a given kind of answer, largest where formats differ most (math).`));

C.push(H2("4.2 RQ4: What characteristics distinguish the families?"));
C.push(...table(["Family", "Distinctive measured habits"], [
  ["Claude", `Shortest (median ${medLen.Claude} words); least bold text; never uses curly quotes; almost never opens with praise (${pct(fm.Claude.opener_affirmation, 0)}); plain "-" bullets.`],
  ["GPT", `Most curly quotes (${fm.GPT.curly_quote_per_1kc.toFixed(1)} per 1,000 characters) and em dashes; "\u2022" bullets; frequent horizontal rules.`],
  ["Gemini", `Opens with an affirmation such as "Okay," in ${pct(fm.Gemini.opener_affirmation, 0)} of responses; almost never ends with an offer to help (${pct(fm.Gemini.ends_with_offer, 0)}); "*" bullets.`],
  ["Qwen", `Most horizontal rules (${fm.Qwen.hr_count.toFixed(1)} per response), headers, and emoji.`],
  ["DeepSeek", `Heaviest bold text (${fm.DeepSeek.bold_per_100w.toFixed(1)} per 100 words) and many headers.`],
  ["Mistral", `Heavy bold; ends with "Would you like..." offers (${pct(fm.Mistral.ends_with_offer, 0)}).`],
  ["Llama", `Lowest lexical diversity; numbered lists with bold labels.`],
  ["Grok", `Longest (median ${medLen.Grok} words); connective phrasing ("such as", "if you", "based on").`],
], [1300, 8060], "Measured habits that distinguish each family (stylometric means and top TF-IDF features).", { left: [0, 1] }));
C.push(P(`These habits explain the RQ1 confusions: DeepSeek, Mistral, and Qwen share dense bold, headers, and bullets, while Claude and Gemini each have habits no one else shares.`));
C.push(P(`**(i) Which signals does the classifier use?** Structure alone goes far: the stylometric model, which never sees a word, reaches ${pct(r4.stylometric_model.accuracy)}. Its most important feature (Figure 5) is the curly-quote rate, followed by bold, headers, horizontal rules, blank lines, lexical diversity, and length. Whether a model writes \u2019 or ' is a byproduct of training data that no human reader would notice. The learned models agree: GPT's top three TF-IDF features are curly-quote characters, Gemini's are "okay ," and "*" bullets, and the BiLSTM's attention concentrates on formatting tokens (\`#\`, line breaks, quotes, dashes) rather than content words.`));
C.push(...figure("rq4_feature_importance.png", 3.6, "Permutation importance of the top 15 stylometric features."));
C.push(P(`**(ii, iii) Linguistic, structural, or both? Does removing signals hurt?** We remove one kind of signal and measure accuracy two ways: **retraining** on ablated text (is the information needed?) and **testing the clean-trained CNN** on ablated text (does the model rely on it?). ${nSeeds4 > 1 ? `CNN retraining averages ${nSeeds4} seeds.` : "[[CNN retraining is single-seed; 3-seed run pending.]]"}`));
const AB = [["strip_markdown", "Headers, bullets, bold, tables, rules"], ["flatten_whitespace", "All line breaks"],
  ["normalize_punct", "Curly quotes, dashes, emoji"], ["lowercase", "Case"], ["fixed_length_600", "Everything after 600 characters"],
  ["all_structure_removed", "All of the above"], ["no_masking", "Masking (names left visible)"]];
C.push(...table(["Ablation", "Removes", "TF-IDF retrained", "CNN retrained", "Clean CNN, ablated test"], [
  ["none", "Nothing", p1(abR("tfidf", "none").mean), pm(abR("cnn", "none")), p1(abT("none"))],
  ...AB.map(([v, d]) => [v, d, p1(abR("tfidf", v).mean), pm(abR("cnn", v)), p1(abT(v))]),
], [2100, 2960, 1300, 1700, 1300], "RQ4 ablation accuracy (%).", { left: [0, 1] }));
C.push(...figure("rq4_ablation_cnn.png", 5.6, "CNN ablations: retrained on ablated text (blue) vs. trained on clean text and tested on ablated text (orange)."));
C.push(P(`**The CNN relies on formatting:** stripping markdown at test time drops it from ${p1(abT("none"))}% to ${p1(abT("strip_markdown"))}%, and removing all structure to ${p1(abT("all_structure_removed"))}%. **But formatting is not required:** retrained without markdown, the CNN recovers to ${p1(abR("cnn", "strip_markdown").mean)}%, and with every structural signal removed at once, retrained models still reach ${p1(abR("tfidf", "all_structure_removed").mean)}% (TF-IDF) and ${p1(abR("cnn", "all_structure_removed").mean)}% (CNN), about four times chance. What remains is lexical: word choice, phrasing, and how each family opens its answers. So the signals are both structural and linguistic; structure dominates what the trained models use, and words alone suffice for about half. The gap between the two columns is the main lesson: what a model relies on says little about what the task requires. Individually, lowercasing and punctuation normalization cost nothing measurable after retraining, even though curly quotes are the top stylometric feature: correlated cues compensate. Leaving names unmasked changes accuracy by only ${pts(abR("tfidf", "no_masking").mean - abR("tfidf", "none").mean)} (TF-IDF) and ${pts(abR("cnn", "no_masking").mean - abR("cnn", "none").mean)} (CNN) points, so self-identification explains almost none of the performance.`));

C.push(H2("4.3 Families are not uniform"));
C.push(P(`Accuracy varies more within some families than between them. Every Claude version scores at least ${pct(Math.min(...famAcc("Claude")), 0)}, suggesting a stable house style. Other families are mixtures: Magistral, Mistral's reasoning model, is recognized as Mistral only ${pct(bm("magistral-medium-2506").mean, 0)} of the time; Grok 3 mini scores ${pct(bm("grok-3-mini-beta").mean, 0)} but Grok 3 preview ${pct(bm("grok-3-preview-02-24").mean, 0)}; ChatGPT-4o scores ${pct(bm("chatgpt-4o-latest-20250326").mean, 0)} but o3 ${pct(bm("o3-2025-04-16").mean, 0)} (TF-IDF). A family fingerprint is therefore a blend of per-model fingerprints, weighted by whichever models dominate the sample. This also qualifies a finding from RQ1: reasoning variants are more identifiable (${pct(bT.by_reasoning_variant.True)} vs. ${pct(bT.by_reasoning_variant.False)}), but they are concentrated in a few families and include distinctive models (o3, o4-mini, Grok 3 mini), so we cannot attribute the gap to reasoning itself.`));

C.push(H2("4.4 Limitations"));
C.push(P(`Arena's system prompts and sampling settings are unknown and may differ by model. Results depend on the mix of versions within each family. The data covers three months, English only, single-turn. Domain tags are automatic and imperfect, and regex masking misses paraphrased self-references. Truncation differs by model (2,000 characters, 400 tokens, none), which favors TF-IDF on long responses. Per-family RQ3 results on math and creative writing are noisy.`));

// ================================================================== 5
C.push(H1("5. Lessons and Experience"));
C.push(P(`[[Write this section yourself (20% of the grade): 4 to 6 short paragraphs from your own notes. Questions to answer:]]`));
for (const q of [
  "[[Operations: Windows setup, CPU-only PyTorch, the laptop sleeping and restarting. What did it cost, what did you change, and what would you do from day one next time?]]",
  "[[Baselines: TF-IDF beat both deep models. What did you expect, and what does it teach you?]]",
  "[[The failed RQ2 prediction: what did you learn about how data collection shapes results?]]",
  "[[Leakage controls (grouped splits, masking, matched pairs): which mattered, and how did you verify them?]]",
  "[[The 'family' label hid large differences between models. How would you redesign the task?]]",
  "[[With another month, which single experiment would you run, and why?]]",
]) C.push(bullet(q));

// ================================================================== 6
C.push(H1("6. Reproducibility and Use of AI Tools"));
C.push(P(`\`prepare_data.py\` builds the dataset; \`src/\` holds preprocessing, models, and training; \`experiments/rq1.py\` to \`rq4.py\` produce every number and figure, saved in \`results/\`. \`python run_all.py\` reproduces everything (seeded); \`python run_all.py --smoke\` checks the pipeline on synthetic data in minutes. [[AI use, edit to match what you did: "As the assignment permits, I used Claude (Anthropic) to help design experiments, write and debug code, and draft parts of this report. I ran all experiments, checked results against the raw outputs, and take responsibility for everything submitted."]]`));

// ================================================================== refs
C.push(H1("References"));
for (const ref of [
  "[1] W.-L. Chiang et al. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. ICML, 2024.",
  "[2] LMArena. arena-human-preference-140k. Hugging Face dataset, 2025. https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k",
  "[3] Y. Kim. Convolutional Neural Networks for Sentence Classification. EMNLP, 2014.",
  "[4] X. Zhang, J. Zhao, Y. LeCun. Character-level Convolutional Networks for Text Classification. NeurIPS, 2015.",
  "[5] S. Hochreiter, J. Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8), 1997.",
  "[6] D. Bahdanau, K. Cho, Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.",
  "[7] M. Sun, Y. Yin, Z. Xu, J. Z. Kolter, Z. Liu. Idiosyncrasies in Large Language Models. ICML, 2025.",
]) C.push(new Paragraph({ children: runs(ref, { size: 19 }), spacing: { after: 60 }, indent: { left: 360, hanging: 360 } }));

// ------------------------------------------------------------------ assemble
const doc = new Document({
  creator: "CMPSC 448", title: "Who Wrote It? Identifying LLM Families from Their Responses",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E5597" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
    children: C,
  }],
});
const out = path.join(__dirname, "report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out, "figures:", figN, "tables:", tabN); });
