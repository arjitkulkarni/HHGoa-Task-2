/**
 * RAGA front end.
 *
 * Two entry points — hold the mic to send audio to /api/voice, or type and hit
 * /api/ask — and one renderer, because a voice answer and a typed answer are
 * the same object with a transcript attached.
 *
 * The trace panel is not decoration. The 200 ms claim is only meaningful if you
 * can see which stages it covers, so every stage the server timed is drawn to
 * scale against the budget, with the two network stages (speech-to-text and the
 * optional LLM rewrite) drawn in a different colour and excluded from the total.
 */

const $ = (id) => document.getElementById(id);

const el = {
  mic: $('mic'),
  micHint: $('mic-hint'),
  form: $('ask-form'),
  question: $('question'),
  submit: document.querySelector('.ask__submit'),
  language: $('language'),
  rewrite: $('rewrite'),
  status: $('status'),
  corpusMeta: $('corpus-meta'),
  examples: $('examples'),
  examplesList: $('examples-list'),
  answerPanel: $('answer-panel'),
  verdict: $('verdict'),
  transcript: $('transcript'),
  answer: $('answer'),
  rewriteBlock: $('rewrite-block'),
  rewriteText: $('rewrite-text'),
  grounding: $('grounding'),
  sources: $('sources'),
  tracePanel: $('trace-panel'),
  budgetFill: $('budget-fill'),
  budgetTotal: $('budget-total'),
  budgetNote: $('budget-note'),
  stages: $('stages').querySelector('tbody'),
  signals: $('signals'),
  indexPanel: $('index-panel'),
  indexLede: $('index-lede'),
  indexFoot: $('index-foot'),
  strategies: $('strategies').querySelector('tbody'),
};

const BUDGET_MS = 200;
/** Stages that leave the process. Timed and shown, never counted in the budget. */
const NETWORK_STAGES = new Set(['stt', 'answer.rewrite']);

let capabilities = { speechToText: false, llmRewrite: false };
let busy = false;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );

const setStatus = (message, isError = false) => {
  el.status.textContent = message;
  el.status.classList.toggle('status--error', isError);
};

const setBusy = (value, label) => {
  busy = value;
  el.submit.disabled = value;
  el.mic.dataset.state = value ? 'busy' : 'idle';
  if (label) setStatus(label);
};

const fmt = (n, digits = 1) => Number(n).toFixed(digits);

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function loadMeta() {
  try {
    const response = await fetch('/api/meta');
    if (!response.ok) throw new Error(`meta ${response.status}`);
    const meta = await response.json();

    capabilities = meta.capabilities;

    el.corpusMeta.innerHTML = [
      `<span class="pill">${meta.index.chunks.toLocaleString()} chunks</span>`,
      `<span class="pill">${meta.index.passages.toLocaleString()} passages</span>`,
      `<span class="pill">${meta.index.dims}d · int8 + 1-bit</span>`,
      capabilities.speechToText
        ? '<span class="pill pill--live">voice ready</span>'
        : '<span class="pill pill--off">voice not configured</span>',
    ].join('');

    if (!capabilities.speechToText) {
      el.micHint.textContent = 'Voice needs SARVAM_API_KEY — type instead';
    }
    if (!capabilities.llmRewrite) {
      el.rewrite.disabled = true;
      el.rewrite.closest('.control').title = 'Set ANTHROPIC_API_KEY to enable';
    }

    renderIndexCard(meta);
    renderExamples(meta.examples ?? []);
  } catch (error) {
    setStatus(`Could not load index metadata: ${error.message}`, true);
  }
}

function renderIndexCard(meta) {
  const totalProposed = meta.index.totalProposed;
  const collapsed = totalProposed > 0 ? 1 - meta.index.chunks / totalProposed : 0;

  el.indexLede.textContent =
    `Six strategies cut every passage independently. Where two or more landed on the same ` +
    `span, it is stored once carrying each of their bits — which is how ${totalProposed.toLocaleString()} ` +
    `proposals became ${meta.index.chunks.toLocaleString()} retrieval units. "Shared" is how ` +
    `often another strategy independently agreed.`;

  el.strategies.innerHTML = meta.strategies
    .map((s) => {
      const pct = s.spans > 0 ? ((s.shared / s.spans) * 100).toFixed(0) : '0';
      return (
        `<tr><td>${escapeHtml(s.label)}</td><td>${s.spans.toLocaleString()}</td>` +
        `<td>${s.shared.toLocaleString()} <span style="opacity:.55">(${pct}%)</span></td></tr>`
      );
    })
    .join('');

  el.indexFoot.textContent =
    `${(collapsed * 100).toFixed(1)}% of proposals collapsed into a shared span · ` +
    `${(meta.index.bytes.total / 1e6).toFixed(1)} MB on disk · ${meta.index.model}`;

  el.indexPanel.hidden = false;
}

function renderExamples(examples) {
  if (examples.length === 0) return;
  // Rotate the sample per load so a repeat visitor does not see the same five.
  const shuffled = [...examples].sort(() => Math.random() - 0.5).slice(0, 5);

  el.examplesList.innerHTML = shuffled
    .map(
      (example) =>
        `<button class="chip" type="button" data-q="${escapeHtml(example.question)}">` +
        `${escapeHtml(example.question)}</button>`,
    )
    .join('');

  el.examplesList.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (!chip || busy) return;
    el.question.value = chip.dataset.q;
    ask(chip.dataset.q);
  });

  el.examples.hidden = false;
}

// ---------------------------------------------------------------------------
// asking
// ---------------------------------------------------------------------------
async function ask(question) {
  if (!question.trim() || busy) return;
  setBusy(true, 'retrieving…');
  const startedAt = performance.now();

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        question,
        rewrite: el.rewrite.checked && capabilities.llmRewrite,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message ?? `request failed (${response.status})`);

    render(payload, performance.now() - startedAt);
    setStatus('');
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function askWithAudio(blob) {
  setBusy(true, 'transcribing…');
  const startedAt = performance.now();

  const params = new URLSearchParams({
    language: el.language.value,
    // Saaras can transcribe or translate. Translating means a Hindi or Tamil
    // question lands in the same language as the index, with no extra hop.
    mode: el.language.value === 'en-IN' ? 'transcribe' : 'translate',
    rewrite: el.rewrite.checked && capabilities.llmRewrite ? '1' : '0',
  });

  try {
    const response = await fetch(`/api/voice?${params}`, {
      method: 'POST',
      headers: { 'content-type': blob.type || 'audio/webm' },
      body: blob,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message ?? `request failed (${response.status})`);

    render(payload, performance.now() - startedAt);
    setStatus('');
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------
function render(result, roundTripMs) {
  el.answerPanel.hidden = false;
  el.tracePanel.hidden = false;

  // verdict
  const status = result.status ?? 'ABSTAINED';
  el.verdict.textContent = status;
  el.verdict.className = `verdict verdict--${status.toLowerCase()}`;

  // transcript
  if (result.transcription?.transcript) {
    const { transcript, languageCode, mode } = result.transcription;
    el.transcript.innerHTML =
      `<span class="transcript__label">heard${languageCode ? ` · ${escapeHtml(languageCode)}` : ''}` +
      `${mode === 'translate' ? ' · translated to English' : ''}</span>` +
      `<p class="transcript__text">${escapeHtml(transcript)}</p>`;
    el.transcript.hidden = false;
  } else {
    el.transcript.hidden = true;
  }

  // answer or refusal
  el.answer.textContent = result.answer || result.refusal?.message || '';

  // rewrite
  if (result.rewritten?.text) {
    el.rewriteText.textContent = result.rewritten.text;
    el.rewriteBlock.querySelector('.rewrite__label').textContent =
      `LLM rewrite · ${result.rewritten.model} · ${fmt(result.rewritten.ms, 0)} ms (off-budget)`;
    el.rewriteBlock.hidden = false;
  } else {
    el.rewriteBlock.hidden = true;
  }

  renderGrounding(result);
  renderSources(result);
  renderTrace(result, roundTripMs);
  renderSignals(result);
}

function renderGrounding(result) {
  if (!result.grounding) {
    el.grounding.hidden = true;
    return;
  }
  const g = result.grounding;
  const chips = [
    `<span class="pill">lexical support ${(g.lexicalSupport * 100).toFixed(0)}%</span>`,
    g.unsupportedNumerics.length === 0
      ? '<span class="pill pill--live">every figure traced to a source</span>'
      : `<span class="pill pill--off">${g.unsupportedNumerics.length} unverified figure(s)</span>`,
    `<span class="pill">${result.citations.length} citation${result.citations.length === 1 ? '' : 's'}</span>`,
  ];
  el.grounding.innerHTML = chips.join('');
  el.grounding.hidden = false;
}

function renderSources(result) {
  const passages = result.context?.passages ?? [];
  if (passages.length === 0) {
    el.sources.innerHTML = '';
    return;
  }

  const chunksByPassage = new Map();
  for (const chunk of result.context.chunks ?? []) {
    if (!chunksByPassage.has(chunk.passageId)) chunksByPassage.set(chunk.passageId, []);
    chunksByPassage.get(chunk.passageId).push(chunk);
  }

  el.sources.innerHTML = passages
    .map((passage) => {
      const cites = (result.citations ?? []).filter((c) => c.passageId === passage.id);
      const chunks = chunksByPassage.get(passage.id) ?? [];
      const strategies = [...new Set(chunks.flatMap((c) => strategyNames(c.strategyMask)))];

      const tags = [
        ...strategies.map((name) => `<span class="tag">${escapeHtml(name)}</span>`),
        chunks[0]
          ? `<span class="tag tag--score">score ${fmt(chunks[0].score, 3)}</span>`
          : '',
      ].join('');

      return (
        `<article class="source">` +
        `<div class="source__head">${tags}</div>` +
        `<p class="source__text">${highlight(passage.text, cites)}</p>` +
        (passage.alt
          ? `<p class="source__alt">${escapeHtml(passage.alt)}</p>`
          : '') +
        `</article>`
      );
    })
    .join('');
}

/** Wraps the cited character ranges of a passage in <mark>, escaping as it goes. */
function highlight(text, citations) {
  if (citations.length === 0) return escapeHtml(text);

  const ranges = citations
    .map((c) => ({ start: c.start, end: c.end }))
    .sort((a, b) => a.start - b.start);

  // merge overlaps so nested marks cannot be produced
  const merged = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && range.start <= last.end) last.end = Math.max(last.end, range.end);
    else merged.push({ ...range });
  }

  let html = '';
  let cursor = 0;
  for (const range of merged) {
    html += escapeHtml(text.slice(cursor, range.start));
    html += `<mark>${escapeHtml(text.slice(range.start, range.end))}</mark>`;
    cursor = range.end;
  }
  return html + escapeHtml(text.slice(cursor));
}

const STRATEGY_BITS = [
  'whole passage',
  'fixed window',
  'sentence window',
  'semantic split',
  'structural split',
  'proposition',
];

const strategyNames = (mask) => STRATEGY_BITS.filter((_, i) => (mask & (1 << i)) !== 0);

function renderTrace(result, roundTripMs) {
  const stages = result.trace?.stages ?? [];
  const pipelineMs = result.trace?.pipelineMs ?? 0;

  const widest = Math.max(...stages.map((s) => s.ms), 1);

  el.stages.innerHTML = stages
    .map((stage) => {
      const isNetwork = NETWORK_STAGES.has(stage.name);
      const isSub = stage.name.startsWith('retrieve.');
      const width = Math.max(2, (stage.ms / widest) * 100);
      const retries =
        stage.attempts && stage.attempts > 1 ? ` <span title="retries">×${stage.attempts}</span>` : '';
      return (
        `<tr data-network="${isNetwork}" data-sub="${isSub}">` +
        `<td>${escapeHtml(stage.name)}${retries}</td>` +
        `<td>${fmt(stage.ms, stage.ms < 10 ? 2 : 1)}</td>` +
        `<td><span class="bar" style="width:${width}%"></span></td>` +
        `</tr>`
      );
    })
    .join('');

  const pct = Math.min(100, (pipelineMs / BUDGET_MS) * 100);
  el.budgetFill.style.width = `${pct}%`;
  el.budgetFill.dataset.over = String(pipelineMs > BUDGET_MS);
  el.budgetTotal.textContent = `${fmt(pipelineMs)} ms`;

  const network = stages
    .filter((s) => NETWORK_STAGES.has(s.name))
    .reduce((sum, s) => sum + s.ms, 0);

  el.budgetNote.textContent =
    `in-process pipeline · ${(pct).toFixed(0)}% of the 200 ms budget` +
    (network > 0 ? ` · +${fmt(network, 0)} ms network (excluded)` : '') +
    (roundTripMs ? ` · ${fmt(roundTripMs, 0)} ms browser round trip` : '');
}

function renderSignals(result) {
  const rows = [];

  if (result.routing?.queryType) {
    rows.push(['query type', `${result.routing.queryType} · confidence ${fmt(result.routing.confidence, 2)}`]);
    const weights = result.routing.strategyWeights ?? {};
    const favoured = Object.entries(weights)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([name, weight]) => `${name.toLowerCase().replace(/_/g, ' ')} ×${fmt(weight, 2)}`)
      .join(', ');
    if (favoured) rows.push(['routed toward', favoured]);
  }

  rows.push(['chunks retrieved', String(result.context?.chunks?.length ?? 0)]);
  rows.push(['passages cited', String(result.context?.passages?.length ?? 0)]);

  if (result.refusal) rows.push(['abstained because', result.refusal.reason]);

  el.signals.innerHTML = rows
    .map(
      ([key, value]) =>
        `<div class="signal"><span class="signal__key">${escapeHtml(key)}</span>` +
        `<span class="signal__value">${escapeHtml(value)}</span></div>`,
    )
    .join('');
}

// ---------------------------------------------------------------------------
// microphone
// ---------------------------------------------------------------------------
let recorder = null;
let chunks = [];
let stream = null;

async function startRecording() {
  if (busy || recorder) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser will not give a page microphone access.', true);
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    setStatus('Microphone permission denied.', true);
    return;
  }

  // Safari has no webm/opus; let the browser pick when our preference is absent.
  const preferred = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  const mimeType = preferred.find((type) => MediaRecorder.isTypeSupported?.(type));

  chunks = [];
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  });
  recorder.start();

  el.mic.dataset.state = 'recording';
  setStatus('listening — release to send');
}

async function stopRecording() {
  if (!recorder) return;
  const active = recorder;
  recorder = null;

  const blob = await new Promise((resolve) => {
    active.addEventListener('stop', () => resolve(new Blob(chunks, { type: active.mimeType })), {
      once: true,
    });
    active.stop();
  });

  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  el.mic.dataset.state = 'idle';

  // A tap rather than a hold produces a few hundred bytes of silence.
  if (blob.size < 2000) {
    setStatus('That was too short to transcribe — hold the button while you speak.', true);
    return;
  }
  await askWithAudio(blob);
}

// press-and-hold, for both pointer and keyboard
el.mic.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  el.mic.setPointerCapture?.(event.pointerId);
  startRecording();
});
for (const type of ['pointerup', 'pointercancel', 'pointerleave']) {
  el.mic.addEventListener(type, () => {
    if (recorder) stopRecording();
  });
}
el.mic.addEventListener('keydown', (event) => {
  if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
    event.preventDefault();
    startRecording();
  }
});
el.mic.addEventListener('keyup', (event) => {
  if (event.key === ' ' || event.key === 'Enter') stopRecording();
});

// ---------------------------------------------------------------------------
el.form.addEventListener('submit', (event) => {
  event.preventDefault();
  ask(el.question.value);
});

loadMeta();
