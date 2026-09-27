"use strict";

const SCORE_OPTIONS = [
  { score: 0, title: "Easy negative", detail: "明显无关" },
  { score: 1, title: "Hard negative", detail: "表面相关但不满足" },
  { score: 2, title: "Positive", detail: "有效但不够直接或完整" },
  { score: 3, title: "Strong positive", detail: "直接且充分满足" },
];

const DATASET_LABELS = {
  msmarco: "MSMARCO",
  text2sql: "Text2SQL",
  "theoremqa-theorems": "TheoremQA · Theorems",
};

const state = {
  annotatorId: "",
  datasetOrder: [],
  samples: [],
  scores: {},
  guidelines: {},
  index: 0,
  saving: false,
};

const elements = {};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  collectElements();
  buildScoreButtons();
  bindControls();
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (!response.ok) throw new Error(`无法载入 session（HTTP ${response.status}）`);
    const session = await response.json();
    state.annotatorId = session.annotator_id;
    state.datasetOrder = session.dataset_order;
    state.samples = session.samples;
    state.scores = session.scores;
    state.index = firstUnansweredIndex();
    elements.annotatorId.textContent = state.annotatorId;
    buildDatasetNavigation();
    await showCurrentSample();
  } catch (error) {
    showError(error.message || String(error));
  }
}

function collectElements() {
  const ids = [
    "annotator-id", "save-status", "overall-progress", "overall-progress-bar",
    "dataset-nav", "error-banner", "completion-banner", "guideline-title",
    "guideline-text", "dataset-name", "sample-position", "sample-id",
    "query-text", "query-translation", "document-text", "document-translation",
    "score-buttons", "previous-button", "unanswered-button", "next-button",
  ];
  for (const id of ids) elements[toCamelCase(id)] = document.getElementById(id);
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function buildScoreButtons() {
  for (const option of SCORE_OPTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `score-button score-${option.score}`;
    button.dataset.score = String(option.score);
    button.setAttribute("aria-label", `${option.score} ${option.title}: ${option.detail}`);

    const number = document.createElement("span");
    number.className = "score-number";
    number.textContent = String(option.score);
    const copy = document.createElement("span");
    copy.className = "score-copy";
    const title = document.createElement("strong");
    title.textContent = option.title;
    const detail = document.createElement("small");
    detail.textContent = option.detail;
    copy.append(title, detail);
    button.append(number, copy);
    button.addEventListener("click", () => saveScore(option.score));
    elements.scoreButtons.append(button);
  }
}

function bindControls() {
  elements.previousButton.addEventListener("click", () => navigateBy(-1));
  elements.nextButton.addEventListener("click", () => navigateBy(1));
  elements.unansweredButton.addEventListener("click", navigateToNextUnanswered);
  document.addEventListener("keydown", (event) => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    if (["0", "1", "2", "3"].includes(event.key)) {
      event.preventDefault();
      saveScore(Number(event.key));
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigateBy(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      navigateBy(1);
    } else if (event.key.toLowerCase() === "u") {
      event.preventDefault();
      navigateToNextUnanswered();
    }
  });
}

function buildDatasetNavigation() {
  elements.datasetNav.replaceChildren();
  for (const dataset of state.datasetOrder) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.dataset = dataset;
    button.addEventListener("click", () => {
      const index = state.samples.findIndex((sample) => sample.dataset === dataset);
      if (index >= 0) goTo(index);
    });
    elements.datasetNav.append(button);
  }
  updateProgress();
}

async function showCurrentSample() {
  if (!state.samples.length) return;
  clearError();
  const sample = state.samples[state.index];
  const datasetSamples = state.samples.filter((item) => item.dataset === sample.dataset);
  const position = datasetSamples.findIndex((item) => item.sample_id === sample.sample_id) + 1;

  elements.datasetName.textContent = DATASET_LABELS[sample.dataset] || sample.dataset;
  elements.samplePosition.textContent = `${position} / ${datasetSamples.length}`;
  elements.sampleId.textContent = sample.sample_id;
  elements.queryText.textContent = sample.query;
  elements.queryTranslation.textContent = sample.query_zh;
  elements.documentText.textContent = sample.document;
  elements.documentTranslation.textContent = sample.document_zh;
  elements.documentText.classList.toggle("sql-document", sample.dataset === "text2sql");
  elements.previousButton.disabled = state.index === 0 || state.saving;
  elements.nextButton.disabled = state.index === state.samples.length - 1 || state.saving;
  updateSelectedScore(sample.sample_id);
  updateProgress();
  await showGuideline(sample.dataset);
  renderMathematics(sample.dataset);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function showGuideline(dataset) {
  elements.guidelineTitle.textContent = `${DATASET_LABELS[dataset] || dataset} instruction`;
  if (!state.guidelines[dataset]) {
    elements.guidelineText.textContent = "正在载入 instruction…";
    const response = await fetch(`/guidelines/${dataset}.md`, { cache: "no-store" });
    if (!response.ok) throw new Error(`无法载入 ${dataset} instruction`);
    state.guidelines[dataset] = await response.text();
  }
  elements.guidelineText.textContent = state.guidelines[dataset];
}

function renderMathematics(dataset) {
  if (dataset !== "theoremqa-theorems" || typeof window.renderMathInElement !== "function") return;
  const options = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  };
  window.renderMathInElement(elements.queryText, options);
  window.renderMathInElement(elements.queryTranslation, options);
  window.renderMathInElement(elements.documentText, options);
  window.renderMathInElement(elements.documentTranslation, options);
}

async function saveScore(score) {
  if (state.saving || !state.samples.length) return;
  const sample = state.samples[state.index];
  state.saving = true;
  setScoreButtonsDisabled(true);
  setSaveStatus("正在保存…", "saving");
  clearError();
  try {
    const response = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sample.sample_id, score }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `保存失败（HTTP ${response.status}）`);
    state.scores[sample.sample_id] = payload.saved_score;
    updateSelectedScore(sample.sample_id);
    updateProgress();
    setSaveStatus("已保存到本地 CSV", "saved");
    if (state.index < state.samples.length - 1) {
      state.index += 1;
      await showCurrentSample();
    }
  } catch (error) {
    setSaveStatus("保存失败", "error");
    showError(`${error.message || error}。当前页面未前进，请重试。`);
  } finally {
    state.saving = false;
    setScoreButtonsDisabled(false);
    elements.previousButton.disabled = state.index === 0;
    elements.nextButton.disabled = state.index === state.samples.length - 1;
  }
}

function updateSelectedScore(sampleId) {
  const selected = state.scores[sampleId];
  for (const button of elements.scoreButtons.querySelectorAll("button")) {
    const isSelected = Number(button.dataset.score) === selected;
    button.classList.toggle("selected", isSelected);
    button.setAttribute("aria-pressed", String(isSelected));
  }
}

function setScoreButtonsDisabled(disabled) {
  for (const button of elements.scoreButtons.querySelectorAll("button")) button.disabled = disabled;
}

function updateProgress() {
  if (!state.samples.length) return;
  const completed = Object.keys(state.scores).length;
  const total = state.samples.length;
  elements.overallProgress.textContent = `${completed} / ${total}`;
  elements.overallProgressBar.style.width = `${(completed / total) * 100}%`;
  elements.completionBanner.hidden = completed !== total;

  for (const button of elements.datasetNav.querySelectorAll("button")) {
    const dataset = button.dataset.dataset;
    const block = state.samples.filter((sample) => sample.dataset === dataset);
    const done = block.filter((sample) => sample.sample_id in state.scores).length;
    button.replaceChildren();
    const label = document.createElement("strong");
    label.textContent = DATASET_LABELS[dataset] || dataset;
    const count = document.createElement("span");
    count.textContent = `${done}/${block.length}`;
    button.append(label, count);
    const current = state.samples[state.index];
    button.classList.toggle("active", Boolean(current && current.dataset === dataset));
    button.classList.toggle("complete", done === block.length);
  }
}

function firstUnansweredIndex() {
  const index = state.samples.findIndex((sample) => !(sample.sample_id in state.scores));
  return index >= 0 ? index : 0;
}

function navigateToNextUnanswered() {
  if (state.saving) return;
  const later = state.samples.findIndex(
    (sample, index) => index > state.index && !(sample.sample_id in state.scores),
  );
  if (later >= 0) return goTo(later);
  const any = state.samples.findIndex((sample) => !(sample.sample_id in state.scores));
  if (any >= 0) return goTo(any);
  elements.completionBanner.hidden = false;
}

function navigateBy(delta) {
  if (state.saving) return;
  goTo(Math.max(0, Math.min(state.samples.length - 1, state.index + delta)));
}

function goTo(index) {
  if (state.saving || index === state.index) return;
  state.index = index;
  showCurrentSample().catch((error) => showError(error.message || String(error)));
}

function setSaveStatus(message, mode) {
  elements.saveStatus.textContent = message;
  elements.saveStatus.dataset.mode = mode;
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}
