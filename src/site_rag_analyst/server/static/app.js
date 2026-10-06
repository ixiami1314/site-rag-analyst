/* site-rag-analyst demo console.
   Vanilla JS, no deps. Polls the run API, renders live stage progress,
   the cited report, and every intermediate artifact. */

"use strict";

const STAGES = ["crawl", "extract", "chunk", "embed", "store", "retrieve", "analyze"];
const $ = (sel) => document.querySelector(sel);

const state = { config: null, runId: null, pollTimer: null, t0: null, result: null };

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */

async function boot() {
  try {
    const res = await fetch("/api/config");
    state.config = await res.json();
    const badge = $("#mode-badge");
    if (state.config.demo_mode) {
      badge.textContent = "demo mode · offline embeddings";
      badge.classList.add("demo");
      badge.title = "No LLM credentials configured: TF-IDF embeddings + extractive analyst. Add LLM_* env vars to go live.";
      $("#foot-note").textContent = "demo mode — analysis is extractive; no LLM is called";
    } else {
      badge.textContent = "live · " + state.config.model;
      badge.classList.add("live");
      $("#foot-note").textContent = "live mode · model: " + state.config.model;
    }
  } catch {
    $("#mode-badge").textContent = "api offline";
  }

  $("#btn-run").addEventListener("click", () => startRun($("#url-input").value));
  $("#btn-demo").addEventListener("click", () => {
    $("#url-input").value = "";
    startRun("");
  });
  $("#url-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startRun(e.target.value);
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.pane));
  });
}

/* ------------------------------------------------------------------ */
/* run lifecycle                                                       */
/* ------------------------------------------------------------------ */

async function startRun(url) {
  $("#run-error").hidden = true;
  $("#btn-run").disabled = true;
  $("#btn-demo").disabled = true;
  $("#pipeline").hidden = false;
  $("#output").hidden = true;
  state.result = null;
  state.t0 = performance.now();
  renderRail([]);

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url.trim() }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || ("HTTP " + res.status));
    }
    const { run_id } = await res.json();
    state.runId = run_id;
    poll(run_id);
  } catch (err) {
    failRun(err.message);
  }
}

function failRun(message) {
  $("#btn-run").disabled = false;
  $("#btn-demo").disabled = false;
  const el = $("#run-error");
  el.textContent = "✗ " + message;
  el.hidden = false;
  if (state.pollTimer) clearInterval(state.pollTimer);
}

function poll(runId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    let payload;
    try {
      const res = await fetch("/api/runs/" + runId);
      if (!res.ok) throw new Error("HTTP " + res.status);
      payload = await res.json();
    } catch {
      return; // transient network hiccup: keep polling
    }
    renderRail(payload.stages || []);
    const elapsed = ((performance.now() - state.t0) / 1000).toFixed(1);
    $("#run-meta").textContent =
      payload.url + "  ·  " + payload.status + "  ·  " + elapsed + "s elapsed";

    if (payload.status === "done" && payload.result) {
      clearInterval(state.pollTimer);
      state.result = payload.result;
      renderResult(payload.result);
    } else if (payload.status === "error" && payload.result) {
      clearInterval(state.pollTimer);
      failRun(payload.result.error || "run failed");
    }
  }, 900);
}

/* ------------------------------------------------------------------ */
/* pipeline rail                                                       */
/* ------------------------------------------------------------------ */

function renderRail(stages) {
  const done = stages.map((s) => s.stage);
  const rail = $("#rail");
  rail.innerHTML = "";
  STAGES.forEach((name) => {
    const node = document.createElement("div");
    node.className = "rail-node";
    const isDone = done.includes(name);
    const isActive = !isDone && done.length === STAGES.indexOf(name);
    if (isDone) node.classList.add("done");
    if (isActive) node.classList.add("active");

    const stat = stages.find((s) => s.stage === name);
    const bulb = document.createElement("div");
    bulb.className = "bulb";
    const label = document.createElement("div");
    label.className = "st-name";
    label.textContent = name;
    const meta = document.createElement("div");
    meta.className = "st-meta";
    meta.textContent = isDone ? stat.duration_seconds.toFixed(2) + "s" : "";

    node.append(bulb, label, meta);
    rail.appendChild(node);
  });
}

/* ------------------------------------------------------------------ */
/* result rendering                                                    */
/* ------------------------------------------------------------------ */

function renderResult(result) {
  $("#btn-run").disabled = false;
  $("#btn-demo").disabled = false;
  $("#output").hidden = false;
  renderReport(result);
  renderPages(result);
  renderChunks(result);
  renderRetrieval(result);
  renderStats(result);
  activateTab("pane-report");
}

function activateTab(paneId) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("is-active", t.dataset.pane === paneId)
  );
  document.querySelectorAll(".pane").forEach((p) =>
    p.classList.toggle("is-active", p.id === paneId)
  );
}

function renderReport(result) {
  const report = result.report;
  const pane = $("#pane-report");
  if (!report) {
    pane.innerHTML = '<p class="error-line">run failed: ' + esc(result.error || "?") + "</p>";
    return;
  }
  const meta = document.createElement("div");
  meta.className = "report-meta";
  meta.innerHTML =
    "<b>site</b> " + esc(result.url) +
    " · <b>mode</b> " + esc(result.mode) +
    (report.model ? " · <b>model</b> " + esc(report.model) : "") +
    " · <b>pages</b> " + report.pages.length +
    " · <b>chunks</b> " + result.chunks.length +
    ' · <a href="/api/runs/' + state.runId + '/report.md" target="_blank">report.md ↗</a>';

  const body = document.createElement("div");
  body.className = "report";
  body.innerHTML = markdownToHtml(buildMarkdown(result));

  pane.innerHTML = "";
  pane.append(meta, body);

  body.querySelectorAll(".cite").forEach((chip) => {
    chip.addEventListener("click", () => jumpToChunk(chip.dataset.chunk));
  });
}

/* Rebuild the report markdown from structured data (keeps the renderer
   subset small and the citation chips precise). */
function buildMarkdown(result) {
  const r = result.report;
  const lines = [];
  lines.push("# " + r.site_url.replace(/^https?:\/\//, ""));
  lines.push("", r.executive_summary, "");
  r.sections.forEach((section) => {
    lines.push("## " + section.title);
    if (section.points.length === 0) {
      lines.push("_No relevant context was retrieved for this section._", "");
    } else {
      section.points.forEach((p) => {
        lines.push("- " + p.text + " _[" + p.citations.join("], [") + "]_");
      });
      lines.push("");
    }
  });
  return lines.join("\n");
}

/* ------------------------------------------------------------------ */
/* intermediates                                                       */
/* ------------------------------------------------------------------ */

function renderPages(result) {
  const pane = $("#pane-pages");
  const rows = result.pages
    .map(
      (p) =>
        "<tr><td class='url-cell'>" + esc(shortUrl(p.url)) + "</td><td>" + esc(p.title) +
        "</td><td class='num'>" + p.word_count + "</td><td class='num'>" +
        chunkCount(result, p.url) + "</td><td class='num'>" +
        (p.summary_line ? esc(p.summary_line).slice(0, 90) : "") + "</td></tr>"
    )
    .join("");
  pane.innerHTML =
    "<p class='report-meta'>" + result.pages.length +
    " pages extracted (nav/footer/boilerplate removed before chunking).</p>" +
    "<div class='table-wrap'><table class='grid'><thead><tr>" +
    "<th>url</th><th>title</th><th class='num'>words</th><th class='num'>chunks</th><th>summary</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table></div>";
}

function chunkCount(result, pageUrl) {
  return result.chunks.filter((c) => c.page_url === pageUrl).length;
}

function renderChunks(result) {
  const pane = $("#pane-chunks");
  const rows = result.chunks
    .map((c) => {
      const preview = esc(c.text).slice(0, 160);
      return (
        "<tr id='chunk-" + c.id + "'><td><span class='chunk-id'>" + c.id + "</span></td>" +
        "<td class='url-cell'>" + esc(shortUrl(c.page_url)) + "</td>" +
        "<td>" + esc(c.heading_path || "—") + "</td>" +
        "<td class='num'>" + c.word_count + "</td>" +
        "<td class='chunk-text'>" + preview + "…</td></tr>"
      );
    })
    .join("");
  pane.innerHTML =
    "<p class='report-meta'>" + result.chunks.length +
    " chunks — structure-aware cuts with heading paths and seam overlap. " +
    "Click a citation chip in the report to locate its chunk.</p>" +
    "<div class='table-wrap'><table class='grid'><thead><tr>" +
    "<th>id</th><th>page</th><th>heading</th><th class='num'>words</th><th>text</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table></div>";
}

function renderRetrieval(result) {
  const pane = $("#pane-retrieval");
  const cards = result.retrieval
    .map((preview) => {
      const hits = preview.results
        .map((hit) => {
          const pct = Math.round(Math.max(0, Math.min(1, hit.score)) * 100);
          return (
            "<div class='hitrow'>" +
            "<div class='score-track'><div class='score-fill' style='width:" + pct + "%'></div></div>" +
            "<div class='score-num'>" + hit.score.toFixed(2) + "</div>" +
            "<div class='hit-text'><b>[" + hit.chunk.id + "]</b> " +
            esc(hit.chunk.heading_path || hit.chunk.page_title || "") + " — " +
            esc(hit.chunk.text).slice(0, 110) + "…</div></div>"
          );
        })
        .join("");
      return "<div class='qcard panel'><p class='q'>" + esc(preview.query) + "</p>" + hits + "</div>";
    })
    .join("");
  pane.innerHTML =
    "<p class='report-meta'>One retrieval query per analysis section — this is exactly " +
    "the context the analyst was grounded on.</p>" + cards;
}

function renderStats(result) {
  const pane = $("#pane-stats");
  const stats = (result.report && result.report.stage_stats) || [];
  const maxDur = Math.max(...stats.map((s) => s.duration_seconds), 0.001);
  const rows = stats
    .map(
      (s) =>
        "<tr><td>" + s.stage + "</td><td class='num'>" + s.items + "</td>" +
        "<td class='num'>" + s.duration_seconds.toFixed(2) + "</td>" +
        "<td><span class='bar-cell' style='width:" +
        Math.round((s.duration_seconds / maxDur) * 180) + "px'></span></td>" +
        "<td class='url-cell'>" + esc(s.detail) + "</td></tr>"
    )
    .join("");
  pane.innerHTML =
    "<div class='table-wrap'><table class='grid'><thead><tr>" +
    "<th>stage</th><th class='num'>items</th><th class='num'>seconds</th><th></th><th>detail</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table></div>";
}

function jumpToChunk(chunkId) {
  activateTab("pane-chunks");
  const row = document.getElementById("chunk-" + chunkId);
  if (row) {
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    row.classList.add("target");
    setTimeout(() => row.classList.remove("target"), 2500);
  }
}

/* ------------------------------------------------------------------ */
/* tiny markdown renderer (report subset)                              */
/* ------------------------------------------------------------------ */

function markdownToHtml(md) {
  const lines = md.split("\n");
  const out = [];
  let inList = false;
  let para = [];

  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + inline(para.join(" ")) + "</p>");
      para = [];
    }
  };
  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushPara();
      closeList();
      continue;
    }
    if (line.startsWith("# ")) {
      flushPara(); closeList(); out.push("<h1>" + inline(line.slice(2)) + "</h1>");
    } else if (line.startsWith("## ")) {
      flushPara(); closeList(); out.push("<h2>" + inline(line.slice(3)) + "</h2>");
    } else if (line.startsWith("### ")) {
      flushPara(); closeList(); out.push("<h3>" + inline(line.slice(4)) + "</h3>");
    } else if (line.startsWith("- ")) {
      flushPara();
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push("<li>" + inline(line.slice(2)) + "</li>");
    } else if (line === "---") {
      flushPara(); closeList(); out.push("<hr>");
    } else {
      para.push(line);
    }
  }
  flushPara();
  closeList();
  return out.join("\n");
}

function inline(text) {
  let s = esc(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_m, code) => {
    codes.push(code);
    return "\u0000" + (codes.length - 1) + "\u0000";
  });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/_([^_]+)_/g, "<em>$1</em>");
  s = s.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>'
  );
  s = s.replace(
    /\[(p\d{2}-c\d+)\]/g,
    '<span class="cite" data-chunk="$1" title="jump to chunk">$1</span>'
  );
  s = s.replace(/\u0000(\d+)\u0000/g, (_m, i) => "<code>" + codes[+i] + "</code>");
  return s;
}

/* ------------------------------------------------------------------ */
/* utils                                                               */
/* ------------------------------------------------------------------ */

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shortUrl(url) {
  return url.replace(/^https?:\/\//, "").replace(/\/$/, "");
}

boot();
