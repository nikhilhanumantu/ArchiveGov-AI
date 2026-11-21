/**
 * ArchiveGov AI — Document Intelligence Lens (Module 4)
 * Precision SVG/Canvas bounding box overlay, epistemic provenance inspection,
 * AI-powered archival interpretation, and cross-vault lineage discovery.
 */

document.addEventListener("DOMContentLoaded", () => {
  const appContainer = document.getElementById("document-lens-app");
  if (!appContainer) return;

  const docId = appContainer.dataset.docId;
  const docImg = document.getElementById("lens-document-image");
  const svgOverlay = document.getElementById("lens-svg-overlay");
  const stage = document.getElementById("lens-stage");
  const regionCountBadge = document.getElementById("region-count-badge");
  const selectedBadge = document.getElementById("selected-badge");
  const sourceFactEl = document.getElementById("inspector-source-fact");
  const confidenceEl = document.getElementById("inspector-confidence");
  const damageOverlapEl = document.getElementById("inspector-damage-overlap");
  const aiInterpretationEl = document.getElementById("inspector-ai-interpretation");
  const relatedListEl = document.getElementById("inspector-related-list");
  const relatedCountLabel = document.getElementById("related-count-label");
  const btnExplain = document.getElementById("btn-explain-selection");
  const btnAskCopilot = document.getElementById("btn-ask-copilot-region");

  let allRegions = [];
  let activeGranularity = "all";
  let selectedRegion = null;
  let zoomLevel = 1.0;

  // Zoom controls
  document.getElementById("zoom-in")?.addEventListener("click", () => {
    zoomLevel = Math.min(3.0, zoomLevel + 0.25);
    applyZoom();
  });
  document.getElementById("zoom-out")?.addEventListener("click", () => {
    zoomLevel = Math.max(0.5, zoomLevel - 0.25);
    applyZoom();
  });
  document.getElementById("zoom-reset")?.addEventListener("click", () => {
    zoomLevel = 1.0;
    applyZoom();
  });

  function applyZoom() {
    if (stage) stage.style.transform = `scale(${zoomLevel})`;
  }

  // Granularity filter tabs
  document.querySelectorAll(".granularity-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".granularity-btn").forEach(b => {
        b.classList.remove("active", "bg-primary/20", "text-primary", "font-medium");
        b.classList.add("text-on-surface-variant");
      });
      btn.classList.add("active", "bg-primary/20", "text-primary", "font-medium");
      btn.classList.remove("text-on-surface-variant");
      activeGranularity = btn.dataset.granularity;
      renderBoundingBoxes();
    });
  });

  // Load Regions from API
  async function loadRegions() {
    try {
      const resp = await fetch(`/api/v1/lens/${docId}/regions`);
      if (!resp.ok) throw new Error("Failed to load regions");
      const data = await resp.json();
      allRegions = data.regions || [];
      if (regionCountBadge) regionCountBadge.textContent = `${allRegions.length} Regions`;
      renderBoundingBoxes();

      // Check URL parameters for deep-linked region
      const urlParams = new URLSearchParams(window.location.search);
      const linkedRegionId = urlParams.get("region");
      if (linkedRegionId) {
        const found = allRegions.find(r => r.region_id === linkedRegionId);
        if (found) selectRegion(found);
      }
    } catch (err) {
      console.warn("Could not load document lens regions:", err);
      if (sourceFactEl) sourceFactEl.textContent = "No OCR regions detected. Re-run pipeline to extract regions.";
    }
  }

  // Load Related Documents
  async function loadRelated() {
    try {
      const resp = await fetch(`/api/v1/lens/${docId}/related`);
      if (!resp.ok) return;
      const data = await resp.json();
      const related = data.related_records || [];
      if (relatedCountLabel) relatedCountLabel.textContent = `${related.length} Linked`;
      if (!relatedListEl) return;

      if (related.length === 0) {
        relatedListEl.innerHTML = `<div class="text-[11px] text-on-surface-variant py-2">No cross-vault related folios found.</div>`;
        return;
      }

      relatedListEl.innerHTML = related.map(r => `
        <a href="/lens/${r.document_id}" class="p-2 rounded-lg bg-surface-container-lowest hover:bg-surface-container border border-outline-variant/20 flex flex-col gap-1 transition-all group">
          <div class="flex items-center justify-between">
            <span class="text-xs font-medium text-on-surface group-hover:text-secondary truncate">${r.title}</span>
            <span class="text-[10px] font-telemetry-code text-on-surface-variant shrink-0">${r.original_date || ''}</span>
          </div>
          <div class="text-[10px] text-on-surface-variant/80 flex items-center gap-1">
            <span class="material-symbols-outlined text-[11px] text-primary">link</span>
            <span class="truncate">${r.similarity_reason}</span>
          </div>
        </a>
      `).join("");
    } catch (err) {
      console.warn("Related records load error:", err);
    }
  }

  // Render SVG Bounding Boxes scaled to rendered image
  function renderBoundingBoxes() {
    if (!svgOverlay || !docImg) return;
    svgOverlay.innerHTML = "";

    const imgW = docImg.clientWidth || docImg.naturalWidth;
    const imgH = docImg.clientHeight || docImg.naturalHeight;
    if (!imgW || !imgH) return;

    const filtered = allRegions.filter(r => {
      if (activeGranularity === "all") return true;
      return r.granularity === activeGranularity;
    });

    filtered.forEach(region => {
      const norm = region.bbox_normalized || [0, 0, 0, 0];
      // [xmin, ymin, width, height] normalized
      const rx = norm[0] * imgW;
      const ry = norm[1] * imgH;
      const rw = norm[2] * imgW;
      const rh = norm[3] * imgH;

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", rx);
      rect.setAttribute("y", ry);
      rect.setAttribute("width", rw);
      rect.setAttribute("height", rh);
      rect.setAttribute("rx", "2");

      // Color code by confidence and damage
      const conf = region.confidence || 0.95;
      const dmg = region.damage_overlap || 0;

      let strokeColor = "#5de6ff"; // cyan
      let fillColor = "rgba(93, 230, 255, 0.08)";

      if (dmg > 0.05) {
        strokeColor = "#ffb4ab"; // red/error
        fillColor = "rgba(255, 180, 171, 0.15)";
      } else if (conf < 0.85) {
        strokeColor = "#efbf66"; // amber
        fillColor = "rgba(239, 191, 102, 0.10)";
      }

      const isSelected = selectedRegion && selectedRegion.region_id === region.region_id;
      if (isSelected) {
        strokeColor = "#d0bcff"; // primary purple
        fillColor = "rgba(208, 188, 255, 0.28)";
      }

      rect.setAttribute("stroke", strokeColor);
      rect.setAttribute("stroke-width", isSelected ? "2.5" : "1.2");
      rect.setAttribute("fill", fillColor);
      rect.setAttribute("class", "cursor-pointer transition-all hover:opacity-100");
      rect.style.cursor = "pointer";

      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        selectRegion(region);
      });

      svgOverlay.appendChild(rect);
    });
  }

  // Select Region and Populate Inspector Panel
  function selectRegion(region) {
    selectedRegion = region;
    renderBoundingBoxes();

    if (selectedBadge) {
      selectedBadge.textContent = `${region.granularity || 'region'} · ${region.region_id}`;
      selectedBadge.className = "text-[10px] font-telemetry-code uppercase tracking-wider px-2 py-0.5 rounded bg-primary/20 text-primary border border-primary/30";
    }

    if (sourceFactEl) sourceFactEl.textContent = region.text || "(Empty text)";
    if (confidenceEl) {
      const pct = Math.round((region.confidence || 0.95) * 100);
      confidenceEl.textContent = `${pct}%`;
      confidenceEl.className = pct >= 90 ? "font-telemetry-code text-sm font-semibold text-secondary mt-0.5" : "font-telemetry-code text-sm font-semibold text-tertiary mt-0.5";
    }
    if (damageOverlapEl) {
      const dmg = region.damage_overlap || 0;
      damageOverlapEl.textContent = dmg > 0 ? `${Math.round(dmg * 100)}% Altered` : "0% (Authentic)";
      damageOverlapEl.className = dmg > 0 ? "font-telemetry-code text-sm font-semibold text-error mt-0.5" : "font-telemetry-code text-sm font-semibold text-secondary mt-0.5";
    }

    if (aiInterpretationEl) {
      aiInterpretationEl.innerHTML = `<span class="text-on-surface-variant">Click <b>Explain</b> to generate sovereign contextual interpretation for this selected text snippet.</span>`;
    }
  }

  // Explain Button Action
  btnExplain?.addEventListener("click", async () => {
    const textToExplain = selectedRegion ? selectedRegion.text : sourceFactEl.textContent;
    if (!textToExplain || textToExplain.includes("Click any word")) return;

    aiInterpretationEl.innerHTML = `
      <div class="flex items-center gap-2 text-primary py-2">
        <span class="material-symbols-outlined animate-spin text-sm">progress_activity</span>
        <span class="text-xs">Synthesizing archival context...</span>
      </div>
    `;

    try {
      const resp = await fetch(`/api/v1/lens/${docId}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          region_id: selectedRegion ? selectedRegion.region_id : null,
          text: textToExplain
        })
      });
      if (!resp.ok) throw new Error("Explanation API error");
      const resData = await resp.json();
      aiInterpretationEl.innerHTML = `<div class="prose prose-invert text-xs leading-relaxed">${formatArchivalMarkdown(resData.ai_interpretation)}</div>`;
    } catch (err) {
      aiInterpretationEl.innerHTML = `<span class="text-error text-xs">Error generating explanation: ${err.message}</span>`;
    }
  });

  // Ask Copilot About Region Action
  btnAskCopilot?.addEventListener("click", () => {
    const textSnippet = selectedRegion ? selectedRegion.text : "";
    const prompt = textSnippet 
      ? `Explain the historical significance of this excerpt from record ${docId}: "${textSnippet}"`
      : `Provide an archival summary of record ${docId}`;

    if (window.ArchiveGovCopilot && typeof window.ArchiveGovCopilot.openWithPrompt === "function") {
      window.ArchiveGovCopilot.openWithPrompt(prompt, docId);
    } else {
      // Fallback redirect to digital archive or query
      window.location.href = `/app/archive?query=${encodeURIComponent(prompt)}`;
    }
  });

  // Reconstructed visual overlay toggle
  let isOverlayActive = false;
  document.getElementById("btn-reconstructed-overlay")?.addEventListener("click", async () => {
    isOverlayActive = !isOverlayActive;
    const btn = document.getElementById("btn-reconstructed-overlay");
    if (isOverlayActive) {
      btn?.classList.add("bg-tertiary/20", "border-tertiary", "text-tertiary");
      try {
        const resp = await fetch(`/api/v1/versions/${docId}/diff-mask?from_stage=original&to_stage=final`);
        if (resp.ok) {
          const d = await resp.json();
          if (d.mask_url && docImg) docImg.src = d.mask_url;
        }
      } catch (e) {
        console.warn("Could not fetch diff mask:", e);
      }
    } else {
      btn?.classList.remove("bg-tertiary/20", "border-tertiary", "text-tertiary");
      if (docImg) docImg.src = `/api/v1/files/${docImg.dataset.defaultKey || ''}`;
      window.location.reload();
    }
  });

  function formatArchivalMarkdown(text) {
    if (!text) return "";
    return text
      .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
      .replace(/\*(.*?)\*/g, '<i>$1</i>')
      .replace(/\n\n/g, '<br/><br/>')
      .replace(/\n/g, '<br/>');
  }

  // Initialize once image has loaded
  if (docImg.complete) {
    loadRegions();
  } else {
    docImg.addEventListener("load", loadRegions);
  }
  loadRelated();
  window.addEventListener("resize", renderBoundingBoxes);
});
