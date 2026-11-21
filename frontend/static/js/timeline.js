/**
 * ArchiveGov AI — Historical Archival Timeline (Module 2)
 * Chronological milestone synthesis, date certainty categorization,
 * and deep-linking into Document Lens & Copilot.
 */

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("timeline-app");
  if (!container) return;

  const streamEl = document.getElementById("timeline-events-stream");
  const counterEl = document.getElementById("timeline-events-counter");
  const searchInput = document.getElementById("timeline-entity-search");
  const btnRefresh = document.getElementById("btn-refresh-timeline");

  let allEvents = [];
  let activeCertainty = "all";
  let activeSearch = "";

  async function fetchTimeline(forceRegenerate = false) {
    if (streamEl) {
      streamEl.innerHTML = `
        <div class="flex items-center justify-center py-16 text-on-surface-variant gap-3">
          <span class="material-symbols-outlined animate-spin text-xl text-primary">progress_activity</span>
          <span class="text-sm font-telemetry-code">Synthesizing chronological milestones across vault...</span>
        </div>
      `;
    }

    try {
      let resp;
      if (forceRegenerate) {
        resp = await fetch("/api/v1/timeline/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ timeline_id: "vault_master" })
        });
      } else {
        resp = await fetch("/api/v1/timeline");
      }

      if (!resp.ok) throw new Error("Timeline API request failed");
      const data = await resp.json();
      allEvents = data.events || [];
      renderTimeline();
    } catch (err) {
      console.error("Timeline error:", err);
      if (streamEl) {
        streamEl.innerHTML = `
          <div class="p-6 rounded-xl bg-error/10 border border-error/30 text-error text-center text-xs">
            Failed to load archival timeline: ${err.message}
          </div>
        `;
      }
    }
  }

  function renderTimeline() {
    if (!streamEl) return;

    let filtered = allEvents.filter(ev => {
      if (activeCertainty !== "all" && ev.certainty !== activeCertainty) return false;
      if (activeSearch) {
        const query = activeSearch.toLowerCase();
        const inTitle = (ev.title || "").toLowerCase().includes(query);
        const inFact = (ev.source_fact || "").toLowerCase().includes(query);
        const inEntities = (ev.entities || []).some(ent => ent.toLowerCase().includes(query));
        const inDept = (ev.department_id || "").toLowerCase().includes(query);
        if (!inTitle && !inFact && !inEntities && !inDept) return false;
      }
      return true;
    });

    if (counterEl) counterEl.textContent = `${filtered.length} Events`;

    if (filtered.length === 0) {
      streamEl.innerHTML = `
        <div class="p-10 rounded-2xl bg-surface-container-low border border-outline-variant/30 text-center text-on-surface-variant text-xs">
          No historical events matched the current certainty or entity filters.
        </div>
      `;
      return;
    }

    streamEl.innerHTML = filtered.map((ev, idx) => {
      const isEven = idx % 2 === 0;
      const certBadge = getCertaintyBadge(ev.certainty);

      return `
        <div class="flex flex-col md:flex-row items-center w-full relative group">
          <!-- Left Content or Spacer -->
          <div class="w-full md:w-1/2 ${isEven ? 'md:pr-10 md:text-right' : 'md:order-2 md:pl-10 md:text-left'} pl-12 md:pl-0">
            <div class="p-4 rounded-xl bg-surface-container-low hover:bg-surface-container border border-outline-variant/30 shadow-lg transition-all group-hover:border-primary/50 text-left">
              <!-- Event Meta -->
              <div class="flex items-center justify-between gap-2 border-b border-outline-variant/20 pb-2 mb-2">
                <div class="flex items-center gap-2">
                  <span class="text-sm font-bold font-telemetry-code text-secondary">${ev.year || 'c. 1900'}</span>
                  ${certBadge}
                </div>
                <span class="text-[10px] font-telemetry-code text-on-surface-variant truncate max-w-[120px]">${ev.archive_reference_id || ''}</span>
              </div>

              <!-- Title & Excerpt -->
              <h3 class="text-xs font-semibold text-on-surface mb-1">${ev.title}</h3>
              <p class="text-[11px] text-on-surface-variant/90 font-mono bg-surface-container-lowest p-2 rounded border border-outline-variant/20 leading-relaxed line-clamp-3 mb-2">
                "${ev.source_fact || ''}"
              </p>

              <!-- Entities -->
              ${ev.entities && ev.entities.length ? `
                <div class="flex items-center gap-1 flex-wrap mb-3">
                  ${ev.entities.map(ent => `
                    <span class="text-[9px] px-1.5 py-0.5 rounded bg-surface-container-high text-on-surface-variant font-telemetry-code">
                      ${ent}
                    </span>
                  `).join("")}
                </div>
              ` : ''}

              <!-- Actions Deep Link -->
              <div class="flex items-center justify-between pt-2 border-t border-outline-variant/20 text-xs">
                <a href="${ev.lens_url || `/lens/${ev.doc_id}`}" class="text-secondary hover:text-secondary-fixed transition-colors flex items-center gap-1 text-[11px] font-medium">
                  <span class="material-symbols-outlined text-xs">center_focus_strong</span>
                  <span>Inspect in Lens</span>
                </a>
                <button onclick="askCopilotAboutTimeline('${encodeURIComponent(ev.title)}', '${ev.doc_id}')" class="text-primary hover:text-primary-container transition-colors flex items-center gap-1 text-[11px] font-medium">
                  <span class="material-symbols-outlined text-xs">chat</span>
                  <span>Ask Copilot</span>
                </button>
              </div>
            </div>
          </div>

          <!-- Center Node Badge -->
          <div class="absolute left-6 md:left-1/2 transform -translate-x-1/2 w-6 h-6 rounded-full bg-surface-container-highest border-2 border-secondary flex items-center justify-center text-secondary shadow-md z-10">
            <span class="w-2 h-2 rounded-full bg-secondary"></span>
          </div>

          <!-- Right Content or Spacer -->
          <div class="hidden md:block w-1/2 ${isEven ? 'order-2 pl-10' : 'pr-10'}"></div>
        </div>
      `;
    }).join("");
  }

  function getCertaintyBadge(certainty) {
    switch (certainty) {
      case "verified":
        return `<span class="text-[9px] font-telemetry-code uppercase tracking-wider px-1.5 py-0.5 rounded bg-secondary/20 text-secondary border border-secondary/30">Verified</span>`;
      case "inferred":
        return `<span class="text-[9px] font-telemetry-code uppercase tracking-wider px-1.5 py-0.5 rounded bg-tertiary/20 text-tertiary border border-tertiary/30">Inferred</span>`;
      case "conflicting":
        return `<span class="text-[9px] font-telemetry-code uppercase tracking-wider px-1.5 py-0.5 rounded bg-error/20 text-error border border-error/30">Conflicting</span>`;
      default:
        return `<span class="text-[9px] font-telemetry-code uppercase tracking-wider px-1.5 py-0.5 rounded bg-surface-container text-on-surface-variant">Uncertain</span>`;
    }
  }

  // Filter Event Listeners
  document.querySelectorAll(".certainty-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".certainty-chip").forEach(c => {
        c.classList.remove("active", "bg-secondary/20", "text-secondary", "border-secondary/40", "font-medium");
        c.classList.add("text-on-surface-variant");
      });
      chip.classList.add("active", "bg-secondary/20", "text-secondary", "border-secondary/40", "font-medium");
      chip.classList.remove("text-on-surface-variant");
      activeCertainty = chip.dataset.certainty;
      renderTimeline();
    });
  });

  searchInput?.addEventListener("input", (e) => {
    activeSearch = e.target.value.trim();
    renderTimeline();
  });

  btnRefresh?.addEventListener("click", () => {
    fetchTimeline(true);
  });

  // Global helper for opening Copilot from timeline cards
  window.askCopilotAboutTimeline = (encodedTitle, docId) => {
    const title = decodeURIComponent(encodedTitle);
    const prompt = `Tell me more about this archival milestone from document ${docId}: "${title}"`;
    if (window.ArchiveGovCopilot && typeof window.ArchiveGovCopilot.openWithPrompt === "function") {
      window.ArchiveGovCopilot.openWithPrompt(prompt, docId);
    } else {
      window.location.href = `/app/archive?query=${encodeURIComponent(prompt)}`;
    }
  };

  fetchTimeline(false);
});
