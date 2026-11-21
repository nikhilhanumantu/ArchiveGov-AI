/**
 * ArchiveGov AI — Restoration Time Machine (Module 3)
 * Interactive horizontal stage scrubber, before/after split-wipe comparison,
 * pixel-level reconstructed visual overlay, and deep-linking support.
 */

class RestorationTimeMachine {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.docId = options.docId || this.container.dataset.docId;
    this.stages = [];
    this.currentStageIndex = 0;
    this.isDiffOverlayActive = false;

    this.init();
  }

  async init() {
    await this.fetchStages();
    this.renderScrubber();
    this.bindEvents();
    this.handleDeepLink();
  }

  async fetchStages() {
    try {
      const resp = await fetch(`/api/v1/versions/${this.docId}/stages`);
      if (!resp.ok) throw new Error("Could not load restoration stages");
      const data = await resp.json();
      this.stages = data.stages || [];
      if (this.stages.length > 0) {
        this.currentStageIndex = this.stages.length - 1; // Default to final
      }
    } catch (err) {
      console.warn("Time Machine fetch stages error:", err);
    }
  }

  renderScrubber() {
    const scrubberContainer = document.getElementById("time-machine-scrubber");
    if (!scrubberContainer || this.stages.length === 0) return;

    scrubberContainer.innerHTML = `
      <div class="flex items-center justify-between w-full relative px-2 py-3 bg-surface-container-lowest rounded-xl border border-outline-variant/30">
        <!-- Connecting Track -->
        <div class="absolute left-6 right-6 top-1/2 -translate-y-1/2 h-0.5 bg-outline-variant/30 z-0"></div>
        <div id="tm-track-fill" class="absolute left-6 top-1/2 -translate-y-1/2 h-0.5 bg-secondary transition-all z-0" style="width: 100%;"></div>

        <!-- Stage Nodes -->
        ${this.stages.map((st, idx) => `
          <button class="tm-stage-node relative z-10 flex flex-col items-center gap-1.5 group focus:outline-none" data-index="${idx}">
            <div class="w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-semibold transition-all border-2 ${idx === this.currentStageIndex ? 'bg-secondary text-on-secondary border-secondary scale-110 shadow-lg' : 'bg-surface-container-high text-on-surface-variant border-outline-variant/40 hover:border-secondary/60'}">
              ${idx}
            </div>
            <span class="text-[10px] font-telemetry-code whitespace-nowrap transition-colors ${idx === this.currentStageIndex ? 'text-secondary font-medium' : 'text-on-surface-variant group-hover:text-on-surface'}">
              ${st.label || st.stage_name}
            </span>
          </button>
        `).join("")}
      </div>
    `;

    this.updateStageDisplay();
  }

  bindEvents() {
    // Stage Node clicks
    document.addEventListener("click", (e) => {
      const node = e.target.closest(".tm-stage-node");
      if (node) {
        const idx = parseInt(node.dataset.index, 10);
        if (!isNaN(idx)) {
          this.setStage(idx);
        }
      }
    });

    // Split Wipe Slider
    const splitSlider = document.getElementById("tm-split-slider");
    const splitWipe = document.getElementById("tm-split-wipe");
    if (splitSlider && splitWipe) {
      splitSlider.addEventListener("input", (e) => {
        const val = e.target.value;
        splitWipe.style.clipPath = `polygon(${val}% 0, 100% 0, 100% 100%, ${val}% 100%)`;
      });
    }

    // Toggle Reconstructed Diff Mask
    const diffToggle = document.getElementById("tm-toggle-diff");
    if (diffToggle) {
      diffToggle.addEventListener("click", () => this.toggleDiffMask());
    }
  }

  setStage(idx) {
    if (idx < 0 || idx >= this.stages.length) return;
    this.currentStageIndex = idx;
    this.updateStageDisplay();

    // Update query param
    const current = this.stages[idx];
    const url = new URL(window.location);
    url.searchParams.set("stage", current.stage_name);
    window.history.replaceState({}, "", url);
  }

  updateStageDisplay() {
    const current = this.stages[this.currentStageIndex];
    if (!current) return;

    // Update Track Fill
    const trackFill = document.getElementById("tm-track-fill");
    if (trackFill && this.stages.length > 1) {
      const pct = (this.currentStageIndex / (this.stages.length - 1)) * 100;
      trackFill.style.width = `${pct}%`;
    }

    // Update Stage Images
    const stageImg = document.getElementById("tm-stage-image");
    if (stageImg && current.file_url) {
      stageImg.src = current.file_url;
    }

    // Update Stage Meta Info
    const stageNameEl = document.getElementById("tm-current-stage-name");
    if (stageNameEl) {
      stageNameEl.textContent = `Stage ${current.stage_order}: ${current.label || current.stage_name}`;
    }

    const stageModelEl = document.getElementById("tm-stage-model");
    if (stageModelEl) {
      stageModelEl.textContent = current.model_used || "OpenCV Native Pipeline";
    }

    // Update active node classes
    document.querySelectorAll(".tm-stage-node").forEach((btn, idx) => {
      const circle = btn.querySelector("div");
      const label = btn.querySelector("span");
      if (idx === this.currentStageIndex) {
        circle?.classList.add("bg-secondary", "text-on-secondary", "border-secondary", "scale-110", "shadow-lg");
        circle?.classList.remove("bg-surface-container-high", "text-on-surface-variant", "border-outline-variant/40");
        label?.classList.add("text-secondary", "font-medium");
        label?.classList.remove("text-on-surface-variant");
      } else {
        circle?.classList.remove("bg-secondary", "text-on-secondary", "border-secondary", "scale-110", "shadow-lg");
        circle?.classList.add("bg-surface-container-high", "text-on-surface-variant", "border-outline-variant/40");
        label?.classList.remove("text-secondary", "font-medium");
        label?.classList.add("text-on-surface-variant");
      }
    });
  }

  async toggleDiffMask() {
    this.isDiffOverlayActive = !this.isDiffOverlayActive;
    const btn = document.getElementById("tm-toggle-diff");
    const diffOverlay = document.getElementById("tm-diff-overlay");

    if (this.isDiffOverlayActive) {
      btn?.classList.add("bg-tertiary/20", "border-tertiary", "text-tertiary");
      try {
        const resp = await fetch(`/api/v1/versions/${this.docId}/diff-mask?from_stage=original&to_stage=final`);
        if (resp.ok) {
          const d = await resp.json();
          if (diffOverlay && d.mask_url) {
            diffOverlay.src = d.mask_url;
            diffOverlay.classList.remove("hidden");
          }
        }
      } catch (err) {
        console.warn("Could not load diff mask:", err);
      }
    } else {
      btn?.classList.remove("bg-tertiary/20", "border-tertiary", "text-tertiary");
      if (diffOverlay) diffOverlay.classList.add("hidden");
    }
  }

  handleDeepLink() {
    const urlParams = new URLSearchParams(window.location.search);
    const stageName = urlParams.get("stage");
    if (stageName) {
      const foundIdx = this.stages.findIndex(s => s.stage_name === stageName);
      if (foundIdx !== -1) {
        this.setStage(foundIdx);
      }
    }
  }
}

window.RestorationTimeMachine = RestorationTimeMachine;
