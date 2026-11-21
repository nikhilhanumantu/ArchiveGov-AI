/**
 * ArchiveGov AI v3.0 — "Ask the Archive" Sovereign RAG Copilot Drawer & Voice UI
 * Vanilla ES6+ implementation with zero external frameworks.
 */

const ArchiveCopilot = {
  drawer: null,
  backdrop: null,
  messagesContainer: null,
  inputField: null,
  sendBtn: null,
  voiceBtn: null,
  scopeToggleBtn: null,
  isListening: false,
  recognition: null,
  currentScope: 'all', // 'all' or 'current'
  currentDepartment: 'all',
  activeDocumentId: null,
  chatHistory: [],
  isSpeaking: false,
  activeSpeechUtterance: null,

  init() {
    // Only mount on authenticated pages with header
    const path = window.location.pathname.toLowerCase();
    const isPublicPage = (
      path === '/' ||
      path === '/landing' ||
      path === '/login' ||
      path === '/forgot-password' ||
      path.includes('login.html') ||
      path.includes('landing.html')
    );

    if (isPublicPage || !document.querySelector('header')) {
      return;
    }

    this.detectActiveDocumentId();
    this.createDrawerDOM();
    this.bindEvents();
    this.initSpeechRecognition();
    this.loadSuggestions();
  },

  detectActiveDocumentId() {
    const urlParams = new URLSearchParams(window.location.search);
    const queryDocId = urlParams.get('document_id');
    if (queryDocId) {
      this.activeDocumentId = queryDocId;
      this.currentScope = 'current';
      return;
    }

    const path = window.location.pathname;
    const match = path.match(/\/(?:archive|history|ocr|restoration|lens)\/(DOC-[A-Za-z0-9\-]+)/i);
    if (match && match[1]) {
      this.activeDocumentId = match[1];
      this.currentScope = 'current';
    }
  },

  createDrawerDOM() {
    // Check if server-rendered template is already present
    this.drawer = document.getElementById('copilot-drawer');
    this.backdrop = document.getElementById('copilot-backdrop');

    if (this.drawer) {
      this.messagesContainer = document.getElementById('copilot-messages');
      this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
      this.sendBtn = document.getElementById('copilot-send-btn');
      this.voiceBtn = document.getElementById('copilot-voice-btn');
      return;
    }

    // Fallback client injection if template was omitted
    const drawerHtml = `
      <div id="copilot-drawer-container" class="relative z-[60]">
        <button id="copilot-floating-toggle" 
                type="button"
                class="fixed bottom-6 right-6 z-[60] px-4 py-2.5 rounded-full bg-surface-container-high border border-primary/40 text-primary hover:bg-primary/20 shadow-2xl flex items-center gap-2 text-xs font-semibold tracking-wide backdrop-blur-md transition-all hover:scale-105 group"
                title="Open Sovereign Archival Copilot (Ctrl+K)">
          <span class="material-symbols-outlined text-base group-hover:rotate-12 transition-transform text-secondary">psychology</span>
          <span>Ask the Archive</span>
          <span class="w-2 h-2 rounded-full bg-secondary animate-pulse"></span>
        </button>

        <div id="copilot-backdrop" class="fixed inset-0 bg-black/60 backdrop-blur-sm hidden transition-opacity z-[65]"></div>

        <aside 
          id="copilot-drawer" 
          class="fixed right-0 top-0 bottom-0 w-full max-w-lg bg-surface-container-lowest border-l border-outline-variant/30 shadow-2xl flex flex-col z-[70] transform translate-x-full transition-transform duration-300 ease-in-out"
        >
          <div class="h-16 px-5 border-b border-outline-variant/30 flex items-center justify-between bg-surface-container-low shrink-0">
            <div class="flex items-center gap-2.5">
              <div class="w-8 h-8 rounded-lg bg-primary/20 border border-primary/40 flex items-center justify-center text-primary shadow-[0_0_12px_rgba(160,120,255,0.3)]">
                <span class="material-symbols-outlined text-lg">auto_awesome</span>
              </div>
              <div>
                <h2 class="font-headline-sm text-sm font-semibold text-on-surface flex items-center gap-2">
                  <span>Ask the Archive</span>
                  <span class="text-[10px] font-telemetry-code uppercase tracking-wider px-1.5 py-0.5 rounded bg-secondary/15 text-secondary border border-secondary/30">RAG v3.0</span>
                </h2>
                <p class="text-[11px] text-on-surface-variant font-mono">Grounded Sovereign Archival Intelligence</p>
              </div>
            </div>

            <div class="flex items-center gap-1">
              <button id="copilot-clear-btn" type="button" class="p-1.5 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors" title="Clear Chat History">
                <span class="material-symbols-outlined text-lg">restart_alt</span>
              </button>
              <button id="copilot-close-btn" type="button" class="p-1.5 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors" title="Close Drawer (Esc)">
                <span class="material-symbols-outlined text-lg">close</span>
              </button>
            </div>
          </div>

          <div class="p-3 bg-surface-container-lowest border-b border-outline-variant/20 flex flex-col gap-2 shrink-0">
            <div class="flex items-center justify-between text-xs">
              <div class="flex items-center bg-surface-container-low p-0.5 rounded-lg border border-outline-variant/30 text-[11px]">
                <button id="copilot-scope-all" type="button" class="px-2.5 py-1 rounded-md text-primary font-medium bg-primary/20 transition-all">Entire Vault</button>
                <button id="copilot-scope-current" type="button" class="px-2.5 py-1 rounded-md text-on-surface-variant hover:text-on-surface transition-all">This Document</button>
              </div>

              <select id="copilot-dept-filter" class="bg-surface-container-low border border-outline-variant/30 text-[11px] text-on-surface rounded-lg px-2 py-1 focus:outline-none focus:border-secondary">
                <option value="all">All Departments</option>
                <option value="Land Revenue">Land Revenue</option>
                <option value="Public Health">Public Health</option>
                <option value="Judicial Records">Judicial Records</option>
              </select>
            </div>
          </div>

          <div id="copilot-messages" class="flex-1 overflow-y-auto p-4 flex flex-col gap-4 text-xs custom-scrollbar">
            <div class="flex gap-3">
              <div class="w-7 h-7 rounded-full bg-secondary/20 border border-secondary/40 text-secondary flex items-center justify-center shrink-0 mt-0.5">
                <span class="material-symbols-outlined text-sm">robot_2</span>
              </div>
              <div class="flex flex-col gap-1.5 max-w-[85%]">
                <div class="p-3.5 rounded-2xl rounded-tl-sm bg-surface-container-low border border-outline-variant/30 text-on-surface leading-relaxed">
                  Welcome to <b>ArchiveGov AI Sovereign Copilot</b>. Query historical records, land settlements, colonial fiscal assessments (rupees/annas/pies), or public health epidemics with verifiable citations.
                </div>
                <span class="text-[10px] font-telemetry-code text-on-surface-variant px-1 flex items-center gap-1">
                  <span class="w-1.5 h-1.5 rounded-full bg-secondary"></span> Grounded on Certified Vault Corpus
                </span>
              </div>
            </div>
          </div>

          <div id="copilot-suggestions-bar" class="px-4 py-2 bg-surface-container-lowest border-t border-outline-variant/20 flex flex-col gap-1.5 shrink-0">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-telemetry-code text-secondary uppercase tracking-wider font-semibold">Suggested Questions:</span>
            </div>
            <div id="copilot-suggestions-container" class="grid grid-cols-2 gap-2 pb-1"></div>
          </div>

          <div class="p-3 bg-surface-container-low border-t border-outline-variant/30 shrink-0">
            <form id="copilot-form" class="relative flex items-center gap-2" onsubmit="event.preventDefault(); ArchiveCopilot.handleSend();">
              <button 
                id="copilot-voice-btn" 
                type="button" 
                class="w-10 h-10 rounded-xl bg-surface-container hover:bg-surface-container-high border border-outline-variant/30 text-outline hover:text-secondary flex items-center justify-center shrink-0 transition-all shadow-inner"
                title="Voice Input (Speech-to-Text)"
              >
                <span class="material-symbols-outlined text-[20px]" id="copilot-voice-icon">mic</span>
              </button>

              <div class="relative flex-1">
                <input 
                  type="text" 
                  id="copilot-query-input" 
                  placeholder="Ask about deeds, acres, assessments, or [DOC-ID]..." 
                  class="w-full bg-surface-container/90 text-on-surface placeholder:text-outline text-xs pl-3 pr-9 py-2.5 rounded-xl border border-outline-variant/30 focus:border-secondary focus:ring-1 focus:ring-secondary/40 focus:outline-none transition-all shadow-inner"
                  autocomplete="off" 
                />
                <button 
                  type="submit" 
                  id="copilot-send-btn" 
                  class="absolute right-1.5 top-1/2 -translate-y-1/2 w-7 h-7 rounded-lg bg-primary-container text-on-primary-container hover:brightness-110 flex items-center justify-center transition-all shadow-sm disabled:opacity-40"
                  title="Send Question"
                >
                  <span class="material-symbols-outlined text-[16px]">arrow_upward</span>
                </button>
              </div>
            </form>
            <div class="flex items-center justify-between text-[10px] font-mono text-outline px-1 mt-1.5">
              <span>Press <kbd class="px-1 py-0.5 rounded bg-surface-container text-white">Enter</kbd> to submit</span>
              <span class="flex items-center gap-1 text-secondary">
                <span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>
                Offline Resilient
              </span>
            </div>
          </div>
        </aside>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', drawerHtml);
    this.drawer = document.getElementById('copilot-drawer');
    this.backdrop = document.getElementById('copilot-backdrop');
    this.messagesContainer = document.getElementById('copilot-messages');
    this.inputField = document.getElementById('copilot-query-input');
    this.sendBtn = document.getElementById('copilot-send-btn');
    this.voiceBtn = document.getElementById('copilot-voice-btn');
  },

  bindEvents() {
    // Header Copilot button
    const headerBtn = document.getElementById('headerCopilotBtn');
    if (headerBtn) {
      headerBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.toggleDrawer();
      });
    }

    // Floating toggle button
    const floatBtn = document.getElementById('copilot-floating-toggle');
    if (floatBtn) {
      floatBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.toggleDrawer();
      });
    }

    // Drawer backdrop click
    const backdrop = document.getElementById('copilot-backdrop');
    if (backdrop) {
      backdrop.addEventListener('click', () => this.closeDrawer());
    }

    // Delegated data-open-copilot triggers
    document.addEventListener('click', (e) => {
      const trigger = e.target.closest('[data-open-copilot="true"]');
      if (trigger) {
        e.preventDefault();
        this.openDrawer();
      }
    });

    // Close buttons (support both standard and alias)
    const closeBtns = document.querySelectorAll('#copilot-close-btn, #copilot-drawer-close');
    closeBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        this.closeDrawer();
      });
    });

    // Clear history button
    const clearBtn = document.getElementById('copilot-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.chatHistory = [];
        if (this.messagesContainer) {
          this.messagesContainer.innerHTML = '';
          this.appendWelcomeMessage();
        }
        this.loadSuggestions();
        if (typeof Toast !== 'undefined') {
          Toast.info('Copilot conversation reset.', 'Memory Cleared');
        }
      });
    }

    // Keyboard shortcut (Ctrl+K or Cmd+K)
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        this.toggleDrawer();
      } else if (e.key === 'Escape' && this.isOpen()) {
        this.closeDrawer();
      }
    });

    // Scope toggle buttons
    const scopeAll = document.getElementById('copilot-scope-all') || document.getElementById('copilot-scope-vault');
    const scopeCurrent = document.getElementById('copilot-scope-current') || document.getElementById('copilot-scope-document');

    if (scopeAll) {
      scopeAll.addEventListener('click', (e) => {
        e.preventDefault();
        this.currentScope = 'all';
        scopeAll.className = 'px-2.5 py-1 rounded-md text-primary font-medium bg-primary/20 transition-all';
        if (scopeCurrent) {
          scopeCurrent.className = 'px-2.5 py-1 rounded-md text-on-surface-variant hover:text-on-surface transition-all';
        }
        this.loadSuggestions();
      });
    }

    if (scopeCurrent) {
      scopeCurrent.addEventListener('click', (e) => {
        e.preventDefault();
        if (!this.activeDocumentId) {
          if (typeof Toast !== 'undefined') {
            Toast.warning('Open a specific archival record to scope Copilot to a single folio.', 'Scope Notice');
          }
          return;
        }
        this.currentScope = 'current';
        scopeCurrent.className = 'px-2.5 py-1 rounded-md text-primary font-medium bg-primary/20 transition-all';
        if (scopeAll) {
          scopeAll.className = 'px-2.5 py-1 rounded-md text-on-surface-variant hover:text-on-surface transition-all';
        }
        this.loadSuggestions();
      });
    }

    // Department filter dropdown
    const deptFilter = document.getElementById('copilot-dept-filter');
    if (deptFilter) {
      deptFilter.addEventListener('change', () => {
        this.currentDepartment = deptFilter.value || 'all';
      });
    }

    // Delegated click handler for suggested questions (100% reliable)
    document.addEventListener('click', (e) => {
      const card = e.target.closest('[data-copilot-query]');
      if (card) {
        e.preventDefault();
        const query = card.getAttribute('data-copilot-query');
        if (query) {
          this.askSuggestion(query);
        }
      }
    });

    // Form submission handlers
    const forms = document.querySelectorAll('#copilot-form, #copilot-query-form');
    forms.forEach(f => {
      f.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleSend();
      });
    });

    // Input Enter key listener
    if (!this.inputField) {
      this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
    }
    if (this.inputField) {
      this.inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.handleSend();
        }
      });
    }

    // Voice button
    if (!this.voiceBtn) {
      this.voiceBtn = document.getElementById('copilot-voice-btn');
    }
    if (this.voiceBtn) {
      this.voiceBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.toggleVoiceInput();
      });
    }
  },

  isOpen() {
    if (!this.drawer) return false;
    return !this.drawer.classList.contains('translate-x-full');
  },

  openDrawer() {
    if (!this.drawer) {
      this.drawer = document.getElementById('copilot-drawer');
    }
    if (!this.drawer) return;

    this.drawer.classList.remove('hidden');
    requestAnimationFrame(() => {
      this.drawer.classList.remove('translate-x-full');
      this.drawer.classList.add('translate-x-0');
    });

    const backdrop = document.getElementById('copilot-backdrop');
    if (backdrop) {
      backdrop.classList.remove('hidden');
    }

    setTimeout(() => {
      if (!this.inputField) {
        this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
      }
      if (this.inputField) this.inputField.focus();
    }, 150);
  },

  closeDrawer() {
    if (!this.drawer) {
      this.drawer = document.getElementById('copilot-drawer');
    }
    if (!this.drawer) return;

    this.drawer.classList.add('translate-x-full');
    this.drawer.classList.remove('translate-x-0');

    const backdrop = document.getElementById('copilot-backdrop');
    if (backdrop) {
      backdrop.classList.add('hidden');
    }

    this.stopSpeaking();
  },

  toggleDrawer() {
    if (this.isOpen()) {
      this.closeDrawer();
    } else {
      this.openDrawer();
    }
  },

  appendWelcomeMessage() {
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }
    if (!this.messagesContainer) return;

    const welcomeHtml = `
      <div class="flex gap-3 animate-fadeIn">
        <div class="w-7 h-7 rounded-full bg-secondary/20 border border-secondary/40 text-secondary flex items-center justify-center shrink-0 mt-0.5">
          <span class="material-symbols-outlined text-sm">robot_2</span>
        </div>
        <div class="flex flex-col gap-1.5 max-w-[85%]">
          <div class="p-3.5 rounded-2xl rounded-tl-sm bg-surface-container-low border border-outline-variant/30 text-on-surface leading-relaxed">
            Welcome to <b>ArchiveGov AI Sovereign Copilot</b>. Query historical records, land settlements, colonial fiscal assessments (rupees/annas/pies), or public health epidemics with verifiable citations.
          </div>
          <span class="text-[10px] font-telemetry-code text-on-surface-variant px-1 flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-secondary"></span> Grounded on Certified Vault Corpus
          </span>
        </div>
      </div>
    `;
    this.messagesContainer.insertAdjacentHTML('beforeend', welcomeHtml);
  },

  async loadSuggestions() {
    const container = document.getElementById('copilot-suggestions-container');
    if (!container) return;

    try {
      const docParam = this.currentScope === 'current' && this.activeDocumentId ? `?document_id=${encodeURIComponent(this.activeDocumentId)}` : '';
      const res = await API.get(`/api/v1/copilot/suggestions${docParam}`);
      if (res && res.suggestions && res.suggestions.length > 0) {
        container.innerHTML = res.suggestions.map(s => `
          <button 
            type="button" 
            class="copilot-suggestion-card text-left p-2 rounded-xl bg-surface-container hover:bg-surface-container-high border border-outline-variant/30 text-on-surface hover:text-secondary text-[11px] transition-all flex items-start gap-1.5 group shadow-sm cursor-pointer"
            data-copilot-query="${this.escapeHtml(s)}"
          >
            <span class="material-symbols-outlined text-[14px] text-tertiary shrink-0 mt-0.5 group-hover:scale-110 transition-transform">help_outline</span>
            <span class="line-clamp-2">${this.escapeHtml(s)}</span>
          </button>
        `).join('');
      }
    } catch (e) {
      console.warn('Failed to load copilot suggestions:', e);
    }
  },

  askSuggestion(question) {
    if (!this.inputField) {
      this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
    }
    if (this.inputField) {
      this.inputField.value = question;
    }
    this.sendDirectQuery(question);
  },

  initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      if (this.voiceBtn) {
        this.voiceBtn.classList.add('opacity-40');
        this.voiceBtn.title = 'Speech recognition not supported in this browser';
      }
      return;
    }

    try {
      this.recognition = new SpeechRec();
      this.recognition.continuous = false;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';

      this.recognition.onstart = () => {
        this.isListening = true;
        this.updateVoiceButtonState(true);
      };

      this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (transcript) {
          if (!this.inputField) {
            this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
          }
          if (this.inputField) {
            this.inputField.value = transcript;
          }
          this.sendDirectQuery(transcript);
        }
      };

      this.recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        this.isListening = false;
        this.updateVoiceButtonState(false);
      };

      this.recognition.onend = () => {
        this.isListening = false;
        this.updateVoiceButtonState(false);
      };
    } catch (e) {
      console.warn('Speech recognition init error:', e);
    }
  },

  toggleVoiceInput() {
    if (!this.recognition) {
      if (typeof Toast !== 'undefined') {
        Toast.warning('Voice recognition is not available in your browser.', 'Speech API');
      }
      return;
    }

    if (this.isListening) {
      this.recognition.stop();
    } else {
      try {
        this.recognition.start();
      } catch (e) {
        console.error('Failed to start recognition:', e);
      }
    }
  },

  updateVoiceButtonState(listening) {
    const icon = document.getElementById('copilot-voice-icon');
    if (!this.voiceBtn) {
      this.voiceBtn = document.getElementById('copilot-voice-btn');
    }
    if (!this.voiceBtn || !icon) return;

    if (listening) {
      this.voiceBtn.classList.add('bg-error/20', 'border-error', 'text-error', 'animate-pulse');
      icon.textContent = 'settings_voice';
    } else {
      this.voiceBtn.classList.remove('bg-error/20', 'border-error', 'text-error', 'animate-pulse');
      icon.textContent = 'mic';
    }
  },

  async handleSend() {
    if (!this.inputField) {
      this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
    }
    if (!this.sendBtn) {
      this.sendBtn = document.getElementById('copilot-send-btn');
    }
    if (!this.inputField) return;

    const query = this.inputField.value.trim();
    if (!query) return;

    this.inputField.value = '';
    await this.sendDirectQuery(query);
  },

  async sendDirectQuery(query) {
    if (!query || !query.trim()) return;
    const cleanQuery = query.trim();

    if (!this.inputField) {
      this.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
    }
    if (!this.sendBtn) {
      this.sendBtn = document.getElementById('copilot-send-btn');
    }
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }

    if (this.inputField) this.inputField.disabled = true;
    if (this.sendBtn) this.sendBtn.disabled = true;

    // Render User Bubble
    this.appendUserMessage(cleanQuery);

    // Render Temporary Assistant Loading Bubble
    const loadingId = 'copilot-loading-' + Date.now();
    this.appendLoadingBubble(loadingId);
    this.scrollToBottom();

    try {
      const deptFilter = document.getElementById('copilot-dept-filter');
      const dept = deptFilter ? deptFilter.value : (this.currentDepartment || 'all');

      // Contextual auto-scoping if user asks about "this document" while on a document page
      let targetDocId = this.currentScope === 'current' ? this.activeDocumentId : null;
      if (!targetDocId && this.activeDocumentId) {
        const qLower = cleanQuery.toLowerCase();
        if (qLower.includes('this document') || qLower.includes('in this document') || qLower.includes('the document') || qLower.includes('this folio') || qLower.includes('this record') || qLower.includes('matter in') || qLower.includes('give the matter')) {
          targetDocId = this.activeDocumentId;
        }
      }

      const payload = {
        query: cleanQuery,
        department: dept,
        document_id: targetDocId,
        chat_history: this.chatHistory.slice(-4)
      };

      const res = await API.post('/api/v1/copilot/ask', payload);
      this.removeLoadingBubble(loadingId);

      // Save to chat history
      this.chatHistory.push({ role: 'user', content: cleanQuery });
      this.chatHistory.push({ role: 'assistant', content: res.answer });

      // Render Assistant Answer Bubble
      this.appendAssistantMessage(res.answer, res.citations || [], res.engine_used);
    } catch (err) {
      console.error('Copilot Query Error:', err);
      this.removeLoadingBubble(loadingId);
      this.appendErrorMessage(err.message || 'Archival RAG retrieval failed. Please verify repository state.');
    } finally {
      if (this.inputField) {
        this.inputField.disabled = false;
        this.inputField.focus();
      }
      if (this.sendBtn) this.sendBtn.disabled = false;
      this.scrollToBottom();
    }
  },

  appendUserMessage(text) {
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }
    if (!this.messagesContainer) return;

    const msgHtml = `
      <div class="flex items-start justify-end gap-2 animate-fadeIn">
        <div class="max-w-[85%] rounded-2xl rounded-tr-sm bg-primary-container text-on-primary-container p-3 text-xs leading-relaxed shadow-md font-medium">
          ${this.escapeHtml(text)}
        </div>
      </div>
    `;
    this.messagesContainer.insertAdjacentHTML('beforeend', msgHtml);
  },

  appendLoadingBubble(id) {
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }
    if (!this.messagesContainer) return;

    const loadingHtml = `
      <div id="${id}" class="flex items-start gap-2 animate-fadeIn">
        <div class="w-7 h-7 rounded-lg bg-surface-container-high border border-outline-variant/30 flex items-center justify-center text-secondary shrink-0 mt-0.5">
          <span class="material-symbols-outlined text-[15px]">auto_awesome</span>
        </div>
        <div class="rounded-2xl rounded-tl-sm bg-surface-container-low border border-outline-variant/30 p-3 text-xs text-on-surface-variant flex items-center gap-2">
          <div class="flex items-center gap-1.5 py-1">
            <span class="w-2 h-2 rounded-full bg-secondary animate-bounce"></span>
            <span class="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]"></span>
            <span class="w-2 h-2 rounded-full bg-tertiary animate-bounce [animation-delay:0.4s]"></span>
          </div>
          <span class="font-mono text-[11px] text-outline ml-1">Synthesizing archival records...</span>
        </div>
      </div>
    `;
    this.messagesContainer.insertAdjacentHTML('beforeend', loadingHtml);
  },

  removeLoadingBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  },

  appendAssistantMessage(answer, citations, engine) {
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }
    if (!this.messagesContainer) return;

    const msgId = 'msg-' + Date.now();
    const formattedAnswer = this.formatMarkdown(answer);

    // Citations list markup
    let citationsHtml = '';
    if (citations && citations.length > 0) {
      citationsHtml = `
        <div class="mt-3 pt-2.5 border-t border-surface-container-highest/60 flex flex-col gap-2">
          <button 
            type="button" 
            class="flex items-center justify-between text-[11px] font-mono text-secondary hover:text-white transition-colors cursor-pointer"
            onclick="document.getElementById('cites-${msgId}').classList.toggle('hidden')"
          >
            <span class="flex items-center gap-1 font-semibold">
              <span class="material-symbols-outlined text-[15px]">verified</span>
              <span>Cited Archival Folios (${citations.length})</span>
            </span>
            <span class="material-symbols-outlined text-[14px]">expand_more</span>
          </button>
          
          <div id="cites-${msgId}" class="flex flex-col gap-2 pt-1">
            ${citations.map(c => `
              <div class="p-2.5 rounded-xl bg-surface-container-lowest/90 border border-outline-variant/30 flex flex-col gap-1.5 shadow-inner">
                <div class="flex items-center justify-between gap-1">
                  <a href="${c.archive_url}" class="font-mono text-[11px] font-bold text-secondary hover:underline truncate">
                    ${c.document_id}
                  </a>
                  <span class="px-1.5 py-0.2 rounded bg-cyan-500/15 text-cyan-400 font-mono text-[9px] font-semibold">
                    ${Math.round(c.relevance_score * 100)}% Match
                  </span>
                </div>
                <div class="text-[11px] font-semibold text-white truncate" title="${this.escapeHtml(c.title)}">
                  ${this.escapeHtml(c.title)}
                </div>
                <p class="text-[10px] text-on-surface-variant/90 line-clamp-2 leading-relaxed">
                  "${this.escapeHtml(c.snippet)}"
                </p>
                <div class="flex items-center justify-between pt-1 border-t border-surface-container-highest/40 text-[10px] font-mono">
                  <span class="text-outline">${c.original_date || 'N/A'} • ${c.department_id || 'General'}</span>
                  <div class="flex items-center gap-2">
                    <a href="${c.lens_url || `/lens/${c.document_id}?page=1`}" class="text-secondary-fixed hover:underline flex items-center gap-0.5">
                      <span class="material-symbols-outlined text-[11px]">center_focus_strong</span>
                      <span>Lens</span>
                    </a>
                    <span class="text-outline">•</span>
                    <a href="${c.archive_url}" class="text-secondary hover:underline">Inspect</a>
                    <span class="text-outline">•</span>
                    <a href="${c.ocr_url}" class="text-primary hover:underline">OCR</a>
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    // Engine badge label
    let engineBadge = 'Offline Sovereign RAG';
    if (engine === 'google_ai_studio') engineBadge = 'Google Gemini 2.5 Flash';
    else if (engine === 'openrouter') engineBadge = 'OpenRouter Cloud';

    const msgHtml = `
      <div id="${msgId}" class="flex items-start gap-2.5 animate-fadeIn">
        <div class="w-7 h-7 rounded-lg bg-surface-container-high border border-outline-variant/30 flex items-center justify-center text-secondary shrink-0 mt-0.5 shadow-md">
          <span class="material-symbols-outlined text-[15px]">history_edu</span>
        </div>
        <div class="flex-1 rounded-2xl rounded-tl-sm bg-surface-container-low border border-outline-variant/30 p-3.5 shadow-md flex flex-col gap-2">
          <!-- Engine badge & TTS button -->
          <div class="flex items-center justify-between border-b border-surface-container-highest/40 pb-1.5 text-[10px] font-mono">
            <span class="flex items-center gap-1 text-tertiary">
              <span class="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
              <span>${engineBadge}</span>
            </span>
            <button 
              type="button" 
              class="text-outline hover:text-white flex items-center gap-1 transition-colors cursor-pointer"
              onclick="ArchiveCopilot.toggleReadAloud(this, '${msgId}')"
              title="Read Aloud (Text-to-Speech)"
            >
              <span class="material-symbols-outlined text-[14px]">volume_up</span>
              <span>Listen</span>
            </button>
          </div>

          <!-- Formatted Content -->
          <div class="copilot-prose text-on-surface leading-relaxed space-y-2 text-xs">
            ${formattedAnswer}
          </div>

          <!-- Citations Component -->
          ${citationsHtml}
        </div>
      </div>
    `;

    this.messagesContainer.insertAdjacentHTML('beforeend', msgHtml);
    this.scrollToBottom();
  },

  appendErrorMessage(errorText) {
    if (!this.messagesContainer) {
      this.messagesContainer = document.getElementById('copilot-messages');
    }
    if (!this.messagesContainer) return;

    const errorHtml = `
      <div class="flex items-start gap-2.5 animate-fadeIn">
        <div class="w-7 h-7 rounded-lg bg-error-container/40 border border-error/40 flex items-center justify-center text-error shrink-0 mt-0.5">
          <span class="material-symbols-outlined text-[15px]">error</span>
        </div>
        <div class="flex-1 rounded-2xl rounded-tl-sm bg-surface-container-low border border-error/40 p-3 text-xs text-error leading-relaxed">
          ${this.escapeHtml(errorText)}
        </div>
      </div>
    `;
    this.messagesContainer.insertAdjacentHTML('beforeend', errorHtml);
  },

  formatMarkdown(text) {
    if (!text) return '';
    let html = this.escapeHtml(text);

    // Markdown Headers: ### Heading -> <h4>
    html = html.replace(/^### (.*$)/gim, '<h4 class="font-bold text-white text-xs mt-2 mb-1">$1</h4>');
    html = html.replace(/^#### (.*$)/gim, '<h5 class="font-semibold text-secondary text-[11px] mt-1.5 mb-0.5">$1</h5>');

    // Markdown Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-white">$1</strong>');

    // Markdown Italics: *text*
    html = html.replace(/\*(.*?)\*/g, '<em class="text-tertiary italic">$1</em>');

    // Transform [DOC-XXXX-...] into interactive badges linking to /app/archive/DOC-XXXX
    html = html.replace(/\[(DOC-[A-Za-z0-9\-]+)\]/g, (match, docId) => {
      return `<a href="/app/archive/${docId}" class="inline-flex items-center gap-0.5 px-1.5 py-0.2 mx-0.5 rounded bg-primary-container/25 hover:bg-primary-container/45 text-primary border border-primary/40 font-mono text-[10px] font-bold transition-all shadow-sm" title="Inspect Folio ${docId}"><span>${docId}</span><span class="material-symbols-outlined text-[10px]">open_in_new</span></a>`;
    });

    // Blockquotes: > quote
    html = html.replace(/^\> (.*$)/gim, '<blockquote class="pl-2.5 border-l-2 border-secondary/60 text-slate-300 italic my-1.5 text-[11px] bg-surface-container-lowest/60 py-1 rounded-r">$1</blockquote>');

    // Bullet points: • item or - item
    html = html.replace(/^[•\-] (.*$)/gim, '<div class="flex items-start gap-1.5 pl-1 my-0.5"><span class="text-secondary">•</span><span>$1</span></div>');

    // Line breaks
    html = html.replace(/\n\n/g, '<div class="h-1.5"></div>');
    html = html.replace(/\n/g, '<br/>');

    return html;
  },

  toggleReadAloud(buttonEl, messageId) {
    if (!('speechSynthesis' in window)) {
      if (typeof Toast !== 'undefined') {
        Toast.warning('Speech synthesis is not supported on this platform.', 'TTS API');
      }
      return;
    }

    if (this.isSpeaking) {
      this.stopSpeaking();
      buttonEl.innerHTML = `<span class="material-symbols-outlined text-[14px]">volume_up</span><span>Listen</span>`;
      return;
    }

    const msgEl = document.getElementById(messageId);
    if (!msgEl) return;
    const text = msgEl.querySelector('.copilot-prose')?.innerText || '';
    if (!text) return;

    this.isSpeaking = true;
    buttonEl.innerHTML = `<span class="material-symbols-outlined text-[14px] text-error">stop</span><span class="text-error">Stop</span>`;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';

    utterance.onend = () => {
      this.isSpeaking = false;
      buttonEl.innerHTML = `<span class="material-symbols-outlined text-[14px]">volume_up</span><span>Listen</span>`;
    };

    utterance.onerror = () => {
      this.isSpeaking = false;
      buttonEl.innerHTML = `<span class="material-symbols-outlined text-[14px]">volume_up</span><span>Listen</span>`;
    };

    this.activeSpeechUtterance = utterance;
    window.speechSynthesis.speak(utterance);
  },

  stopSpeaking() {
    if ('speechSynthesis' in window && this.isSpeaking) {
      window.speechSynthesis.cancel();
      this.isSpeaking = false;
    }
  },

  scrollToBottom() {
    if (this.messagesContainer) {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
  },

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/'/g, '&#39;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
};

// Global interface for cross-module Copilot invocation
window.ArchiveGovCopilot = {
  openWithPrompt(prompt, docId) {
    ArchiveCopilot.openDrawer();
    if (docId) {
      ArchiveCopilot.activeDocumentId = docId;
      ArchiveCopilot.currentScope = 'current';
      const scopeCurrent = document.getElementById('copilot-scope-current') || document.getElementById('copilot-scope-document');
      const scopeAll = document.getElementById('copilot-scope-all') || document.getElementById('copilot-scope-vault');
      if (scopeCurrent) scopeCurrent.className = 'px-2.5 py-1 rounded-md text-primary font-medium bg-primary/20 transition-all';
      if (scopeAll) scopeAll.className = 'px-2.5 py-1 rounded-md text-on-surface-variant hover:text-on-surface transition-all';
    }
    if (!ArchiveCopilot.inputField) {
      ArchiveCopilot.inputField = document.getElementById('copilot-query-input') || document.getElementById('copilot-input');
    }
    if (ArchiveCopilot.inputField) {
      ArchiveCopilot.inputField.value = prompt;
    }
    ArchiveCopilot.sendDirectQuery(prompt);
  }
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => ArchiveCopilot.init());
