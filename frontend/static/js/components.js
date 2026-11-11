/**
 * ArchiveGov AI — UI Components Helper
 * Toasts, Modals, Count-up Numbers, and Formatting
 */

// 1. Toast Notification Manager
const Toast = {
  container: null,

  init() {
    if (!this.container) {
      this.container = document.getElementById('toast-container');
      if (!this.container) {
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.className = 'fixed bottom-5 right-5 z-[9999] flex flex-col gap-2 pointer-events-none max-w-md w-full px-4';
        document.body.appendChild(this.container);
      }
    }
  },

  show(message, type = 'info', title = null) {
    this.init();
    const toast = document.createElement('div');
    toast.className = 'toast-item pointer-events-auto rounded-xl p-4 shadow-2xl backdrop-blur-xl flex items-start gap-3 border transition-all duration-300';

    let icon = 'info';
    let borderColor = 'border-secondary/30';
    let bgColor = 'bg-surface-container-high/95';
    let iconColor = 'text-secondary';

    if (type === 'success') {
      icon = 'check_circle';
      borderColor = 'border-emerald-500/40';
      iconColor = 'text-emerald-400';
    } else if (type === 'error') {
      icon = 'error';
      borderColor = 'border-red-500/40';
      iconColor = 'text-red-400';
    } else if (type === 'warning') {
      icon = 'warning';
      borderColor = 'border-amber-500/40';
      iconColor = 'text-amber-400';
    }

    toast.classList.add(borderColor, bgColor);
    toast.innerHTML = `
      <span class="material-symbols-outlined ${iconColor} text-[22px] shrink-0 mt-0.5">${icon}</span>
      <div class="flex flex-col flex-1 min-w-0">
        ${title ? `<span class="font-headline-sm text-sm text-on-surface font-semibold">${title}</span>` : ''}
        <span class="font-body-sm text-xs text-on-surface-variant leading-relaxed">${message}</span>
      </div>
      <button class="text-outline-variant hover:text-on-surface shrink-0 ml-2" onclick="this.parentElement.remove()">
        <span class="material-symbols-outlined text-[16px]">close</span>
      </button>
    `;

    this.container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
      toast.classList.remove('show');
      toast.classList.add('hide');
      setTimeout(() => toast.remove(), 400);
    }, 4500);
  },

  success(msg, title = 'Operation Successful') { this.show(msg, 'success', title); },
  error(msg, title = 'Action Failed') { this.show(msg, 'error', title); },
  warning(msg, title = 'Security Notice') { this.show(msg, 'warning', title); },
  info(msg, title = 'System Update') { this.show(msg, 'info', title); }
};

// 2. Modal Manager
const Modal = {
  open(modalId) {
    const el = document.getElementById(modalId);
    if (el) {
      el.classList.remove('hidden');
      el.classList.add('flex');
    }
  },
  close(modalId) {
    const el = document.getElementById(modalId);
    if (el) {
      el.classList.add('hidden');
      el.classList.remove('flex');
    }
  }
};

// 3. Count Up Animation for KPI numbers
function animateCountUp(element, target, duration = 1500) {
  if (!element) return;
  const start = 0;
  const startTime = performance.now();
  const isFloat = target.toString().includes('.');

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const currentVal = start + (target - start) * easeOut;

    element.textContent = isFloat ? currentVal.toFixed(1) : Math.floor(currentVal).toLocaleString();

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = isFloat ? target.toFixed(1) : target.toLocaleString();
    }
  }
  requestAnimationFrame(update);
}

