/**
 * ArchiveGov AI — Navigation & Routing Manager
 * Connects all sidebar items, breadcrumbs, and global search to real application routes.
 */

const ROUTES_MAP = {
  'dashboard': '/app/dashboard',
  'restoration-studio': '/app/restoration',
  'document-upload': '/app/upload',
  'digital-archive': '/app/archive',
  'ocr-and-text-extraction': '/app/ocr',
  'history-and-versions': '/app/history/DOC-1912-KDL',
  'reports-and-analytics': '/app/reports',
  'user-management': '/app/users',
  'settings': '/app/settings',
  'damage-analysis': '/app/damage-analysis',
  'selective-inpainting': '/app/inpainting',
  'restoration-result': '/app/restoration-result'
};

document.addEventListener('DOMContentLoaded', () => {
  const currentPath = window.location.pathname;

  // 1. Connect all nav links with data-path or href="#"
  const navLinks = document.querySelectorAll('aside nav a, header a');
  navLinks.forEach(link => {
    const dataPath = link.getAttribute('data-path');
    if (dataPath && ROUTES_MAP[dataPath]) {
      link.href = ROUTES_MAP[dataPath];
    }

    // Preserve document_id query param if navigating between studio views
    const urlParams = new URLSearchParams(window.location.search);
    const docId = urlParams.get('document_id');
    if (docId && link.href.includes('/app/')) {
      const targetUrl = new URL(link.href, window.location.origin);
      if (['/app/restoration', '/app/damage-analysis', '/app/inpainting', '/app/ocr', '/app/restoration-result'].includes(targetUrl.pathname)) {
        targetUrl.searchParams.set('document_id', docId);
        link.href = targetUrl.toString();
      }
    }

    // Highlight active link
    if (link.pathname === currentPath || (link.getAttribute('data-path') === 'dashboard' && currentPath === '/app/dashboard')) {
      link.classList.add('bg-primary-container', 'text-on-primary-container', 'font-semibold', 'shadow-[0_0_16px_rgba(160,120,255,0.35)]');
      link.classList.remove('text-on-surface-variant');
      link.setAttribute('aria-current', 'page');
    }
  });

  // 2. Global Search Input in Header
  const searchInputs = document.querySelectorAll('header input[type="text"]');
  searchInputs.forEach(input => {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = input.value.trim();
        if (query) {
          window.location.href = `/app/archive?q=${encodeURIComponent(query)}`;
        }
      }
    });
  });
});

