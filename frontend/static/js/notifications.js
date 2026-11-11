/**
 * ArchiveGov AI — Notifications Drawer & Badge Synchronizer
 */

const Notifications = {
  drawerEl: null,
  badgeEl: null,
  listEl: null,

  init() {
    // Only initialize notification drawer if the user is authenticated and header is present
    const path = window.location.pathname.toLowerCase();
    const isPublicPage = (
      path === '/login' ||
      path.startsWith('/login') ||
      path === '/' ||
      path === '/landing' ||
      path === '/forgot-password' ||
      path.endsWith('login.html') ||
      path.endsWith('landingpage.html')
    );

    if (isPublicPage || !document.querySelector('header')) {
      return;
    }

    this.badgeEl = document.querySelector('header button span.bg-error, #notif-badge');
    const bellBtn = document.querySelector('header button[title="Notifications"], header button .material-symbols-outlined');
    
    // Find bell button
    const buttons = document.querySelectorAll('header button');
    buttons.forEach(btn => {
      if (btn.innerHTML.includes('notifications')) {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          this.toggleDrawer();
        });
      }
    });

    this.createDrawer();
    this.fetchNotifications();
    // Poll every 30 seconds
    setInterval(() => this.fetchNotifications(), 30000);
  },

  createDrawer() {
    if (document.getElementById('notif-drawer')) return;
    this.drawerEl = document.createElement('div');
    this.drawerEl.id = 'notif-drawer';
    this.drawerEl.className = 'fixed top-16 right-6 w-96 max-w-[90vw] bg-surface-container-high/95 backdrop-blur-2xl border border-outline-variant/40 rounded-2xl shadow-2xl z-50 hidden flex-col overflow-hidden';
    this.drawerEl.innerHTML = `
      <div class="p-4 border-b border-surface-container-highest flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-secondary text-[20px]">notifications</span>
          <span class="font-headline-sm text-sm text-on-surface font-semibold">Archival Notifications</span>
        </div>
        <button id="markAllReadBtn" class="text-xs text-primary hover:underline">Mark all read</button>
      </div>
      <div id="notif-items" class="max-h-80 overflow-y-auto divide-y divide-surface-container-highest/50 p-1">
        <div class="p-4 text-center text-xs text-on-surface-variant">Checking alerts...</div>
      </div>
      <div class="p-2.5 bg-surface-container-lowest/80 text-center border-t border-surface-container-highest">
        <a href="/app/notifications" class="text-xs text-secondary hover:underline">View All Notifications</a>
      </div>
    `;
    document.body.appendChild(this.drawerEl);
    this.listEl = document.getElementById('notif-items');

    document.getElementById('markAllReadBtn')?.addEventListener('click', async () => {
      await API.post('/api/v1/notifications/read-all', {});
      this.fetchNotifications();
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (this.drawerEl && !this.drawerEl.contains(e.target) && !e.target.closest('button[title="Notifications"]')) {
        this.drawerEl.classList.add('hidden');
      }
    });
  },

  toggleDrawer() {
    if (this.drawerEl) {
      this.drawerEl.classList.toggle('hidden');
      if (!this.drawerEl.classList.contains('hidden')) {
        this.fetchNotifications();
      }
    }
  },

  async fetchNotifications() {
    try {
      const data = await API.get('/api/v1/notifications');
      if (this.badgeEl) {
        if (data.unread_count > 0) {
          this.badgeEl.textContent = data.unread_count;
          this.badgeEl.style.display = 'flex';
        } else {
          this.badgeEl.style.display = 'none';
        }
      }

      if (this.listEl) {
        if (!data.notifications || data.notifications.length === 0) {
          this.listEl.innerHTML = '<div class="p-6 text-center text-xs text-on-surface-variant">No new notifications. Vault secure.</div>';
          return;
        }

        this.listEl.innerHTML = data.notifications.slice(0, 5).map(n => `
          <div class="p-3 hover:bg-surface-container-highest/60 transition-colors flex items-start gap-3 cursor-pointer ${n.is_read ? 'opacity-70' : 'bg-surface-container-lowest/40'}" onclick="Notifications.markRead('${n._id}')">
            <span class="material-symbols-outlined text-secondary text-[18px] shrink-0 mt-0.5">${n.type.includes('fail') ? 'error' : (n.type.includes('review') ? 'pending_actions' : 'check_circle')}</span>
            <div class="flex flex-col min-w-0 flex-1">
              <span class="font-headline-sm text-xs text-on-surface font-semibold truncate">${n.title}</span>
              <span class="font-body-sm text-[11px] text-on-surface-variant line-clamp-2 mt-0.5">${n.message}</span>
              <span class="font-telemetry-code text-[10px] text-outline mt-1">${new Date(n.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
            </div>
            ${!n.is_read ? '<span class="w-1.5 h-1.5 rounded-full bg-secondary shrink-0 mt-1.5"></span>' : ''}
          </div>
        `).join('');
      }
    } catch (err) {
      // Ignore if unauthenticated
    }
  },

  async markRead(notifId) {
    await API.post(`/api/v1/notifications/${notifId}/read`, {});
    this.fetchNotifications();
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Notifications.init();
});

