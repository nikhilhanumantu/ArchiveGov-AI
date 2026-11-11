/**
 * ArchiveGov AI — Authentication & Session Manager
 */
const Auth = {
  async login(usernameOrEmail, password) {
    try {
      const res = await API.post('/api/v1/auth/login', {
        username_or_email: usernameOrEmail,
        password: password
      });
      Toast.success('Government session authenticated.', 'Access Granted');
      // Read redirect param or go to dashboard, ensuring no self-looping
      const params = new URLSearchParams(window.location.search);
      let redirect = params.get('redirect') || '/app/dashboard';
      if (!redirect || redirect.toLowerCase().startsWith('/login') || redirect === '/' || redirect === '/landing') {
        redirect = '/app/dashboard';
      }
      setTimeout(() => {
        window.location.href = redirect;
      }, 500);
      return res;
    } catch (err) {
      Toast.error(err.message, 'Authentication Failed');
      throw err;
    }
  },

  async logout() {
    try {
      await API.post('/api/v1/auth/logout', {});
    } catch (e) {
      console.warn('Logout error', e);
    }
    Toast.info('Session terminated safely.', 'Signed Out');
    setTimeout(() => {
      window.location.href = '/login';
    }, 400);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  // Bind any sign-out buttons across the application
  const logoutButtons = document.querySelectorAll('[data-path="login"], [title="Sign Out"], #signOutBtn');
  logoutButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      Auth.logout();
    });
  });
});

