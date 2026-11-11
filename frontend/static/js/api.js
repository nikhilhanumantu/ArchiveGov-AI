/**
 * ArchiveGov AI — REST API Client Wrapper
 */
const API = {
  baseUrl: '',

  async request(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;
    const defaultHeaders = {
      'Accept': 'application/json',
    };

    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers
      },
      credentials: 'same-origin'
    };

    try {
      const response = await fetch(url, config);
      
      // Handle redirect to login on 401
      if (response.status === 401 && !url.includes('/api/v1/auth/login')) {
        const path = window.location.pathname.toLowerCase();
        const isAuthOrPublicPage = (
          path === '/login' ||
          path.startsWith('/login') ||
          path === '/' ||
          path === '/landing' ||
          path === '/forgot-password' ||
          path.endsWith('login.html') ||
          path.endsWith('landingpage.html')
        );

        if (!isAuthOrPublicPage) {
          const redirectTarget = window.location.pathname + window.location.search;
          window.location.href = `/login?redirect=${encodeURIComponent(redirectTarget)}`;
        }
        throw new Error('Session expired or unauthorized. Please sign in.');
      }

      const contentType = response.headers.get('content-type');
      let data = null;
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else if (contentType && (contentType.includes('text/') || contentType.includes('application/'))) {
        data = await response.text();
      }

      if (!response.ok) {
        const errorMsg = (data && data.detail) || (data && data.message) || `Request failed with status ${response.status}`;
        throw new Error(errorMsg);
      }

      return data;
    } catch (err) {
      console.error(`API Error on [${options.method || 'GET'} ${endpoint}]:`, err);
      throw err;
    }
  },

  get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  },

  post(endpoint, body) {
    return this.request(endpoint, {
      method: 'POST',
      body: body instanceof FormData ? body : JSON.stringify(body)
    });
  },

  put(endpoint, body) {
    return this.request(endpoint, {
      method: 'PUT',
      body: body instanceof FormData ? body : JSON.stringify(body)
    });
  },

  delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
};

