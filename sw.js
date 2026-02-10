/**
 * LotusAI Service Worker
 * Version: 2.5.3
 * Description: PWA offline support and caching
 */

const CACHE_NAME = 'lotus-ai-v2.5.3';
const STATIC_CACHE = 'lotus-static-v2.5.3';
const DYNAMIC_CACHE = 'lotus-dynamic-v2.5.3';

// Static resources to cache
const STATIC_URLS = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// ═══════════════════════════════════════════════════════════════
// INSTALL EVENT
// ═══════════════════════════════════════════════════════════════
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[ServiceWorker] Caching static assets');
        return cache.addAll(STATIC_URLS);
      })
      .then(() => {
        console.log('[ServiceWorker] Installed successfully');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[ServiceWorker] Install failed:', error);
      })
  );
});

// ═══════════════════════════════════════════════════════════════
// ACTIVATE EVENT
// ═══════════════════════════════════════════════════════════════
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  
  const cacheWhitelist = [STATIC_CACHE, DYNAMIC_CACHE];
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (!cacheWhitelist.includes(cacheName)) {
              console.log('[ServiceWorker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[ServiceWorker] Activated successfully');
        return self.clients.claim();
      })
  );
});

// ═══════════════════════════════════════════════════════════════
// FETCH EVENT - Cache First Strategy
// ═══════════════════════════════════════════════════════════════
self.addEventListener('fetch', (event) => {
  const { request } = event;
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          console.log('[ServiceWorker] Serving from cache:', request.url);
          return cachedResponse;
        }
        
        // Network request
        return fetch(request)
          .then((networkResponse) => {
            // Don't cache if not successful
            if (!networkResponse || networkResponse.status !== 200) {
              return networkResponse;
            }
            
            // Clone response
            const responseToCache = networkResponse.clone();
            
            // Cache dynamic resources
            if (shouldCache(request.url)) {
              caches.open(DYNAMIC_CACHE)
                .then((cache) => {
                  cache.put(request, responseToCache);
                });
            }
            
            return networkResponse;
          })
          .catch((error) => {
            console.error('[ServiceWorker] Fetch failed:', error);
            
            // Return offline page for HTML requests
            if (request.headers.get('accept').includes('text/html')) {
              return caches.match('/offline.html');
            }
          });
      })
  );
});

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════
function shouldCache(url) {
  // Cache static assets and API responses
  return (
    url.includes('/static/') ||
    url.includes('/api/') ||
    url.endsWith('.css') ||
    url.endsWith('.js') ||
    url.endsWith('.png') ||
    url.endsWith('.jpg') ||
    url.endsWith('.json')
  );
}

// ═══════════════════════════════════════════════════════════════
// MESSAGE EVENT
// ═══════════════════════════════════════════════════════════════
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => caches.delete(cacheName))
      );
    });
  }
});