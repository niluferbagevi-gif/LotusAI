/**
 * LotusAI Service Worker
 * Version: 2.6.0
 * Description: PWA offline support and caching
 */

const STATIC_CACHE = 'lotus-static-v2.6.0';
const DYNAMIC_CACHE = 'lotus-dynamic-v2.6.0';

// Static resources to cache
// Not: Projede icon-192/icon-512 dosyaları yoksa addAll install'ı düşürür.
// Bu yüzden ikonları şimdilik cache listesine eklemiyoruz.
const STATIC_URLS = [
  '/',
  '/static/manifest.json'
];

// Basit offline HTML (dosya gerektirmez)
const OFFLINE_HTML = `
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>LotusAI | Offline</title>
  <style>
    body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial;
         background:#0f1115;color:#e9edef;display:flex;min-height:100vh;align-items:center;justify-content:center;}
    .card{max-width:520px;padding:24px;border:1px solid rgba(255,255,255,.08);
          border-radius:16px;background:rgba(30,34,40,.95);box-shadow:0 10px 30px rgba(0,0,0,.35);}
    h1{margin:0 0 10px 0;font-size:20px;}
    p{margin:0;color:rgba(233,237,239,.75);line-height:1.5}
    .hint{margin-top:14px;font-size:13px;color:rgba(233,237,239,.55)}
  </style>
</head>
<body>
  <div class="card">
    <h1>Bağlantı yok</h1>
    <p>Şu anda internete erişemiyoruz. Bağlantınız geldiğinde sayfayı yenileyin.</p>
    <div class="hint">LotusAI PWA Offline Mod</div>
  </div>
</body>
</html>
`;

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
// FETCH EVENT - Cache First Strategy (static/dynamic assets)
// ═══════════════════════════════════════════════════════════════
self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // API isteklerini cache’leme (stale cevap riskini azaltır)
  const url = new URL(request.url);
  if (url.pathname.startsWith('/api/')) return;

  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) return cachedResponse;

        return fetch(request)
          .then((networkResponse) => {
            if (!networkResponse || networkResponse.status !== 200) {
              return networkResponse;
            }

            const responseToCache = networkResponse.clone();

            if (shouldCache(request.url)) {
              caches.open(DYNAMIC_CACHE)
                .then((cache) => cache.put(request, responseToCache))
                .catch(() => {});
            }

            return networkResponse;
          })
          .catch((error) => {
            console.error('[ServiceWorker] Fetch failed:', error);

            const accept = request.headers.get('accept') || '';
            if (accept.includes('text/html')) {
              return new Response(OFFLINE_HTML, {
                headers: { 'Content-Type': 'text/html; charset=utf-8' }
              });
            }

            // Diğer istekler için “boş” döndürmek yerine fail bırak
            throw error;
          });
      })
  );
});

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════
function shouldCache(url) {
  return (
    url.includes('/static/') ||
    url.endsWith('.css') ||
    url.endsWith('.js') ||
    url.endsWith('.png') ||
    url.endsWith('.jpg') ||
    url.endsWith('.jpeg') ||
    url.endsWith('.webp') ||
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
      return Promise.all(cacheNames.map((cacheName) => caches.delete(cacheName)));
    });
  }
});