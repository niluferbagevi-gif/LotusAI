const CACHE_NAME = 'lotus-ai-v2';
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.png'
  // Buraya CSS veya JS dosyaları eklenebilir
];

// Yükleme (Install) Olayı
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[ServiceWorker] Önbellek açıldı');
        return cache.addAll(urlsToCache);
      })
  );
});

// Yakalama (Fetch) Olayı - Cache First Stratejisi
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Önbellekte varsa döndür
        if (response) {
          return response;
        }
        // Yoksa ağdan çek
        return fetch(event.request).catch(() => {
            // Ağ hatası durumunda (Offline) ve istek bir HTML sayfası ise
            // Burada özel bir offline.html döndürülebilir.
            // Şimdilik console'a hata basıyoruz.
            console.log('[ServiceWorker] Ağ hatası ve önbellekte yok:', event.request.url);
        });
      })
  );
});

// Aktivasyon Olayı - Eski önbellek temizliği
self.addEventListener('activate', (event) => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('[ServiceWorker] Eski önbellek siliniyor:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});