// KAKEN-ATLAS Service Worker（scripts/build_web_map.py が生成。手で編集しない）
// データ版: __DATA_VER__（共有シャードと各ビューの points.bin の内容ハッシュ。変わると古いキャッシュを捨てる）
var DATA_CACHE = 'ka-data-__DATA_VER__';
var PAGE_CACHE = 'ka-pages-v1';
var STATIC_CACHE = 'ka-static-v1';

self.addEventListener('install', function (e) { self.skipWaiting(); });
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) {
      return k.indexOf('ka-') === 0 && k !== DATA_CACHE && k !== PAGE_CACHE && k !== STATIC_CACHE;
    }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});

function cacheFirst(cacheName, req) {
  return caches.open(cacheName).then(function (c) {
    return c.match(req).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        if (res && (res.ok || res.type === 'opaque')) c.put(req, res.clone());
        return res;
      });
    });
  });
}
function networkFirst(cacheName, req) {
  var key = req.url.split('?')[0];  // ?award=… などの検索文字列はページの内容に影響しないので鍵から外す
  return caches.open(cacheName).then(function (c) {
    return fetch(req).then(function (res) {
      if (res && res.ok) c.put(key, res.clone());
      return res;
    }).catch(function () {
      return c.match(key).then(function (hit) { return hit || Response.error(); });
    });
  });
}

self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.hostname.indexOf('goatcounter') >= 0 || url.hostname === 'gc.zgo.at') return;  // 計測は素通し
  var p = url.pathname;
  if (url.origin === location.origin) {
    if (/\/shards\/\d+\.json$/.test(p) || /\/points\.bin$/.test(p)) { e.respondWith(cacheFirst(DATA_CACHE, req)); return; }
    if (/\.(png|jpg|webmanifest)$/.test(p)) { e.respondWith(cacheFirst(STATIC_CACHE, req)); return; }
    if (req.mode === 'navigate' || /\/$/.test(p) || /\.html$/.test(p)) { e.respondWith(networkFirst(PAGE_CACHE, req)); return; }
    return;
  }
  if (url.hostname === 'cdn.plot.ly' || url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
    e.respondWith(cacheFirst(STATIC_CACHE, req)); return;
  }
});
