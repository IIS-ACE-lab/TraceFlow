// services/visualization/mobile/manifest.js
// The home-screen app's manifest, with this link's token (its start_url keeps it): Android's "Install app".
(function () {
  const t = new URLSearchParams(window.location.search).get('t');
  if (!t) return;
  const link = document.createElement('link');
  link.rel = 'manifest';
  link.href = `/manifest.webmanifest?t=${encodeURIComponent(t)}`;
  document.head.appendChild(link);
}());
