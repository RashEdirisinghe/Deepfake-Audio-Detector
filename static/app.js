// Shared behaviour: mobile nav + the animated "spectrogram ribbon" signature element.

document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // Mark current page in nav
  const here = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a[data-page]').forEach(a => {
    if (a.dataset.page === here) a.classList.add('active');
  });

  document.querySelectorAll('.ribbon[data-generate]').forEach(buildRibbon);
});

const MAGMA = ['#4C3BCF', '#7A3AB8', '#A8339E', '#C2317E', '#E1445F', '#F5A623'];

function buildRibbon(el) {
  const count = parseInt(el.dataset.bars || '48', 10);
  const animated = el.classList.contains('animated');
  const frag = document.createDocumentFragment();

  for (let i = 0; i < count; i++) {
    const bar = document.createElement('span');
    // Smooth pseudo-random height profile (a couple of overlapping sine waves)
    const t = i / count;
    const h =
      0.32 +
      0.28 * Math.abs(Math.sin(t * Math.PI * 3.1 + 0.6)) +
      0.22 * Math.abs(Math.sin(t * Math.PI * 7.7));
    const clamped = Math.min(1, Math.max(0.12, h));
    bar.style.height = `${clamped * 100}%`;

    const colorIdx = Math.floor(t * (MAGMA.length - 1));
    bar.style.setProperty('--bar-color', MAGMA[colorIdx]);

    if (animated) {
      const min = Math.max(0.15, clamped - 0.35);
      const max = Math.min(1, clamped + 0.25);
      bar.style.setProperty('--h-min', min.toFixed(2));
      bar.style.setProperty('--h-max', max.toFixed(2));
      bar.style.animationDelay = `${(-i * 0.07).toFixed(2)}s`;
      bar.style.animationDuration = `${(1.8 + (i % 5) * 0.15).toFixed(2)}s`;
    }
    frag.appendChild(bar);
  }
  el.appendChild(frag);
}