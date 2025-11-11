// Theme toggle
document.addEventListener('DOMContentLoaded', () => {
  const html = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  const collapse = document.getElementById('collapseSidebar');
  const saved = localStorage.getItem('theme');
  if (saved) html.setAttribute('data-theme', saved);
  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', current);
      localStorage.setItem('theme', current);
    });
  }
  if (collapse) {
    collapse.addEventListener('click', () => {
      const side = document.querySelector('.sidebar');
      const content = document.querySelector('.content');
      const collapsed = side.classList.toggle('collapsed');
      content.style.marginLeft = collapsed ? '70px' : '240px';
    });
  }

  // Count-up animation
  document.querySelectorAll('.count').forEach(el => {
    const target = Number(el.dataset.count || 0);
    const duration = 900; // ms
    const start = performance.now();
    function step(ts) {
      const p = Math.min(1, (ts - start) / duration);
      el.textContent = Math.round(target * p);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });

  // Trend chart
  if (window.__trendLabels && window.__trendValues) {
    const ctx = document.getElementById('trendChart');
    if (ctx) {
      const tc = new Chart(ctx, {
        type: 'line',
        data: {
          labels: window.__trendLabels,
          datasets: [{
            label: 'Events',
            data: window.__trendValues,
            tension: 0.3,
            borderColor: getComputedStyle(document.documentElement).getPropertyValue('--primary').trim(),
            backgroundColor: 'rgba(110, 168, 254, 0.25)'
          }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
      });
      window.__trendChart = tc;
    }
  }

  // Breakdown chart on result page
  if (window.__breakdown) {
    const ctx = document.getElementById('breakdownChart');
    if (ctx) {
      const labels = Object.keys(window.__breakdown);
      const vals = Object.values(window.__breakdown);
      const bc = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: vals,
            backgroundColor: ['#dcfce7', '#fef3c7', '#fee2e2'],
            borderRadius: 8,
          }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
      });
      window.__breakdownChart = bc;
    }
  }

  // Share button
  const shareBtn = document.getElementById('shareBtn');
  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(window.__shareUrl || location.href);
        shareBtn.textContent = 'Link Copied!';
        setTimeout(() => shareBtn.textContent = 'Share Result', 1500);
      } catch (e) {
        alert('Copy failed: ' + e);
      }
    });
  }

  // Update chart colors when theme toggles
  if (toggle) {
    toggle.addEventListener('click', () => {
      const cssPrimary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim();
      if (window.__trendChart) {
        const ds = window.__trendChart.data.datasets[0];
        ds.borderColor = cssPrimary;
        window.__trendChart.update('none');
      }
    });
  }
});