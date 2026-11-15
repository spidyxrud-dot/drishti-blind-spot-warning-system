document.documentElement.classList.add('js-ready');

const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#main-nav');
menuButton?.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(open));
  navigation?.classList.toggle('open', open);
});
navigation?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
  menuButton?.setAttribute('aria-expanded', 'false');
  navigation.classList.remove('open');
}));

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

const copyButton = document.querySelector('#copy-command');
const copyStatus = document.querySelector('#copy-status');
copyButton?.addEventListener('click', async () => {
  const command = 'run.bat --demo';
  try {
    await navigator.clipboard.writeText(command);
  } catch {
    const temporary = document.createElement('textarea');
    temporary.value = command;
    temporary.setAttribute('readonly', '');
    temporary.style.position = 'fixed';
    temporary.style.opacity = '0';
    document.body.appendChild(temporary);
    temporary.select();
    document.execCommand('copy');
    temporary.remove();
  }
  if (copyStatus) copyStatus.textContent = 'Copied. Run this command from the project folder on Windows.';
  copyButton.textContent = 'Copied';
  window.setTimeout(() => { copyButton.textContent = 'Copy run command'; }, 1800);
});

const roadVideo = document.querySelector('.road-video');
if (roadVideo) {
  const prefersLessMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (prefersLessMotion.matches) roadVideo.pause();
  prefersLessMotion.addEventListener?.('change', (event) => {
    if (event.matches) roadVideo.pause();
    else roadVideo.play().catch(() => {});
  });
}
