const $ = id => document.getElementById(id);

function el(tag, cls, text, tip) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  if (tip) { n.dataset.tip = tip; n.tabIndex = 0; }
  return n;
}

const tipEl = $('tip');
let tipFor = null, tipPinned = false;

function fillTip(text) {
  tipEl.replaceChildren(...text.split('\n').flatMap((line, i) => {
    const m = line.match(/^([^:.]{1,24}):\s+(.*)$/);
    const nodes = m ? [el('b', '', m[1] + ':'), document.createTextNode(' ' + m[2])] : [document.createTextNode(line)];
    return i ? [document.createTextNode('\n'), ...nodes] : nodes;
  }));
}

function placeTip() {
  if (!tipFor) return;
  const r = tipFor.getBoundingClientRect();
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight, m = 8;
  const left = Math.min(Math.max(r.left + r.width / 2 - w / 2, m), innerWidth - w - m);
  const above = r.top - h - 10;
  tipEl.style.left = left + 'px';
  tipEl.style.top = (above >= m ? above : Math.min(r.bottom + 10, innerHeight - h - m)) + 'px';
}

function showTip(target, pin) {
  if (tipFor && tipFor !== target) tipFor.removeAttribute('aria-describedby');
  tipFor = target;
  tipPinned = pin;
  fillTip(target.dataset.tip);
  target.setAttribute('aria-describedby', 'tip');
  placeTip();
  tipEl.classList.add('on');
}

function hideTip() {
  if (tipFor) tipFor.removeAttribute('aria-describedby');
  tipFor = null;
  tipPinned = false;
  tipEl.classList.remove('on');
}

const sweepTip = () => { if (tipFor && !tipFor.isConnected) hideTip(); };

document.addEventListener('pointerover', ev => {
  if (ev.pointerType !== 'mouse' || ev.target.closest('#tip')) return;
  const t = ev.target.closest('[data-tip]');
  if (t && t !== tipFor) showTip(t, false);
  else if (!t && !tipPinned) hideTip();
});
document.addEventListener('click', ev => {
  if (ev.target.closest('#tip')) return;
  const t = ev.target.closest('[data-tip]');
  if (!t || t.hasAttribute('data-act')) hideTip();
  else if (t === tipFor && tipPinned) hideTip();
  else showTip(t, true);
});
document.addEventListener('focusin', ev => {
  const t = ev.target.closest('[data-tip]');
  if (t && t.matches(':focus-visible')) showTip(t, false);
});
document.addEventListener('focusout', ev => { if (ev.target === tipFor && !tipPinned) hideTip(); });
document.addEventListener('keydown', ev => { if (ev.key === 'Escape') hideTip(); });
addEventListener('scroll', placeTip, true);
addEventListener('resize', placeTip);
