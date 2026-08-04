// hirn website behaviour. No framework, no build step.
//
// The initial theme is applied by a tiny inline script in <head> so there is no
// flash of the wrong palette. Everything else lives here and is deferred.

(function () {
  'use strict';

  var THEME_KEY = 'hirn-theme';
  var root = document.documentElement;

  // ── Theme ───────────────────────────────────────────────────────────────
  function currentTheme() {
    return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private mode */ }
    var icon = document.querySelector('[data-theme-icon]');
    if (icon) icon.textContent = theme === 'dark' ? '☾' : '☀';
    document.dispatchEvent(new CustomEvent('hirn:themechange', { detail: { theme: theme } }));
  }

  function initTheme() {
    var icon = document.querySelector('[data-theme-icon]');
    if (icon) icon.textContent = currentTheme() === 'dark' ? '☾' : '☀';
    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
      });
    }
  }

  // ── Mobile navigation ───────────────────────────────────────────────────
  function initNavToggle() {
    var toggle = document.getElementById('nav-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function () {
      var open = document.body.classList.toggle('nav-open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  // ── Copy buttons on code blocks ─────────────────────────────────────────
  // Docs here carry ~350 code blocks; selecting one by hand across a scrolling
  // <pre> is the single most common friction point in a reference page.
  function initCopyButtons() {
    var blocks = document.querySelectorAll('.prose pre');
    if (!blocks.length || !navigator.clipboard) return;

    blocks.forEach(function (pre) {
      // Diagrams are replaced by rendered SVG; a copy button there is noise.
      if (pre.querySelector('code[data-lang="mermaid"]')) return;

      var wrap = document.createElement('div');
      wrap.className = 'code-wrap';
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);

      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy-btn';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');

      button.addEventListener('click', function () {
        var code = pre.querySelector('code') || pre;
        navigator.clipboard.writeText(code.innerText.replace(/\n$/, '')).then(
          function () { flash(button, 'Copied', 'ok'); },
          function () { flash(button, 'Failed', 'err'); }
        );
      });

      wrap.appendChild(button);
    });
  }

  function flash(button, label, state) {
    var original = 'Copy';
    button.textContent = label;
    button.classList.add(state);
    button.disabled = true;
    setTimeout(function () {
      button.textContent = original;
      button.classList.remove(state);
      button.disabled = false;
    }, 1400);
  }

  // ── Scrollspy: highlight the heading currently being read ───────────────
  function initScrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[href^="#"]'));
    if (!links.length) return;

    var byId = {};
    var targets = [];
    links.forEach(function (link) {
      var id = decodeURIComponent(link.getAttribute('href').slice(1));
      var el = document.getElementById(id);
      if (!el) return;
      byId[id] = link;
      targets.push(el);
    });
    if (!targets.length) return;

    var active = null;
    function mark(id) {
      if (active === id) return;
      if (active && byId[active]) byId[active].classList.remove('active');
      active = id;
      if (byId[id]) byId[id].classList.add('active');
    }

    // Bias the viewport to its upper region so the highlighted entry is the
    // heading the reader is under, not one that merely happens to be visible.
    var observer = new IntersectionObserver(
      function (entries) {
        var visible = entries
          .filter(function (e) { return e.isIntersecting; })
          .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
        if (visible.length) mark(visible[0].target.id);
      },
      { rootMargin: '-70px 0px -70% 0px', threshold: 0 }
    );
    targets.forEach(function (t) { observer.observe(t); });
  }

  // ── Keep the active sidebar entry in view ───────────────────────────────
  function revealActiveNavItem() {
    var current = document.querySelector('.sidebar-nav a[aria-current="page"]');
    var sidebar = document.getElementById('sidebar');
    if (!current || !sidebar) return;
    var offset = current.offsetTop - sidebar.clientHeight / 2;
    if (offset > 0) sidebar.scrollTop = offset;
  }

  // ── Diagrams ────────────────────────────────────────────────────────────
  // Loaded only when a diagram exists, and re-rendered when the theme changes
  // so a toggled page does not keep dark diagrams on a light background.
  function initMermaid() {
    var blocks = document.querySelectorAll('pre code[data-lang="mermaid"]');
    if (!blocks.length) return;

    var sources = [];
    var hosts = [];
    blocks.forEach(function (block) {
      var pre = block.closest('pre') || block;
      var host = document.createElement('div');
      host.className = 'mermaid';
      sources.push(block.innerText);
      host.textContent = block.innerText;
      pre.replaceWith(host);
      hosts.push(host);
    });

    var mermaidRef = null;
    function render() {
      if (!mermaidRef) return;
      hosts.forEach(function (host, i) {
        host.removeAttribute('data-processed');
        host.innerHTML = '';
        host.textContent = sources[i];
      });
      mermaidRef.initialize({
        startOnLoad: false,
        theme: currentTheme() === 'dark' ? 'dark' : 'default',
        securityLevel: 'strict',
        fontFamily: 'ui-sans-serif, system-ui, sans-serif'
      });
      mermaidRef.run({ nodes: hosts }).catch(function (error) {
        console.warn('mermaid render failed', error);
      });
    }

    import('https://cdn.jsdelivr.net/npm/mermaid@' + (root.dataset.mermaid || '11.4.1') +
           '/dist/mermaid.esm.min.mjs')
      .then(function (mod) {
        mermaidRef = mod.default;
        render();
        document.addEventListener('hirn:themechange', render);
      })
      .catch(function (error) {
        // The CDN is unreachable. Leave the source visible rather than a blank
        // gap where a picture should have been.
        console.warn('mermaid unavailable; diagrams left as source', error);
        hosts.forEach(function (host, i) {
          var pre = document.createElement('pre');
          var code = document.createElement('code');
          code.textContent = sources[i];
          pre.appendChild(code);
          host.replaceWith(pre);
        });
      });
  }

  // ── Search ──────────────────────────────────────────────────────────────
  function initSearch() {
    var input = document.getElementById('search-input');
    var results = document.getElementById('search-results');
    if (!input || !results) return;

    var index = null;
    var docs = null;
    var loading = false;
    var cursor = -1;

    function status(message) {
      results.innerHTML = '<p class="search-empty">' + message + '</p>';
      results.hidden = false;
    }

    // The index covers full page content, so it is a real download. Showing
    // progress matters: a silent pause on the first keystroke reads as "search
    // is broken" rather than "search is loading".
    function load() {
      if (index || loading) return;
      loading = true;
      if (input.value.trim().length >= 2) status('Loading search index…');

      var script = document.createElement('script');
      script.src = input.dataset.lib;
      script.onload = function () {
        fetch(input.dataset.index)
          .then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
          })
          .then(function (data) {
            index = window.elasticlunr.Index.load(data);
            docs = data.documentStore.docs;
            if (input.value) run();
          })
          .catch(function () {
            loading = false;
            status('Search index failed to load. Reload the page to retry.');
          });
      };
      script.onerror = function () {
        loading = false;
        status('Search is unavailable offline.');
      };
      document.head.appendChild(script);
    }

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
    }

    // Show the part of the page that actually matched, not its first sentence.
    function excerpt(body, terms) {
      var lower = body.toLowerCase();
      var at = -1;
      for (var i = 0; i < terms.length; i++) {
        at = lower.indexOf(terms[i].toLowerCase());
        if (at !== -1) break;
      }
      var start = at === -1 ? 0 : Math.max(0, at - 40);
      var slice = body.slice(start, start + 150).replace(/\s+/g, ' ').trim();
      var html = escapeHtml((start > 0 ? '…' : '') + slice + '…');
      terms.forEach(function (term) {
        if (!term) return;
        html = html.replace(
          new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'),
          '<mark>$1</mark>'
        );
      });
      return html;
    }

    function run() {
      var query = input.value.trim();
      cursor = -1;
      if (query.length < 2) {
        results.hidden = true;
        results.innerHTML = '';
        return;
      }
      if (!index) {
        load();
        status(loading ? 'Loading search index…' : 'Search unavailable.');
        return;
      }
      var terms = query.split(/\s+/);
      var hits = index.search(query, { bool: 'AND', expand: true }).slice(0, 8);
      if (!hits.length) {
        results.innerHTML = '<p class="search-empty">No matches for “' +
          escapeHtml(query) + '”.</p>';
        results.hidden = false;
        return;
      }
      results.innerHTML = hits
        .map(function (hit) {
          var doc = docs[hit.ref];
          return '<a role="option" aria-selected="false" href="' + hit.ref + '">' +
            '<strong>' + escapeHtml(doc.title || hit.ref) + '</strong>' +
            '<span>' + excerpt(doc.body || '', terms) + '</span></a>';
        })
        .join('');
      results.hidden = false;
    }

    function move(step) {
      var options = results.querySelectorAll('a');
      if (!options.length) return;
      if (cursor >= 0) options[cursor].setAttribute('aria-selected', 'false');
      cursor = (cursor + step + options.length) % options.length;
      var option = options[cursor];
      option.setAttribute('aria-selected', 'true');
      option.scrollIntoView({ block: 'nearest' });
    }

    input.addEventListener('focus', load, { once: true });
    input.addEventListener('input', run);

    input.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowDown') { event.preventDefault(); move(1); }
      else if (event.key === 'ArrowUp') { event.preventDefault(); move(-1); }
      else if (event.key === 'Enter') {
        var options = results.querySelectorAll('a');
        if (cursor >= 0 && options[cursor]) {
          event.preventDefault();
          window.location.href = options[cursor].href;
        }
      } else if (event.key === 'Escape') {
        results.hidden = true;
        input.blur();
      }
    });

    document.addEventListener('click', function (event) {
      if (!event.target.closest('.search')) results.hidden = true;
    });

    // "/" focuses search, the convention on every docs site that has one.
    document.addEventListener('keydown', function (event) {
      if (event.key !== '/' || event.metaKey || event.ctrlKey) return;
      var tag = (event.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || event.target.isContentEditable) return;
      event.preventDefault();
      input.focus();
    });
  }

  function init() {
    initTheme();
    initNavToggle();
    initCopyButtons();
    initScrollSpy();
    revealActiveNavItem();
    initSearch();
    initMermaid();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
