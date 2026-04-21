/**
 * KB Graph HUD — standalone neural overlay for the chat window.
 *
 * Architecture coupling contract (grep these if you change server-side):
 *
 *   SSE events consumed (emitted by src/web/app.py chat endpoint):
 *     kb_file         → {path, score}           — RAG retrieval hit
 *     tool_call       → {tool, args}            — agent intends to call
 *     tool_done       → {tool, elapsed_ms, executed} — call completed
 *     tool_result     → {tool, result, info, executed} — result delivered
 *
 *   Graph-relevant tool detection (generic, not a hardcoded list):
 *     Any tool whose args contain "filename" → extract file + heading/section
 *     Falls back to checking "file", "folder" args for folder_tree etc.
 *
 *   HTTP API consumed:
 *     GET /kb/graph/subgraph?file=&heading=&depth=&max_nodes=&max_edges=
 *       → { meta, elements: { nodes: [{data:{id,label,file,heading,tier,source,summary}}], edges: [{data:{id,source,target,type,weight,evidence},classes}] } }
 *
 * Disable: localStorage.setItem('llm_wiki_agent_kb_graph_hud', '0') and reload.
 */
(function () {
  'use strict';
  if (typeof window === 'undefined' || !window.fetch) return;
  if (window.__kbGraphHud) return;
  if (localStorage.getItem('llm_wiki_agent_kb_graph_hud') === '0') return;
  window.__kbGraphHud = true;

  var nativeFetch = window.fetch.bind(window);

  // ── state ────────────────────────────────────────────────────────
  var root = null;       // DOM container
  var cvs = null;        // canvas element
  var c = null;          // 2d context
  var tooltip = null;    // hover tooltip element
  var raf = 0;
  var G = null;          // current graph payload (accumulated across loads)
  var pos = {};          // animated {id: {x,y}}
  var heat = {};         // per-node intensity (0..12)
  var visited = {};      // id → peak heat ever reached (persistence floor)
  var deg = {};          // per-node degree
  var nodeIdx = {};      // id → node data (for propagation lookups)
  var adjList = {};      // id → [{neighbor, edgeType, weight}]
  var angles = {};       // id → deterministic angle around center (stable across frames)
  var nodeOrder = [];    // order nodes were added (drives angle assignment)
  var lastKey = '';
  var fetchT = 0;
  var focus = 0;
  var frame = 0;
  var hoveredNode = null; // currently hovered node id
  var pinnedNode = null;  // clicked-to-pin node id (persists until closed)
  var hideTimer = 0;      // debounced hide timer (gives user time to chase tooltip)
  var pendingCalls = {}; // tool_name → {file, heading} for phase tracking

  // ── styles ───────────────────────────────────────────────────────
  function injectCSS() {
    if (document.getElementById('kbgh-css')) return;
    var s = document.createElement('style');
    s.id = 'kbgh-css';
    s.textContent = [
      // canvas underlay — fixed within chat-container via JS sizing, no scroll interference
      '#kb-graph-hud{position:fixed;z-index:0;pointer-events:none;overflow:hidden;' +
        'opacity:.5;transition:opacity .6s ease}',
      '#kb-graph-hud.active{opacity:.68}',
      '#kb-graph-hud canvas{display:block;width:100%;height:100%}',

      // messages sit above the canvas; push to rails for center river
      '#messages{position:relative;z-index:1;max-width:100%!important;padding:20px 8px!important}',
      '.message.user{max-width:36%!important}',
      '.message.agent{max-width:44%!important}',

      // glass backdrop so text stays readable over the neural overlay
      '.message.user,.message.agent{' +
        'backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px)}',

      // tooltip — hover state is non-interactive; pinned state is fully
      // interactive so the user can copy text, scroll, and linger
      '#kb-graph-tooltip{position:fixed;z-index:100;pointer-events:none;' +
        'background:rgba(10,12,18,0.94);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);' +
        'border:1px solid rgba(120,140,160,0.22);border-radius:8px;padding:12px 16px 14px;' +
        'width:360px;max-width:calc(100vw - 32px);max-height:min(420px,60vh);overflow:hidden;' +
        'opacity:0;transform:translateY(4px);transition:opacity .16s ease,transform .16s ease;' +
        'font-family:"DM Sans",system-ui,sans-serif;box-shadow:0 8px 32px rgba(0,0,0,0.4)}',
      '#kb-graph-tooltip.visible{opacity:1;transform:translateY(0)}',
      '#kb-graph-tooltip.pinned{pointer-events:auto;overflow:auto;' +
        'border-color:rgba(180,200,220,0.35);box-shadow:0 12px 40px rgba(0,0,0,0.55)}',
      '#kb-graph-tooltip .tt-header{display:flex;align-items:flex-start;gap:8px;' +
        'margin-bottom:6px}',
      '#kb-graph-tooltip .tt-label{flex:1;font-size:0.82rem;font-weight:500;color:#e8edf3;' +
        'letter-spacing:0.01em;line-height:1.35;word-break:break-word}',
      '#kb-graph-tooltip .tt-close{flex:0 0 auto;display:none;cursor:pointer;' +
        'width:20px;height:20px;border-radius:4px;color:#64748b;font-size:0.9rem;' +
        'line-height:1;text-align:center;padding:3px 0 0;background:rgba(100,116,139,0.12);' +
        'border:1px solid rgba(100,116,139,0.2);transition:all .15s ease;' +
        'font-family:system-ui,sans-serif}',
      '#kb-graph-tooltip .tt-close:hover{color:#e2e8f0;background:rgba(100,116,139,0.25);' +
        'border-color:rgba(120,140,160,0.4)}',
      '#kb-graph-tooltip.pinned .tt-close{display:block}',
      '#kb-graph-tooltip .tt-file{font-family:"IBM Plex Mono",monospace;font-size:0.65rem;' +
        'color:#64748b;margin-bottom:8px;word-break:break-all;line-height:1.4}',
      '#kb-graph-tooltip .tt-summary{font-size:0.74rem;font-weight:300;color:#aab4c2;' +
        'line-height:1.55;word-break:break-word;white-space:pre-wrap;' +
        'padding:8px 0 0;border-top:1px solid rgba(100,116,139,0.15)}',
      '#kb-graph-tooltip.pinned .tt-summary{color:#c8d0dc}',
      '#kb-graph-tooltip .tt-meta{display:flex;flex-wrap:wrap;gap:6px;' +
        'margin-bottom:4px;align-items:center}',
      '#kb-graph-tooltip .tt-chip{display:inline-block;font-size:0.56rem;font-weight:500;' +
        'text-transform:uppercase;letter-spacing:0.1em;padding:2px 7px;border-radius:3px;' +
        'background:rgba(100,116,139,0.15);color:#94a3b8;border:1px solid rgba(100,116,139,0.18)}',
      '#kb-graph-tooltip .tt-chip.canon{background:rgba(250,200,100,0.12);' +
        'color:#e3c27a;border-color:rgba(250,200,100,0.25)}',
      '#kb-graph-tooltip .tt-chip.wiki{background:rgba(140,200,180,0.12);' +
        'color:#9dc8bc;border-color:rgba(140,200,180,0.25)}',
      '#kb-graph-tooltip .tt-chip.memory{background:rgba(180,160,220,0.12);' +
        'color:#b8a6d8;border-color:rgba(180,160,220,0.25)}',
      '#kb-graph-tooltip .tt-chip.raw{background:rgba(120,130,145,0.12);' +
        'color:#8792a3;border-color:rgba(120,130,145,0.22)}',
      '#kb-graph-tooltip .tt-hint{font-size:0.6rem;color:#4a5668;' +
        'margin-top:8px;letter-spacing:0.05em;font-style:italic}',
      '#kb-graph-tooltip.pinned .tt-hint{display:none}',
      '#kb-graph-tooltip .tt-empty{font-size:0.68rem;color:#5a6478;' +
        'font-style:italic;padding:4px 0 0}',
      // scrollbar when pinned
      '#kb-graph-tooltip::-webkit-scrollbar{width:6px}',
      '#kb-graph-tooltip::-webkit-scrollbar-track{background:transparent}',
      '#kb-graph-tooltip::-webkit-scrollbar-thumb{background:rgba(100,116,139,0.3);' +
        'border-radius:3px}',
    ].join('\n');
    document.head.appendChild(s);
  }

  // ── mount ────────────────────────────────────────────────────────
  function mount() {
    if (root) return;
    injectCSS();
    var chat = document.getElementById('chat-container');
    if (!chat) return;
    root = document.createElement('div');
    root.id = 'kb-graph-hud';
    root.setAttribute('aria-hidden', 'true');
    cvs = document.createElement('canvas');
    root.appendChild(cvs);
    document.body.appendChild(root);
    c = cvs.getContext('2d');

    tooltip = document.createElement('div');
    tooltip.id = 'kb-graph-tooltip';
    document.body.appendChild(tooltip);

    // Pinned tooltip is interactive (pointer-events:auto). Leaving it while
    // pinned does nothing — it stays until explicit close.
    // We can't get click events on the canvas when pointer-events is none,
    // so intercept clicks on the chat container and hit-test ourselves.
    chat.addEventListener('click', onClick, true);
    chat.addEventListener('mousemove', onMouseMove);
    chat.addEventListener('mouseleave', function () {
      if (!pinnedNode) scheduleHide();
    });

    // Global ESC to unpin
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && pinnedNode) unpinTooltip();
    });
    // Click outside tooltip unpins
    document.addEventListener('click', function (ev) {
      if (!pinnedNode) return;
      if (tooltip.contains(ev.target)) return;
      if (ev.target && ev.target.closest && ev.target.closest('#chat-container')) {
        // chat-container click handler will decide (hit node → repin, miss → unpin)
        return;
      }
      unpinTooltip();
    });

    new ResizeObserver(syncSize).observe(chat);
    window.addEventListener('resize', syncSize);
    syncSize();
    startLoop();
  }

  function syncSize() {
    if (!cvs || !root) return;
    var chat = document.getElementById('chat-container');
    if (!chat) return;
    var rect = chat.getBoundingClientRect();
    var dpr = devicePixelRatio || 1;
    var w = Math.max(1, rect.width | 0);
    var h = Math.max(1, rect.height | 0);
    root.style.left = rect.left + 'px';
    root.style.top = rect.top + 'px';
    root.style.width = w + 'px';
    root.style.height = h + 'px';
    cvs.width = (w * dpr) | 0;
    cvs.height = (h * dpr) | 0;
    cvs.style.width = w + 'px';
    cvs.style.height = h + 'px';
    if (c) c.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ── canonical file normalization ─────────────────────────────────
  // Mirrors kb_paths.to_canonical logic so we match node data.file.
  // Handles canonical strings (`canon:foo.md`), bare relative paths
  // (`wiki/foo.md`), and container-absolute paths (`/app/canon/foo.md`
  // or `/app/knowledge/wiki/foo.md`) that arrive via kb_file SSE events.
  function normFile(f) {
    if (!f || typeof f !== 'string') return '';
    f = f.trim();
    // Already canonical (`canon:...`, `knowledge:...`) — pass through,
    // but strip any accidentally-embedded `/app/` prefix in the path part.
    var colon = f.indexOf(':');
    if (colon > 0 && colon < 12) {
      var src = f.slice(0, colon);
      var rest = f.slice(colon + 1);
      rest = rest.replace(/^\/app\//, '');
      // Absolute path after source prefix means the wrapping was wrong:
      // e.g. `knowledge:canon/foo.md` → re-route to `canon:foo.md`.
      if (src === 'knowledge' && /^canon\//.test(rest)) {
        return 'canon:' + rest.slice(6);
      }
      return src + ':' + rest;
    }
    // Strip container-absolute prefix.
    f = f.replace(/^\/app\//, '');
    if (/^canon\//.test(f)) return 'canon:' + f.slice(6);
    if (/^knowledge\//.test(f)) {
      // /app/knowledge/wiki/foo.md → knowledge:wiki/foo.md
      return 'knowledge:' + f.slice(10);
    }
    if (/^(wiki|memory|raw)\//.test(f)) return 'knowledge:' + f;
    return 'knowledge:' + f;
  }

  // ── generic tool → graph root extraction ─────────────────────────
  // Instead of hardcoding tool names, we look for any arg named
  // "filename" (the universal KB tool pattern). This survives new
  // tools being added without HUD changes.
  function extractRoot(args) {
    if (!args || typeof args !== 'object') return null;
    var file = args.filename || args.file || '';
    if (!file && args.folder) file = args.folder;
    if (!file) return null;
    file = normFile(file);
    var heading = args.heading || args.section || '';
    return { file: file, heading: heading };
  }

  // ── data fetch ───────────────────────────────────────────────────
  function schedule(rt) {
    var key = rt.file + '\0' + (rt.heading || '');
    if (key === lastKey && G) { bumpFocus(); return; }
    lastKey = key;
    if (fetchT) clearTimeout(fetchT);
    fetchT = setTimeout(function () { fetchT = 0; load(rt); }, 350);
  }

  function load(rt) {
    var url = '/kb/graph/subgraph?file=' + encodeURIComponent(rt.file) +
      '&heading=' + encodeURIComponent(rt.heading || '') +
      '&depth=2&max_nodes=80&max_edges=200';
    nativeFetch(url, { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (body) {
        if (!body || !body.elements) return;
        var ns = body.elements.nodes || [];
        var es = body.elements.edges || [];

        // Accumulate into G — merge new nodes/edges with existing
        if (!G) G = { elements: { nodes: [], edges: [] } };
        var existingNodeIds = {};
        for (var en = 0; en < G.elements.nodes.length; en++) {
          var eid = G.elements.nodes[en].data && G.elements.nodes[en].data.id;
          if (eid) existingNodeIds[eid] = en;
        }
        var existingEdgeIds = {};
        for (var ee = 0; ee < G.elements.edges.length; ee++) {
          var eeid = G.elements.edges[ee].data && G.elements.edges[ee].data.id;
          if (eeid) existingEdgeIds[eeid] = true;
        }

        var i, id, e;
        for (i = 0; i < ns.length; i++) {
          id = ns[i].data && ns[i].data.id;
          if (!id) continue;
          if (existingNodeIds[id] != null) {
            G.elements.nodes[existingNodeIds[id]] = ns[i];
          } else {
            G.elements.nodes.push(ns[i]);
            nodeOrder.push(id);
          }
          nodeIdx[id] = ns[i].data;
          if (!adjList[id]) adjList[id] = [];
          if (deg[id] == null) deg[id] = 0;
          // Assign a deterministic angle once per node — golden-angle spiral
          // spreads nodes evenly around center, no clustering on a single ring
          if (angles[id] == null) {
            var idx = nodeOrder.indexOf(id);
            angles[id] = (idx * 2.399963229728653) % (Math.PI * 2); // ~137.5°
          }
          // Seed position at the node's current resting target so new nodes
          // appear in place rather than animating from an arbitrary start
          if (!pos[id]) {
            var t0 = targetFor(id);
            pos[id] = { x: t0.x, y: t0.y };
          }
        }
        for (i = 0; i < es.length; i++) {
          e = es[i].data; if (!e) continue;
          if (existingEdgeIds[e.id]) continue;
          existingEdgeIds[e.id] = true;
          G.elements.edges.push(es[i]);
          if (deg[e.source] != null) deg[e.source]++;
          if (deg[e.target] != null) deg[e.target]++;
          if (adjList[e.source]) adjList[e.source].push({ n: e.target, t: e.type, w: e.weight });
          if (adjList[e.target]) adjList[e.target].push({ n: e.source, t: e.type, w: e.weight });
        }

        pulseRoot(rt);
        bumpFocus();
      })
      .catch(function () {});
  }

  // ── heat-driven radial layout ────────────────────────────────────
  // Each node has a deterministic angle (golden-angle spiral) around center.
  // Its radius is inversely proportional to its relevance (heat + visited).
  // Hot nodes pull inward toward center; cold nodes drift to outer ring.
  // Result: a stable radial "nebula" where importance = proximity to center.
  function targetFor(id) {
    var h = heat[id] || 0;
    var v = visited[id] || 0;
    // Relevance combines current heat (transient) + visited floor (persistent)
    var rel = Math.max(h, v * 0.35);
    var a = angles[id] || 0;

    // Radius map: hot nodes tight to center, cold scattered in outer annulus.
    // Anchor points: rel=0 → r~0.30-0.42, rel=2 → r=0.22, rel=6 → r=0.10, rel=12+ → r=0.06
    var r;
    if (rel >= 6) {
      r = 0.10 - Math.min(0.04, (rel - 6) * 0.008);
    } else if (rel >= 2) {
      r = 0.22 - (rel - 2) * 0.03;
    } else if (rel >= 0.3) {
      r = 0.28 - (rel - 0.3) * 0.035;
    } else {
      // Cold nodes: scatter across outer annulus 0.30..0.44 so hundreds of
      // nodes don't stack on one circle. Use angle as a cheap hash.
      var scatter = (Math.sin(a * 7.919) + 1) * 0.5; // 0..1
      r = 0.30 + scatter * 0.14;
    }

    // Tiny per-node radial jitter for hot/mid tier so equal-heat nodes
    // don't perfectly collide (deterministic, not animated)
    if (rel >= 0.3) r += Math.sin(a * 13.37) * 0.018;

    // Ellipse: chat panel is taller than it is wide, squish y slightly
    // so the cluster reads as a wide oval nebula rather than a circle
    return {
      x: 0.5 + Math.cos(a) * r,
      y: 0.5 + Math.sin(a) * r * 0.78,
    };
  }

  // ── heat system ──────────────────────────────────────────────────
  function normH(s) { return (s || '').toLowerCase().trim(); }

  function pulseRoot(rt) {
    if (!G || !G.elements || !G.elements.nodes) return;
    var wf = (rt.file || '').toLowerCase();
    var wh = normH(rt.heading);
    var ns = G.elements.nodes;
    for (var i = 0; i < ns.length; i++) {
      var d = ns[i].data || {};
      if ((d.file || '').toLowerCase() !== wf) continue;
      if (!wh) { fireNode(d.id, 4); continue; }
      var h = normH(d.heading);
      var leaf = h.split(' > ').pop();
      if (h.indexOf(wh) >= 0 || wh.indexOf(leaf) >= 0 || leaf.indexOf(wh) >= 0)
        fireNode(d.id, 6);
      else
        fireNode(d.id, 1.5);
    }
  }

  // Neural propagation: heat spreads along edges to neighbors,
  // attenuated by edge type. similar = high conductance,
  // inter_file = medium, cross_domain/references = low.
  var CONDUCTANCE = { similar: 0.45, inter_file: 0.3, cross_domain: 0.2, references: 0.2, relates_to: 0.15 };

  function fireNode(id, amt) {
    if (!id) return;
    heat[id] = Math.min(12, (heat[id] || 0) + amt);
    visited[id] = Math.max(visited[id] || 0, heat[id]);
    var adj = adjList[id];
    if (!adj) return;
    for (var i = 0; i < adj.length; i++) {
      var cond = CONDUCTANCE[adj[i].t] || 0.1;
      var spread = amt * cond * (adj[i].w || 0.5);
      var nid = adj[i].n;
      heat[nid] = Math.min(12, (heat[nid] || 0) + spread);
      visited[nid] = Math.max(visited[nid] || 0, heat[nid]);
    }
  }

  function addHeat(id, amt) {
    if (!id) return;
    heat[id] = Math.min(12, (heat[id] || 0) + amt);
    visited[id] = Math.max(visited[id] || 0, heat[id]);
  }

  function bumpFocus() {
    focus = Math.min(1, focus + 0.15);
    if (root) root.classList.toggle('active', focus > 0.2);
  }

  function decayHeat() {
    for (var k in heat) {
      if (!Object.prototype.hasOwnProperty.call(heat, k)) continue;
      heat[k] *= 0.982;
      // visited nodes persist at a visible floor proportional to their peak
      var floor = visited[k] ? Math.min(0.6, visited[k] * 0.12) : 0;
      if (heat[k] < floor) heat[k] = floor;
      else if (heat[k] < 0.01 && !visited[k]) delete heat[k];
    }
    // ensure all visited nodes have at least floor heat even if deleted
    for (var v in visited) {
      if (!Object.prototype.hasOwnProperty.call(visited, v)) continue;
      if (!heat[v] && nodeIdx[v]) {
        heat[v] = Math.min(0.6, visited[v] * 0.12);
      }
    }
    focus *= 0.995;
    if (root) root.classList.toggle('active', focus > 0.2);
  }

  // ── SSE event handling ───────────────────────────────────────────
  // Phase-aware: tool_call = intent (mild heat), tool_done+executed = confirmed (strong fire)
  function onEvent(ev, data) {
    // RAG retrieval — file hit before tool loop even starts
    if (ev === 'kb_file') {
      try {
        var kf = JSON.parse(data);
        if (kf && kf.path) schedule({ file: normFile(kf.path), heading: '' });
      } catch (e) {}
      return;
    }

    // Tool intent — agent decided to call (mild pulse)
    if (ev === 'tool_call') {
      try {
        var tc = JSON.parse(data);
        var rt = extractRoot(tc.args);
        if (rt) {
          pendingCalls[tc.tool] = rt;
          schedule(rt);
          // mild intent heat — agent is looking at this area
          pulseRoot(rt);
        }
      } catch (e) {}
      return;
    }

    // Tool done — execution finished. Strong fire only if executed=true.
    if (ev === 'tool_done') {
      try {
        var td = JSON.parse(data);
        var pending = pendingCalls[td.tool];
        if (pending && td.executed) {
          // confirmed read — strong neural fire with propagation
          fireNode_byRoot(pending, 5);
          bumpFocus();
        }
        delete pendingCalls[td.tool];
      } catch (e) {}
      return;
    }

    // tool_executing — same shape as tool_call, use for subgraph load
    if (ev === 'tool_executing') {
      try {
        var te = JSON.parse(data);
        var rt2 = extractRoot(te.args);
        if (rt2) schedule(rt2);
      } catch (e) {}
      return;
    }
  }

  // Fire all graph nodes matching a root (file+heading)
  function fireNode_byRoot(rt, amt) {
    if (!G || !G.elements || !G.elements.nodes) return;
    var wf = (rt.file || '').toLowerCase();
    var wh = normH(rt.heading);
    var ns = G.elements.nodes;
    for (var i = 0; i < ns.length; i++) {
      var d = ns[i].data || {};
      if ((d.file || '').toLowerCase() !== wf) continue;
      if (!wh) { fireNode(d.id, amt); continue; }
      var h = normH(d.heading);
      var leaf = h.split(' > ').pop();
      if (h.indexOf(wh) >= 0 || wh.indexOf(leaf) >= 0 || leaf.indexOf(wh) >= 0)
        fireNode(d.id, amt);
    }
  }

  // ── SSE stream parser ────────────────────────────────────────────
  function parseSse(buf, chunk) {
    buf += chunk;
    var out = [], idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      var block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      var lines = block.split('\n');
      var ev = 'message', dl = [];
      for (var i = 0; i < lines.length; i++) {
        if (lines[i].indexOf('event:') === 0) ev = lines[i].slice(6).trim();
        else if (lines[i].indexOf('data:') === 0) dl.push(lines[i].slice(5).replace(/^\s/, ''));
      }
      out.push({ type: ev, data: dl.join('\n') });
    }
    return { buffer: buf, events: out };
  }

  function consumeStream(resp) {
    if (!resp || !resp.body) return;
    var reader = resp.body.getReader(), dec = new TextDecoder(), buf = '';
    (function step() {
      reader.read().then(function (ch) {
        if (ch.done) return;
        var r = parseSse(buf, dec.decode(ch.value, { stream: true }));
        buf = r.buffer;
        for (var i = 0; i < r.events.length; i++) onEvent(r.events[i].type, r.events[i].data);
        step();
      }).catch(function () {});
    })();
  }

  function isPostChat(args) {
    var req = args[0], opt = args[1] || {};
    var method = (opt.method || 'GET').toUpperCase();
    var u = '';
    if (typeof req === 'string') u = req;
    else if (typeof Request !== 'undefined' && req instanceof Request) {
      u = req.url; if (!opt.method) method = (req.method || 'GET').toUpperCase();
    } else u = (req && req.url) || '';
    if (method !== 'POST' || !u) return false;
    try { return new URL(u, location.origin).pathname.endsWith('/chat'); }
    catch (e) { return false; }
  }

  window.fetch = function () {
    var a = arguments;
    return nativeFetch.apply(window, a).then(function (r) {
      try { if (isPostChat(a) && r && r.ok && r.body && r.clone) consumeStream(r.clone()); }
      catch (e) {}
      return r;
    });
  };

  // ── hit-testing ──────────────────────────────────────────────────
  // Returns the closest node id under the cursor, or null. Hit radius
  // scales with node size (degree + heat) with a forgiving minimum.
  function hitTest(e) {
    if (!cvs || !G || !G.elements) return null;
    var rect = cvs.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var w = rect.width || 1, h = rect.height || 1;
    if (mx < 0 || my < 0 || mx > w || my > h) return null;

    var hitId = null, hitDist = 28; // more generous default radius
    var nodes = G.elements.nodes || [];
    for (var i = 0; i < nodes.length; i++) {
      var d = nodes[i].data || {};
      var id = d.id, p = pos[id];
      if (!p) continue;
      var nx = p.x * w, ny = p.y * h;
      var dx = mx - nx, dy = my - ny;
      var dist = Math.sqrt(dx * dx + dy * dy);
      var nH = heat[id] || 0;
      var nD = deg[id] != null ? deg[id] : 1;
      var r = 2.5 + Math.min(7, nD * 0.45) + nH * 1.0;
      // Bigger hit area for small/cold nodes so they're not impossible to grab
      var hitR = Math.max(r * 2.2, 18);
      if (dist < hitR && dist < hitDist) {
        hitDist = dist;
        hitId = id;
      }
    }
    return hitId;
  }

  // ── hover ────────────────────────────────────────────────────────
  function onMouseMove(e) {
    if (pinnedNode) return; // pinned tooltip doesn't chase the cursor
    var hitId = hitTest(e);

    if (hitId && hitId !== hoveredNode) {
      if (hideTimer) { clearTimeout(hideTimer); hideTimer = 0; }
      hoveredNode = hitId;
      showTooltip(hitId, { clientX: e.clientX, clientY: e.clientY });
    } else if (hitId && hitId === hoveredNode) {
      moveTooltip(e);
    } else if (!hitId && hoveredNode) {
      scheduleHide();
    }
  }

  // ── click (pin / unpin) ──────────────────────────────────────────
  function onClick(e) {
    var hitId = hitTest(e);
    if (hitId) {
      // Stop the event so the document-level "click outside" listener
      // doesn't immediately unpin us.
      e.stopPropagation();
      pinTooltip(hitId, e);
    } else if (pinnedNode) {
      unpinTooltip();
    }
  }

  function pinTooltip(id, e) {
    if (!tooltip) return;
    if (hideTimer) { clearTimeout(hideTimer); hideTimer = 0; }
    pinnedNode = id;
    hoveredNode = id;
    renderTooltip(id);
    tooltip.classList.add('visible');
    tooltip.classList.add('pinned');
    anchorTooltipToNode(id, e);
    addHeat(id, 0.8); // clicked pulse is brighter than hover
  }

  function unpinTooltip() {
    pinnedNode = null;
    if (tooltip) tooltip.classList.remove('pinned');
    scheduleHide();
  }

  function scheduleHide() {
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(function () {
      hideTimer = 0;
      if (pinnedNode) return; // pinned overrides any queued hide
      hoveredNode = null;
      if (tooltip) tooltip.classList.remove('visible');
    }, 280); // ~280ms grace lets the user chase the tooltip with the cursor
  }

  function hideTooltip() {
    if (pinnedNode) return;
    hoveredNode = null;
    if (tooltip) tooltip.classList.remove('visible');
  }

  function showTooltip(id, e) {
    if (!tooltip) return;
    if (!nodeIdx[id]) { hideTooltip(); return; }
    renderTooltip(id);
    tooltip.classList.add('visible');
    moveTooltip(e);
    addHeat(id, 0.3);
  }

  // Count edges this node participates in (works without loading the whole
  // subgraph — we already accumulated adjList as the graph loaded).
  function edgeCountFor(id) {
    var adj = adjList[id];
    return adj ? adj.length : 0;
  }

  function renderTooltip(id) {
    var d = nodeIdx[id];
    if (!d) return;
    var label = d.label || d.heading || d.name || '(untitled)';
    var file = d.file || '';
    var heading = d.heading || '';
    var tier = (d.tier || '').toLowerCase();
    var summary = d.summary || '';
    var nt = d.node_type || '';
    var edges = edgeCountFor(id);
    var isPinned = pinnedNode === id;

    var html = '<div class="tt-header">' +
      '<div class="tt-label">' + escHtml(label) + '</div>' +
      '<div class="tt-close" title="Close (Esc)">×</div>' +
      '</div>';

    var chips = '';
    if (tier) chips += '<span class="tt-chip ' + escHtml(tier) + '">' + escHtml(tier) + '</span>';
    if (nt && nt !== 'chunk') chips += '<span class="tt-chip">' + escHtml(nt) + '</span>';
    if (edges > 0) chips += '<span class="tt-chip">' + edges + ' edge' + (edges === 1 ? '' : 's') + '</span>';
    if (chips) html += '<div class="tt-meta">' + chips + '</div>';

    if (file) {
      var path = escHtml(file);
      if (heading && heading !== label) path += ' &nbsp;›&nbsp; ' + escHtml(heading);
      html += '<div class="tt-file">' + path + '</div>';
    }

    if (summary) {
      html += '<div class="tt-summary">' + escHtml(summary) + '</div>';
    } else {
      html += '<div class="tt-summary tt-empty">No summary stored for this node.</div>';
    }

    if (!isPinned) {
      html += '<div class="tt-hint">click node to pin — esc to close</div>';
    }

    tooltip.innerHTML = html;
    // Close button wiring (rendered even when not pinned, hidden by CSS)
    var closeBtn = tooltip.querySelector('.tt-close');
    if (closeBtn) closeBtn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      unpinTooltip();
    });
  }

  function moveTooltip(e) {
    if (!tooltip || !tooltip.classList.contains('visible')) return;
    if (pinnedNode) return; // pinned tooltip stays put
    positionTooltip(e.clientX + 14, e.clientY - 10);
  }

  // Anchor pinned tooltip near the node's screen position (not the click
  // point) so it's stable if the node drifts slightly with the layout.
  function anchorTooltipToNode(id, fallbackEvent) {
    var p = pos[id];
    if (!p) {
      if (fallbackEvent) positionTooltip(fallbackEvent.clientX + 14, fallbackEvent.clientY - 10);
      return;
    }
    var rect = cvs.getBoundingClientRect();
    var nx = rect.left + p.x * rect.width;
    var ny = rect.top + p.y * rect.height;
    positionTooltip(nx + 18, ny - 20);
  }

  function positionTooltip(x, y) {
    if (!tooltip) return;
    // Clamp inside viewport with 12px padding
    var tw = tooltip.offsetWidth || 360;
    var th = tooltip.offsetHeight || 120;
    if (x + tw + 12 > window.innerWidth) x = Math.max(12, x - tw - 32);
    if (y + th + 12 > window.innerHeight) y = Math.max(12, window.innerHeight - th - 12);
    if (y < 12) y = 12;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }

  function escHtml(s) {
    var d = document.createElement('span');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── rendering ────────────────────────────────────────────────────

  // Label accent by tier — muted, only visible when node is warm
  var TIER_RGB = {
    canon:  [195, 170, 120],
    wiki:   [130, 175, 180],
    memory: [155, 140, 190],
    raw:    [140, 145, 150],
  };
  var DEF_RGB = [150, 155, 165];

  function drawFrame() {
    if (!c || !cvs) return;
    var w = cvs.clientWidth || 1, h = cvs.clientHeight || 1;
    c.clearRect(0, 0, w, h);
    if (!G || !G.elements) return;
    frame++;

    // Compute each node's target from current heat and spring toward it.
    // Critically-damped interpolation: fast approach, graceful settle,
    // no oscillation. Snap to rest when close enough to avoid perpetual
    // micro-jitter.
    for (var k in pos) {
      if (!Object.prototype.hasOwnProperty.call(pos, k)) continue;
      if (!nodeIdx[k]) continue;
      var target = targetFor(k);
      var ddx = target.x - pos[k].x;
      var ddy = target.y - pos[k].y;
      var dsq = ddx * ddx + ddy * ddy;
      if (dsq < 2e-7) {
        pos[k].x = target.x;
        pos[k].y = target.y;
      } else {
        // Ease faster when far, slower when near — smooth arrival
        var ease = 0.08 + Math.min(0.06, dsq * 40);
        pos[k].x += ddx * ease;
        pos[k].y += ddy * ease;
      }
    }

    var nodes = G.elements.nodes || [];
    var edges = G.elements.edges || [];

    // ── edges ──────────────────────────────────────────────
    for (var ei = 0; ei < edges.length; ei++) {
      var ed = edges[ei].data || {};
      var sp = pos[ed.source], tp = pos[ed.target];
      if (!sp || !tp) continue;
      var sx = sp.x * w, sy = sp.y * h, tx = tp.x * w, ty = tp.y * h;
      var eH = (heat[ed.source] || 0) + (heat[ed.target] || 0);

      // edge brightness: nearly invisible at rest, brightens with heat
      var eA = 0.025 + Math.min(0.22, eH * 0.018);
      var eW = 0.3 + Math.min(1.6, eH * 0.1);
      var eg = 70 + Math.min(90, eH * 7);

      // bezier curve (perpendicular offset for organic feel)
      var mx = (sx + tx) / 2, my = (sy + ty) / 2;
      var dx = tx - sx, dy = ty - sy;
      var len = Math.sqrt(dx * dx + dy * dy) + 1;
      var off = len * 0.1;
      var cx = mx + (-dy / len) * off, cy = my + (dx / len) * off;

      c.strokeStyle = 'rgba(' + eg + ',' + eg + ',' + (eg + 6) + ',' + eA + ')';
      c.lineWidth = eW;
      c.beginPath();
      c.moveTo(sx, sy);
      c.quadraticCurveTo(cx, cy, tx, ty);
      c.stroke();

      // traveling pulse — only on hot edges
      if (eH > 0.8) {
        var t = ((frame * 0.01 + ei * 0.31) % 1);
        var u = 1 - t;
        var px = u * u * sx + 2 * u * t * cx + t * t * tx;
        var py = u * u * sy + 2 * u * t * cy + t * t * ty;
        var pA = Math.min(0.5, eH * 0.04);
        var pR = 1.2 + Math.min(2.5, eH * 0.18);
        var g = c.createRadialGradient(px, py, 0, px, py, pR * 2.5);
        g.addColorStop(0, 'rgba(200,205,215,' + pA + ')');
        g.addColorStop(1, 'rgba(200,205,215,0)');
        c.fillStyle = g;
        c.beginPath();
        c.arc(px, py, pR * 2.5, 0, Math.PI * 2);
        c.fill();
      }
    }

    // ── nodes ──────────────────────────────────────────────
    c.textAlign = 'center';
    c.textBaseline = 'top';

    for (var ni = 0; ni < nodes.length; ni++) {
      var d = nodes[ni].data || {};
      var id = d.id, p = pos[id];
      if (!p) continue;
      var nH = heat[id] || 0;
      var nD = deg[id] != null ? deg[id] : 1;
      var nx = p.x * w, ny = p.y * h;

      var isHovered = (id === hoveredNode);
      var baseR = 2.5 + Math.min(7, nD * 0.45);
      var r = baseR + nH * 1.0 + (isHovered ? 2 : 0);

      // core brightness: dark grey → white
      var core = 38 + Math.min(210, nH * 25) + (isHovered ? 40 : 0);

      // outer bloom halo
      if (nH > 0.06) {
        var glR = r * 4 + nH * 3;
        var glA = Math.min(0.18, nH * 0.018);
        var g1 = c.createRadialGradient(nx, ny, r * 0.2, nx, ny, glR);
        g1.addColorStop(0, 'rgba(' + core + ',' + core + ',' + Math.min(255, core + 15) + ',' + glA + ')');
        g1.addColorStop(1, 'rgba(' + core + ',' + core + ',' + Math.min(255, core + 15) + ',0)');
        c.fillStyle = g1;
        c.beginPath();
        c.arc(nx, ny, glR, 0, Math.PI * 2);
        c.fill();
      }

      // core dot — radial gradient
      var cA = 0.1 + Math.min(0.8, nH * 0.07);
      var g2 = c.createRadialGradient(nx, ny, 0, nx, ny, r);
      var cH = Math.min(255, core + 35);
      g2.addColorStop(0, 'rgba(' + cH + ',' + cH + ',' + Math.min(255, cH + 12) + ',' + cA + ')');
      g2.addColorStop(0.55, 'rgba(' + core + ',' + core + ',' + Math.min(255, core + 8) + ',' + (cA * 0.5) + ')');
      g2.addColorStop(1, 'rgba(' + core + ',' + core + ',' + Math.min(255, core + 8) + ',0)');
      c.fillStyle = g2;
      c.beginPath();
      c.arc(nx, ny, r, 0, Math.PI * 2);
      c.fill();

      // crisp ring
      var rA = 0.06 + Math.min(0.35, nH * 0.04);
      c.strokeStyle = 'rgba(' + Math.min(255, core + 45) + ',' + Math.min(255, core + 45) + ',' + Math.min(255, core + 55) + ',' + rA + ')';
      c.lineWidth = 0.4 + nH * 0.06;
      c.beginPath();
      c.arc(nx, ny, r * 0.8, 0, Math.PI * 2);
      c.stroke();

      // label — appears when warm, high-degree, visited, or hovered
      var label = (d.label || '').slice(0, 26);
      if (label && (isHovered || nH > 0.15 || nD > 5 || visited[id])) {
        var fs = 9 + Math.min(2.5, nH * 0.3);
        c.font = '300 ' + fs + 'px "DM Sans",-apple-system,system-ui,sans-serif';
        var lc = TIER_RGB[(d.tier || '').toLowerCase()] || DEF_RGB;
        var la = isHovered ? 0.85 : Math.min(0.65, (visited[id] ? 0.12 : 0.06) + nH * 0.05);
        c.fillStyle = 'rgba(' + lc[0] + ',' + lc[1] + ',' + lc[2] + ',' + la + ')';
        c.fillText(label, nx, ny + r + 4);
      }
    }
  }

  // ── loop ─────────────────────────────────────────────────────────
  function startLoop() {
    if (raf) return;
    (function tick() { decayHeat(); drawFrame(); raf = requestAnimationFrame(tick); })();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
