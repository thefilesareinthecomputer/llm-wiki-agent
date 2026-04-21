// R (R) UI - 2026 Glassmorphism Chat Interface
// Uses marked.js for markdown + mermaid.js for diagrams

const API_BASE = '';
let eventSource = null;
let currentFolder = 'knowledge';
let mermaidModule = null;
let currentConversationId = null;
let conversations = [];
let conversationListExpanded = false;
let conversationTotalCount = 0;
const CONVERSATION_PAGE_SIZE = 10;

// Configure marked.js
function initMarkdown() {
  if (typeof marked !== 'undefined') {
    const renderer = new marked.Renderer();
    renderer.code = function(code, language) {
      if (language && typeof hljs !== 'undefined' && hljs.getLanguage(language)) {
        const highlighted = hljs.highlight(code, { language }).value;
        return `<pre><code class="hljs ${language}">${highlighted}</code></pre>`;
      }
      if (typeof hljs !== 'undefined') {
        const highlighted = hljs.highlightAuto(code).value;
        return `<pre><code class="hljs">${highlighted}</code></pre>`;
      }
      return `<pre><code>${code}</code></pre>`;
    };
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
      renderer: renderer,
    });
  }
}

// Initialize mermaid
async function initMermaid() {
  if (typeof mermaid !== 'undefined') {
    mermaidModule = mermaid;
    mermaidModule.initialize({
      theme: 'dark',
      startOnLoad: false,
      securityLevel: 'loose',
    });
  }
}

// DOM Elements
let statusDot, statusText, messagesContainer, textInput, sendBtn;
let fileList, searchInput, searchResults, reindexBtn, ctxBar, ctxLabel, modelSelect;
let sidebar, sidebarToggle, mainLayout;

async function init() {
  // Get DOM elements
  statusDot = document.getElementById('status-dot');
  statusText = document.getElementById('status-text');
  messagesContainer = document.getElementById('messages');
  textInput = document.getElementById('text-input');
  sendBtn = document.getElementById('send-btn');
  fileList = document.getElementById('file-list');
  searchInput = document.getElementById('search-input');
  searchResults = document.getElementById('search-results');
  reindexBtn = document.getElementById('reindex-btn');
  ctxBar = document.getElementById('ctx-bar');
  ctxLabel = document.getElementById('ctx-label');
  modelSelect = document.getElementById('model-select');
  sidebar = document.getElementById('sidebar');
  sidebarToggle = document.getElementById('sidebar-toggle');
  mainLayout = document.getElementById('main-layout');

  // Restore sidebar state
  const sidebarCollapsed = localStorage.getItem('sidebar_collapsed') === 'true';
  if (sidebarCollapsed && sidebar && mainLayout) {
    sidebar.classList.add('collapsed');
    mainLayout.classList.add('sidebar-collapsed');
  }

  // Initialize markdown + mermaid
  initMarkdown();
  await initMermaid();

  // Load conversations and restore session
  await loadConversations();
  const savedId = localStorage.getItem('llm_wiki_agent_current_conversation');
  if (savedId && conversations.find(c => c.id === savedId)) {
    await switchConversation(savedId);
  } else if (conversations.length > 0) {
    await switchConversation(conversations[0].id);
  } else {
    await createNewConversation();
  }

  // Load initial data
  loadToken();
  loadModels();
  setupEventListeners();
  loadFileList();
  loadIndexStats();
}

document.addEventListener('DOMContentLoaded', init);

// Conversation Management
async function createNewConversation() {
  try {
    const resp = await fetch(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await resp.json();
    currentConversationId = data.id;
    localStorage.setItem('llm_wiki_agent_current_conversation', currentConversationId);
    clearMessages();
    // Refresh list
    await loadConversations();
    renderConversationList();
    return data.id;
  } catch (err) {
    console.error('Failed to create conversation:', err);
  }
}

async function loadConversations() {
  try {
    const limit = conversationListExpanded ? 0 : CONVERSATION_PAGE_SIZE;
    const url = limit
      ? `${API_BASE}/conversations?limit=${limit}`
      : `${API_BASE}/conversations`;
    const resp = await fetch(url);
    const totalHeader = resp.headers.get('X-Total-Count');
    conversationTotalCount = totalHeader ? parseInt(totalHeader, 10) : 0;
    conversations = await resp.json();
    renderConversationList();
  } catch (err) {
    console.error('Failed to load conversations:', err);
    conversations = [];
    conversationTotalCount = 0;
  }
}

async function switchConversation(id) {
  currentConversationId = id;
  localStorage.setItem('llm_wiki_agent_current_conversation', id);
  await loadConversationMessages(id);
  renderConversationList();
}

async function deleteConversation(id) {
  try {
    await fetch(`${API_BASE}/conversations/${id}`, { method: 'DELETE' });
  } catch (err) {
    console.error('Failed to delete conversation:', err);
  }
  conversations = conversations.filter(c => c.id !== id);
  if (currentConversationId === id) {
    if (conversations.length > 0) {
      await switchConversation(conversations[0].id);
    } else {
      await createNewConversation();
    }
  } else {
    renderConversationList();
  }
}

async function loadConversationMessages(conversationId) {
  try {
    const resp = await fetch(`${API_BASE}/conversations/${conversationId}`);
    const data = await resp.json();
    clearMessages();
    for (const turn of data.turns) {
      if (turn.role === 'user') {
        appendUserMessage(turn.content);
      } else if (turn.role === 'assistant') {
        appendAgentMessage(turn.content);
      }
    }
    scrollToBottom();
  } catch (err) {
    console.error('Failed to load conversation:', err);
    clearMessages();
  }
}

function renderConversationList() {
  const listEl = document.getElementById('conversation-list');
  if (!listEl) return;
  const items = conversations.map(c => `
    <li class="conversation-item ${c.id === currentConversationId ? 'active' : ''}"
        onclick="switchConversation('${c.id}')">
      <span class="conversation-title">${escapeHtml(c.title || 'New Chat')}</span>
      <button class="conversation-delete" onclick="event.stopPropagation(); deleteConversation('${c.id}')" title="Delete">&times;</button>
    </li>
  `).join('');
  let footer = '';
  const showingTruncated =
    !conversationListExpanded &&
    conversationTotalCount > conversations.length;
  if (showingTruncated) {
    const more = conversationTotalCount - conversations.length;
    footer = `
      <li class="conversation-item conversation-show-more"
          onclick="toggleConversationListExpanded()">
        <span class="conversation-title">Show ${more} more...</span>
      </li>`;
  } else if (conversationListExpanded && conversationTotalCount > CONVERSATION_PAGE_SIZE) {
    footer = `
      <li class="conversation-item conversation-show-more"
          onclick="toggleConversationListExpanded()">
        <span class="conversation-title">Show fewer</span>
      </li>`;
  }
  listEl.innerHTML = items + footer;
}

async function toggleConversationListExpanded() {
  conversationListExpanded = !conversationListExpanded;
  await loadConversations();
}
window.toggleConversationListExpanded = toggleConversationListExpanded;

function clearMessages() {
  if (messagesContainer) {
    messagesContainer.innerHTML = '';
  }
}

// Token Management
function loadToken() {
  let token = localStorage.getItem('llm_wiki_agent_token');
  if (!token) {
    token = 'lwa_' + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('llm_wiki_agent_token', token);
  }
  connect(token);
}

function saveToken(token) {
  localStorage.setItem('llm_wiki_agent_token', token);
}

// Model Management
async function loadModels() {
  try {
    const resp = await fetch(`${API_BASE}/models`);
    const data = await resp.json();
    const { models, current } = data;
    // Populate datalist with available models
    const datalist = document.getElementById('model-list');
    if (datalist) {
      datalist.innerHTML = models.map(m =>
        `<option value="${m}">`
      ).join('');
    }
    // Set current model in the input
    if (modelSelect) {
      modelSelect.value = current || '';
    }
  } catch (err) {
    console.error('Failed to load models:', err);
    if (modelSelect) modelSelect.placeholder = 'Error loading models';
  }
}

function switchModel(model) {
  fetch(`${API_BASE}/model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  }).catch(err => console.error('Failed to switch model:', err));
}

// SSE Connection
function connect(token) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`${API_BASE}/sse?token=${token}`);

  eventSource.onopen = () => {
    console.log('SSE connected');
    statusDot.className = 'dot dot-connected';
    statusText.textContent = 'Connected';
    if (document.getElementById('input-bar')) {
      document.getElementById('input-bar').classList.remove('hidden');
    }
    if (ctxBar) {
      ctxBar.classList.remove('hidden');
    }
  };

  eventSource.onerror = (err) => {
    console.log('SSE error:', err);
    statusDot.className = 'dot dot-disconnected';
    statusText.textContent = 'Disconnected';
    if (document.getElementById('input-bar')) {
      document.getElementById('input-bar').classList.add('hidden');
    }
    if (ctxBar) {
      ctxBar.classList.add('hidden');
    }
  };

  eventSource.addEventListener('message', handleSSEMessage);
  eventSource.addEventListener('token_usage', handleTokenUsage);
}

function handleSSEMessage(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    appendAgentChunk(data.content);
  } else if (data.type === 'done') {
    finalizeAgentMessage();
  } else if (data.type === 'file_created' || data.type === 'file_updated') {
    loadFileList();
    loadIndexStats();
  }
}

function handleTokenUsage(event) {
  const data = JSON.parse(event.data);
  const used = data.used || 0;
  const total = data.total || 256000;
  const pct = (used / total) * 100;
  const ctxFill = document.getElementById('ctx-fill');
  if (ctxFill) {
    ctxFill.innerHTML = `<div class="fill" style="width:${pct}%"></div>`;
  }
  if (ctxLabel) {
    ctxLabel.textContent = `${used.toLocaleString()} / ${total.toLocaleString()} ctx`;
  }
}

// Chat with streaming
async function sendMessage() {
  const content = textInput.value.trim();
  if (!content) return;

  // Auto-create conversation if none exists
  if (!currentConversationId) {
    await createNewConversation();
    if (!currentConversationId) return;
  }

  appendUserMessage(content);
  textInput.value = '';
  textInput.style.height = 'auto';
  showActivity('Searching knowledge base...');

  fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: content, conversation_id: currentConversationId }),
  })
  .then(r => {
    if (!r.ok) throw new Error('Network error');
    return r;
  })
  .then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentMsg = null;

    function readStream() {
      reader.read().then(({ done, value }) => {
        if (done) {
          finalizeAgentMessage();
          hideActivity();
          // Refresh conversation list (title may have updated)
          loadConversations();
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = '';

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          if (line.startsWith('event:')) {
            const eventType = line.slice(6).trim();
            const nextLine = lines[i + 1];
            if (nextLine && nextLine.trim().startsWith('data:')) {
              const data = nextLine.trim().slice(5).replace(/^ /, '');
              if (eventType === 'token_usage') {
                handleTokenUsage({ data });
              } else {
                handleStreamEvent(eventType, data, () => {
                  if (!currentMsg) currentMsg = createAgentMessage();
                  return currentMsg;
                });
              }
            }
          }
        }
        readStream();
      });
    }
    readStream();
  })
  .catch(err => {
    console.error('Chat error:', err);
    hideActivity();
    appendAgentMessage('Error: Connection lost');
  });
}

// --- Phase 2: per-iteration message DOM ---
//
// Each tool-loop iteration gets its own .iteration block with three slots:
//   .iteration-thinking : reasoning tokens
//   .iteration-tools    : tool-call bubbles (PERSISTENT, never re-rendered)
//   .iteration-text     : streamed markdown (re-rendered every 150ms)
//
// Re-renders only target .iteration-text so tool bubbles never get wiped.

const KNOWN_TOOLS = [
  'list_knowledge', 'read_knowledge_section', 'read_knowledge',
  'search_knowledge', 'save_knowledge',
  'graph_neighbors', 'graph_traverse', 'graph_search', 'graph_stats',
];
const BRACKETED_TOOL_RE = /\[TOOL:\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\([^)]*\))?\s*\]/g;
const BARE_TOOL_RE = new RegExp(
  '^[ \\t]*(?:[-*\\u2022]|\\d+\\.)?[ \\t]*' +
  '`?(' + KNOWN_TOOLS.join('|') + ')\\(([^)]*)\\)`?\\s*$',
  'gm'
);

function stripToolMarkers(text) {
  if (!text) return text;
  return text.replace(BRACKETED_TOOL_RE, '').replace(BARE_TOOL_RE, '');
}

function ensureIteration(msg, iterationIndex) {
  const contentEl = msg.querySelector('.message-content');
  let iter = contentEl.querySelector(`.iteration[data-i="${iterationIndex}"]`);
  if (!iter) {
    iter = document.createElement('div');
    iter.className = 'iteration';
    iter.dataset.i = String(iterationIndex);
    iter.innerHTML = `
      <div class="iteration-thinking"></div>
      <div class="iteration-tools"></div>
      <div class="iteration-text"></div>
    `;
    contentEl.appendChild(iter);
  }
  return iter;
}

function getCurrentIteration(msg) {
  const contentEl = msg.querySelector('.message-content');
  // Default to iteration 0 if server didn't emit iteration_start (back-compat)
  const all = contentEl.querySelectorAll('.iteration');
  if (all.length === 0) return ensureIteration(msg, 0);
  return all[all.length - 1];
}

function setStreamingCursor(msg, on) {
  let cursor = msg.querySelector('.streaming-cursor');
  if (on) {
    if (!cursor) {
      cursor = document.createElement('span');
      cursor.className = 'streaming-cursor';
      cursor.textContent = '▌';
      msg.querySelector('.message-content').appendChild(cursor);
    }
  } else if (cursor) {
    cursor.remove();
  }
}

function setActivityPulse(msg, label) {
  let pulse = msg.querySelector('.activity-pulse');
  if (label) {
    if (!pulse) {
      pulse = document.createElement('div');
      pulse.className = 'activity-pulse';
      pulse.innerHTML = `<span class="activity-pulse-dots"><span></span><span></span><span></span></span> <span class="activity-pulse-label"></span>`;
      msg.querySelector('.message-content').appendChild(pulse);
    }
    pulse.querySelector('.activity-pulse-label').textContent = label;
  } else if (pulse) {
    pulse.remove();
  }
}

function noteActivity(msg) {
  msg._lastTokenAt = Date.now();
  setActivityPulse(msg, null);
  if (msg._idleWatcher) {
    clearTimeout(msg._idleWatcher);
  }
  msg._idleWatcher = setTimeout(() => {
    if (Date.now() - (msg._lastTokenAt || 0) >= 2000 && msg.classList.contains('streaming')) {
      setActivityPulse(msg, 'Thinking…');
    }
  }, 2200);
}

function startToolTimer(bubble) {
  if (bubble._timerInterval) {
    clearInterval(bubble._timerInterval);
    bubble._timerInterval = null;
  }
  const startedAt = Date.now();
  bubble._timerInterval = setInterval(() => {
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    const elapsedEl = bubble.querySelector('.tool-bubble-elapsed');
    if (elapsedEl) elapsedEl.textContent = ` · ${elapsed}s…`;
  }, 500);
}

function stopToolTimer(bubble, elapsedMs) {
  if (bubble._timerInterval) {
    clearInterval(bubble._timerInterval);
    bubble._timerInterval = null;
  }
  const elapsedEl = bubble.querySelector('.tool-bubble-elapsed');
  if (elapsedEl) {
    if (typeof elapsedMs === 'number') {
      const secs = (elapsedMs / 1000).toFixed(elapsedMs < 1000 ? 2 : 1);
      elapsedEl.textContent = ` · ${secs}s`;
    } else {
      elapsedEl.textContent = '';
    }
  }
}

/** Last running bubble for this tool in the iteration (tool_done matches the same). */
function findLastRunningToolBubble(iter, toolName) {
  const all = iter.querySelectorAll('.tool-call-bubble');
  let found = null;
  for (const b of all) {
    if (b.dataset.tool === toolName && b.classList.contains('running')) {
      found = b;
    }
  }
  return found;
}

/**
 * Compact tool args for the bubble header — never dump full save_knowledge body.
 */
function formatToolArgsForDisplay(tool, args) {
  if (!args || typeof args !== 'object') {
    return '';
  }
  const MAX_INLINE = 380;
  const parts = [];
  for (const [k, v] of Object.entries(args)) {
    let s = v === null || v === undefined ? '' : String(v);
    if (tool === 'save_knowledge' && k === 'content') {
      const n = s.length;
      if (n <= 200) {
        parts.push(`content: "${escapeHtml(s)}"`);
      } else {
        const preview = s.slice(0, 120).replace(/\s+/g, ' ').trim();
        parts.push(`content: <${n.toLocaleString()} chars> "${escapeHtml(preview)}…"`);
      }
      continue;
    }
    if (s.length > MAX_INLINE) {
      const preview = escapeHtml(s.slice(0, 120).replace(/\n/g, ' '));
      parts.push(`${k}: <${s.length.toLocaleString()} chars> "${preview}…"`);
    } else {
      parts.push(`${k}: ${escapeHtml(s)}`);
    }
  }
  return parts.join(', ');
}

function handleStreamEvent(type, data, getMsg) {
  const msg = getMsg();

  switch (type) {
    case 'kb_context':
      showActivity(`Found ${data} in knowledge base`);
      break;
    case 'kb_file':
      try {
        const file = JSON.parse(data);
        appendKBResult(file.path, file.score);
      } catch (e) {}
      break;

    case 'iteration_start': {
      let iterIdx = 0;
      try {
        const info = JSON.parse(data);
        iterIdx = info.iteration ?? 0;
      } catch (e) {}
      ensureIteration(msg, iterIdx);
      noteActivity(msg);
      break;
    }

    case 'thinking': {
      const decoded = decodeURIComponent(data);
      const iter = getCurrentIteration(msg);
      let thinkingBubble = iter.querySelector('.iteration-thinking .thinking-bubble');
      if (!thinkingBubble) {
        thinkingBubble = document.createElement('div');
        thinkingBubble.className = 'thinking-bubble';
        thinkingBubble.innerHTML = `
          <div class="thinking-bubble-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="thinking-bubble-title">
              <svg class="thinking-bubble-toggle" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 11L3 6h10l-5 5z"/>
              </svg>
              Thinking
            </span>
          </div>
          <div class="thinking-bubble-content"></div>
        `;
        iter.querySelector('.iteration-thinking').appendChild(thinkingBubble);
      }
      thinkingBubble.querySelector('.thinking-bubble-content').textContent += decoded;
      noteActivity(msg);
      scrollToBottom();
      break;
    }

    case 'token': {
      const decodedToken = decodeURIComponent(data);
      const iter = getCurrentIteration(msg);
      const textEl = iter.querySelector('.iteration-text');
      // Strip raw [TOOL: ...] / bare tool_name(args) from the visible stream;
      // server-side parsing still acts on them via the tool_call SSE event.
      const cleaned = stripToolMarkers(decodedToken);
      textEl._rawText = (textEl._rawText || '') + cleaned;
      if (!textEl._renderTimer) {
        textEl._renderTimer = setTimeout(() => {
          // IMPORTANT: re-render targets only the text slot, not the whole
          // message-content. Tool bubbles in .iteration-tools survive.
          const fullClean = stripToolMarkers(textEl._rawText);
          textEl.innerHTML = marked.parse(fullClean);
          textEl._renderTimer = null;
          setStreamingCursor(msg, true);
          scrollToBottom();
        }, 150);
      }
      noteActivity(msg);
      break;
    }

    case 'error': {
      const iter = getCurrentIteration(msg);
      const textEl = iter.querySelector('.iteration-text');
      textEl._rawText = (textEl._rawText || '') + `\n[Error: ${data}]`;
      textEl.innerHTML = marked.parse(textEl._rawText);
      break;
    }

    case 'tool_call': {
      try {
        const toolInfo = JSON.parse(data);
        const iter = getCurrentIteration(msg);
        const argsStr = formatToolArgsForDisplay(toolInfo.tool, toolInfo.args);
        const bubble = document.createElement('div');
        bubble.className = 'tool-call-bubble running';
        bubble.dataset.tool = toolInfo.tool;
        bubble.innerHTML = `
          <div class="tool-bubble-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="tool-bubble-title">
              <svg class="tool-bubble-toggle" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 11L3 6h10l-5 5z"/>
              </svg>
              Tool: ${escapeHtml(toolInfo.tool)}(${argsStr})<span class="tool-bubble-elapsed"> · 0s…</span>
            </span>
          </div>
          <div class="tool-bubble-content"></div>
        `;
        iter.querySelector('.iteration-tools').appendChild(bubble);
        startToolTimer(bubble);
        noteActivity(msg);
        scrollToBottom();
      } catch (e) {}
      break;
    }

    case 'tool_executing': {
      try {
        const toolInfo = JSON.parse(data);
        const iter = getCurrentIteration(msg);
        let bubble = findLastRunningToolBubble(iter, toolInfo.tool);
        if (!bubble) {
          const argsStr = formatToolArgsForDisplay(toolInfo.tool, toolInfo.args);
          bubble = document.createElement('div');
          bubble.className = 'tool-call-bubble running';
          bubble.dataset.tool = toolInfo.tool;
          bubble.innerHTML = `
          <div class="tool-bubble-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="tool-bubble-title">
              <svg class="tool-bubble-toggle" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 11L3 6h10l-5 5z"/>
              </svg>
              Tool: ${escapeHtml(toolInfo.tool)}(${argsStr})<span class="tool-bubble-elapsed"> · 0s…</span>
            </span>
          </div>
          <div class="tool-bubble-content"></div>
        `;
          iter.querySelector('.iteration-tools').appendChild(bubble);
        }
        startToolTimer(bubble);
        noteActivity(msg);
        scrollToBottom();
      } catch (e) {}
      break;
    }

    case 'tool_done': {
      try {
        const info = JSON.parse(data);
        const iter = getCurrentIteration(msg);
        const bubbles = iter.querySelectorAll(`.tool-call-bubble[data-tool="${info.tool}"].running`);
        const bubble = bubbles[bubbles.length - 1];
        if (bubble) {
          stopToolTimer(bubble, info.elapsed_ms);
          bubble.classList.remove('running');
        }
      } catch (e) {}
      break;
    }

    case 'tool_result': {
      try {
        const resultInfo = JSON.parse(data);
        const iter = getCurrentIteration(msg);
        // Match the most recent bubble for this tool (running or just-stopped)
        const matching = iter.querySelectorAll(`.tool-call-bubble[data-tool="${resultInfo.tool}"]`);
        const lastToolBubble = matching[matching.length - 1] ||
                               iter.querySelector('.tool-call-bubble:last-of-type');
        if (lastToolBubble) {
          stopToolTimer(lastToolBubble);
          lastToolBubble.classList.remove('running');
          const executed = resultInfo.executed !== false;
          if (!executed) {
            lastToolBubble.classList.add('tool-not-executed');
          }
          const resultEl = lastToolBubble.querySelector('.tool-bubble-content');
          // Phase 1 servers emit { result, info }; older servers emit { result_preview }.
          const fullResult = resultInfo.result || resultInfo.result_preview || '';
          if (resultEl) {
            resultEl.textContent = fullResult;
            // Auto-collapse very large results to keep the chat readable.
            if (fullResult.length > 2000) {
              lastToolBubble.classList.add('collapsed');
              const titleEl = lastToolBubble.querySelector('.tool-bubble-title');
              if (titleEl && resultInfo.info) {
                const meta = document.createElement('span');
                meta.className = 'tool-bubble-meta';
                meta.textContent = ` · ${resultInfo.info.delivered_chars.toLocaleString()} chars` +
                  (resultInfo.info.truncated ? ' (truncated)' : '');
                titleEl.appendChild(meta);
              }
            }
          }
          lastToolBubble.classList.add('has-result');
        }
        noteActivity(msg);
        scrollToBottom();
      } catch (e) {}
      break;
    }

    case 'heartbeat': {
      try {
        const info = JSON.parse(data);
        if (info.label) {
          setActivityPulse(msg, info.label);
        }
      } catch (e) {
        setActivityPulse(msg, 'Working…');
      }
      break;
    }

    case 'done':
      finalizeAgentMessage();
      break;
  }
}

function createAgentMessage() {
  const div = document.createElement('div');
  div.className = 'message agent streaming';
  div.innerHTML = `
    <div class="message-role">R</div>
    <div class="message-content"></div>
    <div class="message-actions">
      <button class="message-action-btn" title="Copy" onclick="copyMessage(this)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    </div>
  `;
  messagesContainer.appendChild(div);
  scrollToBottom();
  return div;
}

function appendKBResult(path, score) {
  showActivity(`KB: ${path.split('/').pop()} (score: ${score})`);
}

function appendUserMessage(content) {
  const div = document.createElement('div');
  div.className = 'message user';
  const contentEl = document.createElement('div');
  contentEl.className = 'message-content';
  contentEl.textContent = content;
  div.innerHTML = `
    <div class="message-role">You</div>
    <div class="message-actions">
      <button class="message-action-btn" title="Copy" onclick="copyMessage(this)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    </div>
  `;
  div.insertBefore(contentEl, div.querySelector('.message-actions'));
  messagesContainer.appendChild(div);
  scrollToBottom();
}

function appendAgentChunk(content) {
  let lastMsg = messagesContainer.lastElementChild;
  if (!lastMsg || !lastMsg.classList.contains('agent')) {
    const div = document.createElement('div');
    div.className = 'message agent streaming';
    div.innerHTML = `
      <div class="message-role">R</div>
      <div class="message-content"></div>
      <div class="message-actions">
        <button class="message-action-btn" title="Copy" onclick="copyMessage(this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2 2v1"></path>
          </svg>
        </button>
      </div>
    `;
    messagesContainer.appendChild(div);
    lastMsg = div;
  }
  const contentEl = lastMsg.querySelector('.message-content');
  contentEl._rawText = (contentEl._rawText || '') + content;
  if (!contentEl._renderTimer) {
    contentEl._renderTimer = setTimeout(() => {
      contentEl.innerHTML = marked.parse(contentEl._rawText) + '<span class="streaming-cursor">▌</span>';
      contentEl._renderTimer = null;
      scrollToBottom();
    }, 150);
  }
}

async function finalizeAgentMessage() {
  const lastMsg = messagesContainer.lastElementChild;
  if (lastMsg && lastMsg.classList.contains('agent')) {
    lastMsg.classList.remove('streaming');
    if (lastMsg._idleWatcher) {
      clearTimeout(lastMsg._idleWatcher);
      lastMsg._idleWatcher = null;
    }
    setStreamingCursor(lastMsg, false);
    setActivityPulse(lastMsg, null);
    if (!lastMsg.querySelector('.message-actions')) {
      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'message-actions';
      actionsDiv.innerHTML = `
        <button class="message-action-btn" title="Copy" onclick="copyMessage(this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2 2v1"></path>
          </svg>
        </button>
      `;
      lastMsg.appendChild(actionsDiv);
    }
    // Collapse tool bubbles into a compact summary strip per iteration.
    const iterations = lastMsg.querySelectorAll('.iteration');
    iterations.forEach((iter) => {
      const toolBubbles = iter.querySelectorAll('.tool-call-bubble');
      if (toolBubbles.length > 0) {
        const toolContainer = iter.querySelector('.iteration-tools');
        if (toolContainer && !toolContainer.querySelector('.tool-summary-strip')) {
          const toolNames = [];
          toolBubbles.forEach(b => {
            b.classList.add('collapsed');
            const name = b.dataset.tool || 'tool';
            if (!toolNames.includes(name)) toolNames.push(name);
          });
          const strip = document.createElement('div');
          strip.className = 'tool-summary-strip';
          const label = toolBubbles.length === 1
            ? `1 tool call · ${toolNames[0]}`
            : `${toolBubbles.length} tool calls · ${toolNames.join(', ')}`;
          strip.innerHTML = `<span class="tool-strip-label">${escapeHtml(label)}</span><span class="tool-strip-chevron">▸</span>`;
          strip.addEventListener('click', () => {
            const expanded = toolContainer.classList.toggle('tools-expanded');
            strip.querySelector('.tool-strip-chevron').textContent = expanded ? '▾' : '▸';
          });
          toolContainer.classList.add('tools-collapsed');
          toolContainer.insertBefore(strip, toolContainer.firstChild);
        }
      }
      const textEl = iter.querySelector('.iteration-text');
      if (textEl && textEl._rawText) {
        if (textEl._renderTimer) {
          clearTimeout(textEl._renderTimer);
          textEl._renderTimer = null;
        }
        textEl.innerHTML = marked.parse(stripToolMarkers(textEl._rawText));
        delete textEl._rawText;
      }
    });
    // Back-compat: if the message has no iterations (e.g. legacy server)
    // but has buffered raw text on the content element, render it.
    const contentEl = lastMsg.querySelector('.message-content');
    if (contentEl && contentEl._rawText && iterations.length === 0) {
      if (contentEl._renderTimer) {
        clearTimeout(contentEl._renderTimer);
        contentEl._renderTimer = null;
      }
      contentEl.innerHTML = marked.parse(stripToolMarkers(contentEl._rawText));
      delete contentEl._rawText;
    }
    await renderMermaidDiagrams(lastMsg);
  }
}

async function renderMermaidDiagrams(container) {
  if (!mermaidModule) return;
  const mermaidElements = container.querySelectorAll('pre.mermaid, code.language-mermaid');
  if (mermaidElements.length === 0) return;
  try {
    await mermaidModule.run({
      nodes: Array.from(mermaidElements),
      suppressErrors: true,
    });
  } catch (e) {
    console.error('Mermaid render error:', e);
  }
}

function appendAgentMessage(content) {
  const div = document.createElement('div');
  div.className = 'message agent';
  div.innerHTML = `
    <div class="message-role">R</div>
    <div class="message-content">${marked.parse(content)}</div>
    <div class="message-actions">
      <button class="message-action-btn" title="Copy" onclick="copyMessage(this)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    </div>
  `;
  messagesContainer.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  if (messagesContainer) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

// File Browser
function loadFileList() {
  fetch(`${API_BASE}/kb/${currentFolder}`)
    .then(r => r.json())
    .then(files => {
      files.sort((a, b) => {
        const folderA = a.folder || '';
        const folderB = b.folder || '';
        if (folderA !== folderB) return folderA.localeCompare(folderB);
        return a.name.localeCompare(b.name);
      });
      fileList.innerHTML = files.map(f => {
        const indent = f.folder && f.folder !== '.' ? '  ' : '';
        const folderDisplay = f.folder && f.folder !== '.' ? `<span class="file-folder">${f.folder}/</span>` : '';
        return `
          <li data-path="${f.path}" onclick="loadFile('${f.path}')">
            ${indent}${folderDisplay}${f.name}
            ${f.last_modified_by ? `<span class="meta">[${f.last_modified_by}]</span>` : ''}
          </li>
        `;
      }).join('');
    });
}

function loadFile(path) {
  fetch(`${API_BASE}/kb/file/${encodeURIComponent(path)}`)
    .then(r => r.text())
    .then(content => {
      const win = window.open('', '_blank');
      win.document.write(`<pre style="white-space:pre-wrap;font-family:monospace;">${escapeHtml(content)}</pre>`);
    });
}

// Search
let searchDebounce = null;

function doSearch(query) {
  if (!query) {
    searchResults.innerHTML = '';
    return;
  }
  fetch(`${API_BASE}/kb/search?q=${encodeURIComponent(query)}`)
    .then(r => r.json())
    .then(results => {
      searchResults.innerHTML = results.slice(0, 5).map(r => `
        <div class="search-result" onclick="loadFile('${r.path}')">
          <div class="search-result-title">${r.path}</div>
          <div class="search-result-snippet">${escapeHtml(r.snippet)}</div>
        </div>
      `).join('');
    });
}

// Index Stats
function loadIndexStats() {
  fetch(`${API_BASE}/kb/stats`)
    .then(r => r.json())
    .then(stats => {
      const fileCount = document.getElementById('file-count');
      const vectorCount = document.getElementById('vector-count');
      if (fileCount) fileCount.textContent = stats.files || 0;
      if (vectorCount) vectorCount.textContent = stats.vectors || 0;
    });
}

// Event Listeners
function setupEventListeners() {
  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (textInput) {
    textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    textInput.addEventListener('input', () => {
      textInput.style.height = 'auto';
      textInput.style.height = textInput.scrollHeight + 'px';
    });
  }

  if (searchInput) searchInput.addEventListener('input', (e) => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => doSearch(e.target.value), 300);
  });

  if (reindexBtn) reindexBtn.addEventListener('click', () => {
    const summaries = document.getElementById('reindex-summaries')?.checked || false;
    const entities = document.getElementById('reindex-entities')?.checked || false;
    const body = { summaries, entities };
    reindexBtn.disabled = true;
    reindexBtn.textContent = 'Rebuilding...';
    fetch(`${API_BASE}/kb/reindex`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
      .then(r => r.json())
      .then(data => {
        loadIndexStats();
        loadFileList();
        const parts = ['Index rebuilt'];
        if (summaries) parts.push('with LLM summaries');
        if (entities) parts.push('with entities');
        alert(parts.join(' '));
      })
      .catch(err => alert('Reindex failed: ' + err.message))
      .finally(() => {
        reindexBtn.disabled = false;
        reindexBtn.textContent = 'Rebuild Index';
      });
  });

  if (modelSelect) modelSelect.addEventListener('change', (e) => {
    const model = e.target.value.trim();
    if (model) switchModel(model);
  });
  if (modelSelect) modelSelect.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const model = e.target.value.trim();
      if (model) switchModel(model);
    }
  });

  document.querySelectorAll('.kb-tabs .tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.kb-tabs .tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      currentFolder = e.target.dataset.folder;
      loadFileList();
    });
  });

  if (sidebarToggle && sidebar && mainLayout) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      mainLayout.classList.toggle('sidebar-collapsed');
      localStorage.setItem('sidebar_collapsed', sidebar.classList.contains('collapsed'));
    });
  }
}

// Activity Indicator
function showActivity(text) {
  let activity = document.getElementById('activity-bar');
  if (!activity) {
    activity = document.createElement('div');
    activity.id = 'activity-bar';
    activity.className = 'hidden';
    activity.innerHTML = `<span id="activity-text"></span>`;
    const ctxBar = document.getElementById('ctx-bar');
    if (ctxBar && ctxBar.parentNode) {
      ctxBar.parentNode.insertBefore(activity, ctxBar.nextSibling);
    }
  }
  activity.classList.remove('hidden');
  const activityText = document.getElementById('activity-text');
  if (activityText) activityText.textContent = text;
}

function hideActivity() {
  const activity = document.getElementById('activity-bar');
  if (activity) activity.classList.add('hidden');
}

// Utilities
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function copyMessage(btn) {
  const message = btn.closest('.message');
  const content = message.querySelector('.message-content').innerText;
  navigator.clipboard.writeText(content).then(() => {
    const originalIcon = btn.innerHTML;
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    setTimeout(() => {
      btn.innerHTML = originalIcon;
    }, 1500);
  });
}