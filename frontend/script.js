/* ═══════════════════════════════════════════════════════
   LexAI — JavaScript Controller
   Handles API communication, chat UI, and interactions
   ═══════════════════════════════════════════════════════ */

const API_BASE = "http://127.0.0.1:8000";

// ── DOM Elements ──────────────────────────────────────
const elements = {
    statusIndicator: document.getElementById("statusIndicator"),
    statusText: document.querySelector(".status-text"),
    welcomeScreen: document.getElementById("welcomeScreen"),
    chatContainer: document.getElementById("chatContainer"),
    queryForm: document.getElementById("queryForm"),
    questionInput: document.getElementById("questionInput"),
    btnSend: document.getElementById("btnSend"),
    btnClearChat: document.getElementById("btnClearChat"),
    charCount: document.getElementById("charCount"),
    statChunks: document.getElementById("statChunks"),
    statModel: document.getElementById("statModel"),
    main: document.getElementById("main"),
};

// ── State ─────────────────────────────────────────────
let isLoading = false;
let messageCount = 0;
let conversationHistory = [];  // Track conversation for continuity

// ── Initialize ────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setupEventListeners();
    elements.questionInput.focus();

    // Health check every 30 seconds
    setInterval(checkHealth, 30000);
});

// ── Event Listeners ───────────────────────────────────
function setupEventListeners() {
    // Form submission
    elements.queryForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleQuery();
    });

    // Textarea auto-resize + char count
    elements.questionInput.addEventListener("input", () => {
        autoResizeTextarea();
        updateCharCount();
    });

    // Enter to send (Shift+Enter for newline)
    elements.questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });

    // Clear chat
    elements.btnClearChat.addEventListener("click", clearChat);

    // Suggestion cards
    document.querySelectorAll(".suggestion-card").forEach((card) => {
        card.addEventListener("click", () => {
            const question = card.getAttribute("data-question");
            elements.questionInput.value = question;
            autoResizeTextarea();
            updateCharCount();
            handleQuery();
        });
    });
}

// ── Health Check ──────────────────────────────────────
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            signal: AbortSignal.timeout(5000),
        });
        const data = await response.json();

        elements.statusIndicator.className = "status-indicator online";
        elements.statusText.textContent = `Online · ${formatNumber(data.chunks_indexed)} chunks`;

        // Update stats
        elements.statChunks.textContent = formatNumber(data.chunks_indexed);
        elements.statModel.textContent = data.model || "LLaMA 3.1";
    } catch (error) {
        elements.statusIndicator.className = "status-indicator offline";
        elements.statusText.textContent = "Offline";
    }
}

// ── Query Handler ─────────────────────────────────────
async function handleQuery() {
    const question = elements.questionInput.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    elements.btnSend.disabled = true;

    // Hide welcome, show chat
    if (elements.welcomeScreen) {
        elements.welcomeScreen.classList.add("hidden");
    }

    // Add user message
    addMessage("user", question);

    // Clear input
    elements.questionInput.value = "";
    autoResizeTextarea();
    updateCharCount();

    // Show loading
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                question, 
                top_k: 3,
                history: conversationHistory.slice(-6)  // Send last 6 messages
            }),
        });

        // Remove loading
        removeMessage(loadingId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();

        // Store in conversation history
        conversationHistory.push({ role: "user", content: question });
        conversationHistory.push({ role: "assistant", content: data.answer });

        addMessage("assistant", data.answer, {
            sources: data.sources,
            responseTime: data.response_time,
        });
    } catch (error) {
        removeMessage(loadingId);
        addErrorMessage(error.message || "Failed to get response. Is the API running?");
    } finally {
        isLoading = false;
        elements.btnSend.disabled = false;
        elements.questionInput.focus();
    }
}

// ── Message Rendering ─────────────────────────────────
function addMessage(role, content, meta = {}) {
    messageCount++;
    const id = `msg-${messageCount}`;
    const time = new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
    });

    const isUser = role === "user";

    let metaHtml = "";
    if (!isUser && meta.sources) {
        metaHtml = `
            <div class="response-meta">
                <span class="response-time">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                    ${meta.responseTime}s
                </span>
            </div>
            <div class="sources-section">
                <div class="sources-label">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Sources
                </div>
                <div class="source-tags">
                    ${meta.sources.map((s) => `<span class="source-tag">${s}</span>`).join("")}
                </div>
            </div>
        `;
    }

    const html = `
        <div class="message message-${role}" id="${id}">
            <div class="message-avatar">
                ${isUser ? "You" : "AI"}
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-name">${isUser ? "You" : "LexAI"}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-body">
                    ${formatAnswer(content)}
                    ${metaHtml}
                </div>
            </div>
        </div>
    `;

    elements.chatContainer.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
    return id;
}

function addLoadingMessage() {
    messageCount++;
    const id = `msg-${messageCount}`;

    const html = `
        <div class="message message-assistant" id="${id}">
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-name">LexAI</span>
                    <span class="message-time">thinking...</span>
                </div>
                <div class="message-body">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            </div>
        </div>
    `;

    elements.chatContainer.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
    return id;
}

function addErrorMessage(message) {
    messageCount++;
    const id = `msg-${messageCount}`;

    const html = `
        <div class="message message-assistant" id="${id}">
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="message-body">
                    <div class="error-message">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="15" y1="9" x2="9" y2="15"/>
                            <line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                        ${escapeHtml(message)}
                    </div>
                </div>
            </div>
        </div>
    `;

    elements.chatContainer.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) {
        el.style.opacity = "0";
        el.style.transform = "translateY(-8px)";
        el.style.transition = "all 0.2s ease";
        setTimeout(() => el.remove(), 200);
    }
}

// ── Format Answer ─────────────────────────────────────
function formatAnswer(text) {
    // Escape HTML first
    let formatted = escapeHtml(text);

    // Bold: **text**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Italic: *text*
    formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Line breaks
    formatted = formatted.replace(/\n/g, "<br>");

    // Numbered lists: 1. item → formatted list
    formatted = formatted.replace(
        /(?:^|\<br\>)(\d+)\.\s+(.*?)(?=\<br\>\d+\.|\<br\>\<br\>|$)/g,
        '<br><span style="color: var(--accent-gold); font-weight: 600;">$1.</span> $2'
    );

    // Bullet points: - item
    formatted = formatted.replace(
        /(?:^|\<br\>)[-•]\s+(.*?)(?=\<br\>[-•]|\<br\>\<br\>|$)/g,
        '<br><span style="color: var(--accent-gold);">▸</span> $1'
    );

    return formatted;
}

// ── Utilities ─────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    }
    return num.toString();
}

function autoResizeTextarea() {
    const textarea = elements.questionInput;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

function updateCharCount() {
    const count = elements.questionInput.value.length;
    elements.charCount.textContent = `${count}/1000`;
    elements.charCount.style.color = count > 900 ? "#f87171" : "";
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: "smooth",
        });
    });
}

function clearChat() {
    elements.chatContainer.innerHTML = "";
    elements.welcomeScreen.classList.remove("hidden");
    messageCount = 0;
    conversationHistory = [];  // Clear history too
    elements.questionInput.focus();
}
