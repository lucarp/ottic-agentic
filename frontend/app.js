/**
 * WebSocket client with specialized artifact renderers
 */

// State management
const state = {
    ws: null,
    artifacts: [],
    activeArtifactId: null,
    isConnected: false,
    charts: {}  // Store Chart.js instances
};

// DOM elements
const terminalMessages = document.getElementById('terminalMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const connectionDot = document.getElementById('connectionDot');
const connectionStatus = document.getElementById('connectionStatus');
const artifactTabs = document.getElementById('artifactTabs');
const artifactContent = document.getElementById('artifactContent');

// ============================================================================
// Artifact Renderers - Each type has its own renderer
// ============================================================================

const ArtifactRenderers = {
    csv: (data) => {
        const { headers, rows, title, description } = data;

        let html = '';
        if (title) html += `<h3 style="margin-bottom: 8px; color: #fff;">${title}</h3>`;
        if (description) html += `<p style="margin-bottom: 16px; color: #94a3b8;">${description}</p>`;

        html += `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: #1a1a1a;">
                    <thead>
                        <tr style="background: #262626;">
                            ${headers.map(h => `<th style="padding: 12px; text-align: left; border-bottom: 2px solid #333; color: #fff; font-weight: 600;">${h}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map(row => `
                            <tr style="border-bottom: 1px solid #333;">
                                ${row.map(cell => `<td style="padding: 10px; color: #e0e0e0;">${cell}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        return html;
    },

    html: (data) => {
        const { html, title, css } = data;

        let renderedHtml = '';
        if (title) renderedHtml += `<h3 style="margin-bottom: 16px; color: #fff;">${title}</h3>`;
        if (css) renderedHtml += `<style>${css}</style>`;
        renderedHtml += `<div style="background: #fff; padding: 20px; border-radius: 8px; color: #000;">${html}</div>`;

        return renderedHtml;
    },

    chart: (data, containerId) => {
        const { chart_type, labels, datasets, title, x_axis_label, y_axis_label } = data;

        let html = '';
        if (title) html += `<h3 style="margin-bottom: 16px; color: #fff;">${title}</h3>`;

        const canvasId = `chart-${containerId}`;
        html += `<canvas id="${canvasId}" style="max-height: 400px;"></canvas>`;

        // Render chart after DOM update
        setTimeout(() => {
            const ctx = document.getElementById(canvasId);
            if (ctx && !state.charts[canvasId]) {
                state.charts[canvasId] = new Chart(ctx, {
                    type: chart_type,
                    data: { labels, datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {
                            legend: {
                                labels: { color: '#e0e0e0' }
                            }
                        },
                        scales: chart_type !== 'pie' && chart_type !== 'doughnut' ? {
                            x: {
                                title: { display: !!x_axis_label, text: x_axis_label, color: '#e0e0e0' },
                                ticks: { color: '#94a3b8' },
                                grid: { color: '#333' }
                            },
                            y: {
                                title: { display: !!y_axis_label, text: y_axis_label, color: '#e0e0e0' },
                                ticks: { color: '#94a3b8' },
                                grid: { color: '#333' }
                            }
                        } : {}
                    }
                });
            }
        }, 100);

        return html;
    },

    payment_link: (data) => {
        const { amount, currency, description, success_message } = data;

        return `
            <div style="max-width: 400px; margin: 0 auto; text-align: center;">
                <h3 style="color: #fff; margin-bottom: 16px;">Payment Request</h3>
                <div style="background: #262626; padding: 24px; border-radius: 8px; border: 1px solid #333;">
                    <p style="color: #94a3b8; margin-bottom: 16px;">${description}</p>
                    <div style="font-size: 48px; font-weight: bold; color: #10b981; margin: 24px 0;">
                        $${amount.toFixed(2)}
                    </div>
                    <p style="color: #64748b; font-size: 14px; margin-bottom: 24px;">${currency.toUpperCase()}</p>
                    <button style="
                        width: 100%;
                        padding: 16px;
                        background: #5469d4;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: 600;
                        cursor: pointer;
                    " onclick="alert('Payment integration not implemented in POC')">
                        Pay with Stripe
                    </button>
                    ${success_message ? `<p style="color: #64748b; font-size: 12px; margin-top: 16px;">${success_message}</p>` : ''}
                </div>
            </div>
        `;
    },

    markdown: (data) => {
        const { content, title } = data;

        let html = '';
        if (title) html += `<h2 style="margin-bottom: 16px; color: #fff;">${title}</h2>`;
        html += `<div style="color: #e0e0e0; line-height: 1.8;">${marked.parse(content)}</div>`;

        return html;
    },

    code: (data) => {
        const { code, language, title, description } = data;

        let html = '';
        if (title) html += `<h3 style="margin-bottom: 8px; color: #fff;">${title}</h3>`;
        if (description) html += `<p style="margin-bottom: 16px; color: #94a3b8;">${description}</p>`;

        const highlighted = hljs.highlight(code, { language }).value;
        html += `<pre style="background: #0a0a0a; padding: 16px; border-radius: 6px; overflow-x: auto; border: 1px solid #333;"><code class="hljs language-${language}">${highlighted}</code></pre>`;

        return html;
    },

    fetched_link: (data) => {
        const { url, title, content, content_type, fetch_timestamp, metadata, summary } = data;

        let html = '';

        // Header with URL and metadata
        html += `<div style="border-bottom: 2px solid #333; padding-bottom: 16px; margin-bottom: 20px;">`;
        html += `<h2 style="color: #fff; margin-bottom: 8px;">${title || 'Fetched Content'}</h2>`;
        html += `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: #3b82f6; text-decoration: none; font-size: 14px; word-break: break-all;">
            🔗 ${url}
        </a>`;

        if (metadata) {
            html += `<div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #94a3b8;">`;
            if (metadata.author) {
                html += `<span>✍️ ${metadata.author}</span>`;
            }
            if (metadata.published_date) {
                const date = new Date(metadata.published_date).toLocaleDateString();
                html += `<span>📅 ${date}</span>`;
            }
            if (fetch_timestamp) {
                const fetchDate = new Date(fetch_timestamp).toLocaleString();
                html += `<span>⏰ Fetched: ${fetchDate}</span>`;
            }
            html += `</div>`;
        }
        html += `</div>`;

        // Summary if available
        if (summary) {
            html += `<div style="background: #1a2530; border-left: 4px solid #3b82f6; padding: 16px; margin-bottom: 20px; border-radius: 4px;">`;
            html += `<h4 style="color: #3b82f6; margin-bottom: 8px; font-size: 14px; text-transform: uppercase;">Summary</h4>`;
            html += `<p style="color: #e0e0e0; line-height: 1.6; margin: 0;">${summary}</p>`;
            html += `</div>`;
        }

        // Main content
        html += `<div style="color: #e0e0e0; line-height: 1.8; font-size: 15px;">`;
        if (content_type === 'markdown') {
            html += marked.parse(content);
        } else {
            // Plain text - preserve line breaks
            const formattedContent = content.replace(/\n/g, '<br>');
            html += formattedContent;
        }
        html += `</div>`;

        return html;
    }
};

// ============================================================================
// WebSocket and Terminal Functions
// ============================================================================

function connectWebSocket() {
    const wsUrl = 'ws://localhost:8000/ws';
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('WebSocket connected');
        state.isConnected = true;
        updateConnectionStatus(true);
        sendButton.disabled = false;
    };

    state.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };

    state.ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.isConnected = false;
        updateConnectionStatus(false);
        sendButton.disabled = true;
        setTimeout(connectWebSocket, 3000);
    };
}

function updateConnectionStatus(connected) {
    if (connected) {
        connectionDot.classList.remove('disconnected');
        connectionStatus.textContent = 'Connected';
    } else {
        connectionDot.classList.add('disconnected');
        connectionStatus.textContent = 'Disconnected';
    }
}

function handleMessage(message) {
    console.log('Received message:', message);

    switch (message.type) {
        case 'user_message':
            addUserMessage(message.content);
            break;
        case 'reasoning':
            addReasoningMessage(message.content);
            break;
        case 'assistant_message':
            addAssistantMessage(message.content);
            break;
        case 'tool_execution':
            addToolExecutionMessage(message);
            break;
        case 'artifact_created':
            addArtifact(message.artifact);
            break;
        case 'error':
            addErrorMessage(message.error);
            break;
    }
}

function addUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.textContent = content;
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addAssistantMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.textContent = content;
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addReasoningMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message reasoning-message';
    messageDiv.textContent = `💭 Thinking: ${content}`;
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addToolExecutionMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message tool-execution ${message.status}`;

    let content = `🔧 ${message.tool_name} - ${message.status}`;
    if (message.status === 'completed' && message.output) {
        content += '\n' + JSON.stringify(message.output, null, 2);
    }

    messageDiv.textContent = content;
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addErrorMessage(error) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error-message';
    messageDiv.textContent = `❌ Error: ${error}`;
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    terminalMessages.scrollTop = terminalMessages.scrollHeight;
}

// ============================================================================
// Artifact Management
// ============================================================================

function addArtifact(artifact) {
    console.log('Adding artifact:', artifact);

    state.artifacts.push(artifact);

    // Create tab
    const tab = document.createElement('div');
    tab.className = 'artifact-tab';
    tab.textContent = `${artifact.type} (${artifact.id.substring(0, 8)})`;
    tab.onclick = () => selectArtifact(artifact.id);
    tab.dataset.artifactId = artifact.id;
    artifactTabs.appendChild(tab);

    // Create viewer with specialized renderer
    const viewer = document.createElement('div');
    viewer.className = 'artifact-viewer';
    viewer.id = `artifact-${artifact.id}`;

    const renderer = ArtifactRenderers[artifact.type];
    const renderedContent = renderer ? renderer(artifact.data, artifact.id) : renderFallback(artifact);

    viewer.innerHTML = `
        <div class="artifact-header">
            <div class="artifact-title">${artifact.type.toUpperCase()}</div>
            <div class="artifact-meta">
                <span>ID: ${artifact.id.substring(0, 8)}</span>
                <span class="status-badge ${artifact.status}">${artifact.status}</span>
                <span>${new Date(artifact.created_at).toLocaleString()}</span>
            </div>
        </div>
        <div class="artifact-data">
            ${renderedContent}
        </div>
    `;

    artifactContent.appendChild(viewer);

    // Remove empty state
    const emptyState = artifactContent.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    // Auto-select new artifact
    selectArtifact(artifact.id);
}

function renderFallback(artifact) {
    return `<pre style="color: #e0e0e0;">${JSON.stringify(artifact.data, null, 2)}</pre>`;
}

function selectArtifact(artifactId) {
    state.activeArtifactId = artifactId;

    document.querySelectorAll('.artifact-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.artifactId === artifactId);
    });

    document.querySelectorAll('.artifact-viewer').forEach(viewer => {
        viewer.classList.toggle('active', viewer.id === `artifact-${artifactId}`);
    });
}

// ============================================================================
// Input Handling
// ============================================================================

function sendMessage() {
    const message = userInput.value.trim();
    if (!message || !state.isConnected) return;

    state.ws.send(JSON.stringify({
        type: 'user_message',
        content: message
    }));

    userInput.value = '';
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Initialize
connectWebSocket();
