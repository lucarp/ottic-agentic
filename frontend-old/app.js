/**
 * WebSocket client with specialized artifact renderers
 */

// State management
const state = {
    ws: null,
    artifacts: [],
    activeArtifactId: null,
    isConnected: false,
    charts: {},  // Store Chart.js instances
    currentMessageDiv: null,  // Track current streaming message
    currentReasoningDiv: null,  // Track current streaming reasoning
    pendingResponseId: null,  // Track response_id for continue execution
    lastUserInput: null  // Track last user input for continue
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
    },

    // ========================================================================
    // SEO Analysis Artifact Renderers - SE Ranking Integration
    // ========================================================================

    domain_overview: (data) => {
        const {
            domain,
            total_keywords,
            organic_traffic,
            organic_traffic_value,
            paid_keywords,
            paid_traffic,
            paid_traffic_value,
            currency,
            title
        } = data;

        const formatNumber = (num) => num.toLocaleString();
        const formatCurrency = (num) => `${currency} ${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        let html = '';
        if (title) html += `<h2 style="margin-bottom: 24px; color: #fff; border-bottom: 2px solid #3b82f6; padding-bottom: 12px;">${title}</h2>`;

        html += `<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 24px;">`;

        // Organic Traffic Card
        html += `
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="background: #10b981; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                        <span style="font-size: 20px;">📈</span>
                    </div>
                    <h3 style="color: #10b981; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Organic Traffic</h3>
                </div>
                <div style="font-size: 36px; font-weight: bold; color: #fff; margin-bottom: 8px;">${formatNumber(organic_traffic)}</div>
                <div style="color: #94a3b8; font-size: 13px; margin-bottom: 4px;">${formatNumber(total_keywords)} keywords</div>
                <div style="color: #10b981; font-size: 16px; font-weight: 600;">${formatCurrency(organic_traffic_value)}/mo</div>
            </div>
        `;

        // Paid Traffic Card
        html += `
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="background: #f59e0b; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                        <span style="font-size: 20px;">💰</span>
                    </div>
                    <h3 style="color: #f59e0b; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Paid Traffic</h3>
                </div>
                <div style="font-size: 36px; font-weight: bold; color: #fff; margin-bottom: 8px;">${formatNumber(paid_traffic)}</div>
                <div style="color: #94a3b8; font-size: 13px; margin-bottom: 4px;">${formatNumber(paid_keywords)} keywords</div>
                <div style="color: #f59e0b; font-size: 16px; font-weight: 600;">${formatCurrency(paid_traffic_value)}/mo</div>
            </div>
        `;

        // Total Value Card
        const totalTraffic = organic_traffic + paid_traffic;
        const totalValue = organic_traffic_value + paid_traffic_value;
        html += `
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="background: #6366f1; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                        <span style="font-size: 20px;">🎯</span>
                    </div>
                    <h3 style="color: #6366f1; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Total Estimated</h3>
                </div>
                <div style="font-size: 36px; font-weight: bold; color: #fff; margin-bottom: 8px;">${formatNumber(totalTraffic)}</div>
                <div style="color: #94a3b8; font-size: 13px; margin-bottom: 4px;">visitors/month</div>
                <div style="color: #6366f1; font-size: 16px; font-weight: 600;">${formatCurrency(totalValue)}/mo</div>
            </div>
        `;

        html += `</div>`;

        // Domain info footer
        html += `<div style="background: #0f172a; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">`;
        html += `<div style="color: #94a3b8; font-size: 13px;">Analyzing domain: <span style="color: #3b82f6; font-weight: 600;">${domain}</span></div>`;
        html += `<div style="color: #64748b; font-size: 12px; margin-top: 4px;">Data source: SE Ranking API</div>`;
        html += `</div>`;

        return html;
    },

    competitor_analysis: (data) => {
        const { target_domain, source, type, competitors, total_competitors, title } = data;

        let html = '';
        if (title) html += `<h2 style="margin-bottom: 24px; color: #fff; border-bottom: 2px solid #8b5cf6; padding-bottom: 12px;">${title}</h2>`;

        // Summary header
        html += `<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155;">`;
        html += `<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">`;
        html += `<span style="font-size: 24px;">🎯</span>`;
        html += `<h3 style="color: #fff; margin: 0;">Found ${total_competitors} ${type} competitors</h3>`;
        html += `</div>`;
        html += `<div style="color: #94a3b8; font-size: 14px;">Competing with <span style="color: #8b5cf6; font-weight: 600;">${target_domain}</span> in <span style="color: #8b5cf6; font-weight: 600;">${source.toUpperCase()}</span></div>`;
        html += `</div>`;

        if (competitors.length === 0) {
            html += `<div style="text-align: center; padding: 40px; color: #64748b;">No competitors found</div>`;
            return html;
        }

        // Competitors table
        html += `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background: #262626;">
                            <th style="padding: 14px; text-align: left; border-bottom: 2px solid #333; color: #8b5cf6; font-weight: 600; width: 80px;">Rank</th>
                            <th style="padding: 14px; text-align: left; border-bottom: 2px solid #333; color: #8b5cf6; font-weight: 600;">Competitor Domain</th>
                            <th style="padding: 14px; text-align: right; border-bottom: 2px solid #333; color: #8b5cf6; font-weight: 600; width: 180px;">Common Keywords</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        competitors.forEach((comp, index) => {
            const isTopCompetitor = comp.rank === 1;
            const bgColor = isTopCompetitor ? '#2d1b4e' : (index % 2 === 0 ? '#1a1a1a' : '#0f0f0f');

            html += `
                <tr style="background: ${bgColor}; border-bottom: 1px solid #333;">
                    <td style="padding: 14px; color: ${isTopCompetitor ? '#8b5cf6' : '#e0e0e0'}; font-weight: ${isTopCompetitor ? 'bold' : 'normal'};">
                        ${isTopCompetitor ? '👑 ' : ''}#${comp.rank}
                    </td>
                    <td style="padding: 14px;">
                        <a href="https://${comp.domain}" target="_blank" rel="noopener noreferrer" style="color: ${isTopCompetitor ? '#a78bfa' : '#3b82f6'}; text-decoration: none; font-weight: 500;">
                            ${comp.domain}
                        </a>
                    </td>
                    <td style="padding: 14px; text-align: right; color: #e0e0e0; font-weight: 600;">
                        ${comp.common_keywords.toLocaleString()}
                    </td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        return html;
    },

    keyword_research: (data) => {
        const {
            analysis_type,
            primary_keyword,
            target_domain,
            competitor_domain,
            source,
            keywords,
            total_results,
            title
        } = data;

        const getDifficultyColor = (difficulty) => {
            if (difficulty <= 30) return '#10b981'; // Easy - green
            if (difficulty <= 60) return '#f59e0b'; // Medium - orange
            return '#ef4444'; // Hard - red
        };

        const getDifficultyLabel = (difficulty) => {
            if (difficulty <= 30) return 'Easy';
            if (difficulty <= 60) return 'Medium';
            return 'Hard';
        };

        let html = '';
        if (title) html += `<h2 style="margin-bottom: 24px; color: #fff; border-bottom: 2px solid #10b981; padding-bottom: 12px;">${title}</h2>`;

        // Analysis info header
        html += `<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155;">`;
        html += `<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">`;
        html += `<span style="font-size: 24px;">🔍</span>`;
        html += `<h3 style="color: #fff; margin: 0;">Found ${total_results} keywords</h3>`;
        html += `</div>`;
        html += `<div style="color: #94a3b8; font-size: 14px;">`;
        if (analysis_type === 'similar') {
            html += `Similar to: <span style="color: #10b981; font-weight: 600;">${primary_keyword}</span>`;
        } else if (analysis_type === 'gap') {
            html += `Keywords where <span style="color: #ef4444; font-weight: 600;">${competitor_domain}</span> ranks but <span style="color: #3b82f6; font-weight: 600;">${target_domain}</span> doesn't`;
        }
        html += ` • Region: <span style="color: #10b981; font-weight: 600;">${source.toUpperCase()}</span>`;
        html += `</div>`;
        html += `</div>`;

        if (keywords.length === 0) {
            html += `<div style="text-align: center; padding: 40px; color: #64748b;">No keywords found</div>`;
            return html;
        }

        // Keywords table
        html += `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background: #262626;">
                            <th style="padding: 14px; text-align: left; border-bottom: 2px solid #333; color: #10b981; font-weight: 600;">Keyword</th>
                            <th style="padding: 14px; text-align: right; border-bottom: 2px solid #333; color: #10b981; font-weight: 600; width: 120px;">Volume</th>
                            <th style="padding: 14px; text-align: right; border-bottom: 2px solid #333; color: #10b981; font-weight: 600; width: 100px;">CPC</th>
                            <th style="padding: 14px; text-align: center; border-bottom: 2px solid #333; color: #10b981; font-weight: 600; width: 140px;">Difficulty</th>
                            ${analysis_type === 'gap' ? '<th style="padding: 14px; text-align: center; border-bottom: 2px solid #333; color: #10b981; font-weight: 600; width: 100px;">Position</th>' : ''}
                        </tr>
                    </thead>
                    <tbody>
        `;

        keywords.forEach((kw, index) => {
            const bgColor = index % 2 === 0 ? '#1a1a1a' : '#0f0f0f';
            const diffColor = getDifficultyColor(kw.difficulty);
            const diffLabel = getDifficultyLabel(kw.difficulty);

            html += `
                <tr style="background: ${bgColor}; border-bottom: 1px solid #333;">
                    <td style="padding: 14px; color: #e0e0e0; font-weight: 500;">${kw.keyword}</td>
                    <td style="padding: 14px; text-align: right; color: #94a3b8;">${kw.volume.toLocaleString()}</td>
                    <td style="padding: 14px; text-align: right; color: #94a3b8;">$${kw.cpc.toFixed(2)}</td>
                    <td style="padding: 14px; text-align: center;">
                        <div style="display: inline-block; padding: 4px 12px; background: ${diffColor}22; border: 1px solid ${diffColor}; border-radius: 12px; color: ${diffColor}; font-size: 12px; font-weight: 600;">
                            ${kw.difficulty} - ${diffLabel}
                        </div>
                    </td>
                    ${analysis_type === 'gap' && kw.position ? `<td style="padding: 14px; text-align: center; color: #e0e0e0; font-weight: 600;">#${kw.position}</td>` : (analysis_type === 'gap' ? '<td style="padding: 14px; text-align: center; color: #64748b;">-</td>' : '')}
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        // Legend footer
        html += `<div style="margin-top: 16px; padding: 12px; background: #0f172a; border-radius: 6px; font-size: 12px; color: #64748b;">`;
        html += `<strong style="color: #94a3b8;">Legend:</strong> `;
        html += `<span style="color: #10b981;">Easy (0-30)</span> • `;
        html += `<span style="color: #f59e0b;">Medium (31-60)</span> • `;
        html += `<span style="color: #ef4444;">Hard (61-100)</span>`;
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
            // Reset streaming state for new conversation turn
            state.currentMessageDiv = null;
            state.currentReasoningDiv = null;
            break;
        case 'reasoning':
            addReasoningMessage(message.content);
            break;
        case 'reasoning_delta':
            // STREAMING: Accumulate reasoning deltas
            handleReasoningDelta(message.delta);
            break;
        case 'text_delta':
            // STREAMING: Accumulate text deltas
            handleTextDelta(message.delta);
            break;
        case 'text_done':
            // STREAMING: Finalize current message
            handleTextDone();
            break;
        case 'assistant_message':
            // Legacy: Full message at once
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
        case 'max_turns_exceeded':
            addMaxTurnsMessage(message);
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

function addMaxTurnsMessage(message) {
    // Store response_id for continue execution
    state.pendingResponseId = message.response_id;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message max-turns-message';
    messageDiv.style.backgroundColor = '#2a2a2a';
    messageDiv.style.borderLeft = '3px solid #ffa500';
    messageDiv.style.padding = '15px';
    messageDiv.style.marginLeft = '5px';

    const textDiv = document.createElement('div');
    textDiv.textContent = message.message;
    textDiv.style.marginBottom = '10px';
    textDiv.style.color = '#ffa500';

    const buttonDiv = document.createElement('div');
    const continueButton = document.createElement('button');
    continueButton.textContent = 'Continue';
    continueButton.style.padding = '8px 20px';
    continueButton.style.backgroundColor = '#4CAF50';
    continueButton.style.color = 'white';
    continueButton.style.border = 'none';
    continueButton.style.borderRadius = '4px';
    continueButton.style.cursor = 'pointer';
    continueButton.style.fontSize = '14px';
    continueButton.style.fontWeight = 'bold';

    continueButton.onclick = () => {
        // Send continue execution request
        if (state.pendingResponseId && state.lastUserInput) {
            sendContinue(state.lastUserInput, state.pendingResponseId);
            // Disable button after click
            continueButton.disabled = true;
            continueButton.textContent = 'Continuing...';
            continueButton.style.backgroundColor = '#666';
        }
    };

    buttonDiv.appendChild(continueButton);
    messageDiv.appendChild(textDiv);
    messageDiv.appendChild(buttonDiv);
    terminalMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ============================================================================
// STREAMING: Text Delta Handling
// ============================================================================

function handleTextDelta(delta) {
    // Create message div if it doesn't exist
    if (!state.currentMessageDiv) {
        state.currentMessageDiv = document.createElement('div');
        state.currentMessageDiv.className = 'message assistant-message';
        terminalMessages.appendChild(state.currentMessageDiv);
    }

    // Append delta to current message
    state.currentMessageDiv.textContent += delta;
    scrollToBottom();
}

function handleTextDone() {
    // Finalize current streaming message
    state.currentMessageDiv = null;
}

function handleReasoningDelta(delta) {
    // Create reasoning div if it doesn't exist
    if (!state.currentReasoningDiv) {
        state.currentReasoningDiv = document.createElement('div');
        state.currentReasoningDiv.className = 'message reasoning-message';
        state.currentReasoningDiv.style.color = '#ffd700';  // Gold color for reasoning
        state.currentReasoningDiv.style.fontStyle = 'italic';
        state.currentReasoningDiv.style.borderLeft = '3px solid #ffd700';
        state.currentReasoningDiv.style.paddingLeft = '10px';
        state.currentReasoningDiv.style.marginLeft = '5px';
        terminalMessages.appendChild(state.currentReasoningDiv);
    }

    // Append delta to current reasoning message
    state.currentReasoningDiv.textContent += delta;
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
    console.log('sendMessage called, message:', message, 'isConnected:', state.isConnected);
    if (!message || !state.isConnected) {
        console.log('Message not sent - empty or not connected');
        return;
    }

    // Track last user input for continue execution
    state.lastUserInput = message;

    console.log('Sending message to WebSocket');
    state.ws.send(JSON.stringify({
        type: 'user_message',
        content: message
    }));

    userInput.value = '';
    console.log('Message sent successfully');
}

function sendContinue(originalInput, responseId) {
    if (!state.isConnected || !state.ws) {
        console.log('Cannot continue - not connected');
        return;
    }

    console.log('Sending continue_execution with response_id:', responseId);

    const message = {
        type: 'continue_execution',
        content: originalInput,
        previous_response_id: responseId
    };

    state.ws.send(JSON.stringify(message));
    console.log('Continue execution request sent');
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Initialize
connectWebSocket();
