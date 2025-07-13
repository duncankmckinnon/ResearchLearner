document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const endpointSelect = document.getElementById('endpoint-select');
    const endpointParams = document.getElementById('endpoint-params');
    const userIdInput = document.getElementById('user-id-input');
    const limitInput = document.getElementById('limit-input');
    
    // Add welcome message
    addMessageToChat("Hello! I'm your research assistant. I can help you find and analyze academic papers, query knowledge, and answer questions. Select an endpoint and try it out!", 'bot-message');
    
    // Handle endpoint selection
    endpointSelect.addEventListener('change', function() {
        const selectedEndpoint = endpointSelect.value;
        updateUIForEndpoint(selectedEndpoint);
    });
    
    // Update UI based on selected endpoint
    function updateUIForEndpoint(endpoint) {
        if (endpoint === 'agent') {
            endpointParams.style.display = 'none';
            messageInput.placeholder = "Ask me about research papers, topics, or general questions...";
        } else if (endpoint === 'knowledge/search') {
            endpointParams.style.display = 'flex';
            messageInput.placeholder = "Enter your search query (e.g., 'machine learning')";
        } else if (endpoint === 'knowledge/papers') {
            endpointParams.style.display = 'flex';
            messageInput.placeholder = "Enter topic for research papers (e.g., 'neural networks')";
        } else if (endpoint === 'knowledge/insights') {
            endpointParams.style.display = 'flex';
            messageInput.placeholder = "Enter topic for research insights (e.g., 'deep learning')";
        } else if (endpoint === 'knowledge/summary') {
            endpointParams.style.display = 'flex';
            limitInput.style.display = 'none';
            limitInput.previousElementSibling.style.display = 'none'; // Hide limit label
            messageInput.placeholder = "Enter topic for knowledge summary (e.g., 'artificial intelligence')";
        } else if (endpoint === 'knowledge/memories') {
            endpointParams.style.display = 'flex';
            messageInput.placeholder = "Press Send to get all memories (input field ignored)";
        }
        
        // Show/hide limit input based on endpoint
        if (endpoint === 'knowledge/summary') {
            limitInput.style.display = 'none';
            limitInput.previousElementSibling.style.display = 'none';
        } else if (endpoint !== 'agent') {
            limitInput.style.display = 'inline-block';
            limitInput.previousElementSibling.style.display = 'inline';
        }
    }
    
    // Generate or retrieve conversation hash
    function getConversationHash() {
        let hash = localStorage.getItem('conversationHash');
        if (!hash) {
            // Generate a random hash if none exists
            hash = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            localStorage.setItem('conversationHash', hash);
        }
        return hash;
    }
    
    // Show status indicator
    function showStatus(message, isError = false) {
        statusIndicator.textContent = message;
        statusIndicator.className = `status-indicator ${isError ? 'status-error' : 'status-healthy'}`;
        statusIndicator.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            statusIndicator.style.display = 'none';
        }, 3000);
    }
    
    // Create progress indicator
    function createProgressIndicator() {
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message bot-message progress-indicator';
        progressDiv.innerHTML = `
            <div class="progress-content">
                <div class="progress-text">Starting analysis...</div>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
                <div class="progress-step">Step 1 of 5</div>
            </div>
        `;
        return progressDiv;
    }
    
    // Update progress indicator
    function updateProgressIndicator(progressDiv, message, step, totalSteps) {
        const progressText = progressDiv.querySelector('.progress-text');
        const progressBar = progressDiv.querySelector('.progress-bar');
        const progressStep = progressDiv.querySelector('.progress-step');
        
        if (progressText) progressText.textContent = message;
        if (progressBar) {
            const percentage = (step / totalSteps) * 100;
            progressBar.style.width = `${percentage}%`;
        }
        if (progressStep) progressStep.textContent = `Step ${step} of ${totalSteps}`;
    }
    
    // Send message with streaming or direct API calls
    function sendMessage() {
        const message = messageInput.value.trim();
        const requestTimestamp = new Date().toISOString();
        const conversationHash = getConversationHash();
        const selectedEndpoint = endpointSelect.value;
        
        // For agent endpoint, require message. For memories endpoint, message is optional
        if (!message && selectedEndpoint !== 'knowledge/memories') {
            return;
        }
        
        // Add user message to chat (unless it's memories endpoint with no message)
        if (message || selectedEndpoint !== 'knowledge/memories') {
            addMessageToChat(message || 'Getting all memories...', 'user-message');
        }
        messageInput.value = '';
        
        // Disable send button during processing
        sendBtn.disabled = true;
        sendBtn.textContent = 'Processing...';
        
        const startTime = Date.now();
        showStatus('Processing your request...', false);
        
        // Route to appropriate endpoint
        if (selectedEndpoint === 'agent') {
            // Use existing streaming logic for agent
            sendToAgentStream(message, conversationHash, requestTimestamp, startTime);
        } else {
            // Use direct API calls for knowledge endpoints
            sendToKnowledgeEndpoint(selectedEndpoint, message, startTime);
        }
    }
    
    // Send to agent with streaming
    function sendToAgentStream(message, conversationHash, requestTimestamp, startTime) {
        // Add progress indicator
        const progressIndicator = createProgressIndicator();
        chatMessages.appendChild(progressIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Use streaming endpoint
        fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_hash: conversationHash,
                message: message,
                request_timestamp: requestTimestamp
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = ''; // Buffer for partial chunks
            
            function processStream() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        // Stream finished - process any remaining buffer
                        if (buffer.trim()) {
                            processBufferedData(buffer);
                        }
                        
                        // Clean up
                        console.log('Stream finished - checking if response was received');
                        if (progressIndicator && progressIndicator.parentNode) {
                            console.log('Progress indicator still present - removing it now');
                            progressIndicator.remove();
                        }
                        sendBtn.disabled = false;
                        sendBtn.textContent = 'Send';
                        
                        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                        showStatus(`Completed in ${duration}s`, false);
                        return;
                    }
                    
                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;
                    
                    // Process complete data lines from buffer
                    processCompleteLines();
                    
                    // Continue reading
                    return processStream();
                });
            }
            
            function processCompleteLines() {
                let startIndex = 0;
                
                while (true) {
                    // Find the next "data: " line
                    const dataStart = buffer.indexOf('data: ', startIndex);
                    if (dataStart === -1) {
                        // No more data lines found, keep remaining buffer
                        buffer = buffer.substring(startIndex);
                        break;
                    }
                    
                    // Find the end of this data line (next newline followed by either another "data: " or end of buffer)
                    let dataEnd = buffer.indexOf('\n', dataStart);
                    if (dataEnd === -1) {
                        // Line is not complete yet, keep from dataStart
                        buffer = buffer.substring(dataStart);
                        break;
                    }
                    
                    // Check if this might be part of a multi-line JSON by looking for the next data line
                    const nextDataStart = buffer.indexOf('\ndata: ', dataEnd);
                    
                    // If we found a complete line and it's followed by another data line or end of buffer
                    if (nextDataStart !== -1 || dataEnd === buffer.length - 1) {
                        const line = buffer.substring(dataStart, dataEnd);
                        processDataLine(line);
                        startIndex = dataEnd + 1;
                    } else {
                        // This line might be incomplete, keep from dataStart
                        buffer = buffer.substring(dataStart);
                        break;
                    }
                }
            }
            
            function processDataLine(line) {
                if (line.startsWith('data: ')) {
                    const jsonData = line.slice(6).trim();
                    if (jsonData) {
                        // Validate JSON before parsing
                        if (isValidJSON(jsonData)) {
                            try {
                                const data = JSON.parse(jsonData);
                                handleStreamEvent(data, progressIndicator);
                            } catch (e) {
                                console.error('Error parsing validated JSON:', e, 'Line:', line);
                            }
                        } else {
                            console.warn('Invalid JSON detected, skipping:', jsonData.substring(0, 100) + '...');
                        }
                    }
                }
            }
            
            function isValidJSON(str) {
                try {
                    // Quick validation - check if it starts with { and has matching braces
                    if (!str.startsWith('{')) return false;
                    
                    let braceCount = 0;
                    let inString = false;
                    let escaped = false;
                    
                    for (let i = 0; i < str.length; i++) {
                        const char = str[i];
                        
                        if (escaped) {
                            escaped = false;
                            continue;
                        }
                        
                        if (char === '\\') {
                            escaped = true;
                            continue;
                        }
                        
                        if (char === '"') {
                            inString = !inString;
                            continue;
                        }
                        
                        if (!inString) {
                            if (char === '{') braceCount++;
                            else if (char === '}') braceCount--;
                        }
                    }
                    
                    return braceCount === 0 && !inString;
                } catch (e) {
                    return false;
                }
            }
            
            function processBufferedData(data) {
                const lines = data.split('\n');
                for (const line of lines) {
                    processDataLine(line);
                }
            }
            
            return processStream();
        })
        .catch(error => {
            console.error('Streaming error:', error);
            progressIndicator.remove();
            addMessageToChat('Sorry, I encountered an error while processing your request. Please try again.', 'bot-message error-message');
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            showStatus('Error processing request', true);
        });
    }
    
    // Handle stream events
    function handleStreamEvent(data, progressIndicator) {
        console.log('Received stream event:', data.type, data);
        
        switch (data.type) {
            case 'status':
                updateProgressIndicator(progressIndicator, data.message, 1, 5);
                break;
                
            case 'progress':
                updateProgressIndicator(progressIndicator, data.message, data.step, data.total_steps);
                break;
                
            case 'response':
                console.log('🎯 RESPONSE EVENT RECEIVED!');
                console.log('Processing response event with data:', data);
                console.log('Response text:', data.response);
                console.log('Response length:', data.response ? data.response.length : 'undefined');
                
                // Remove progress indicator and add final response
                if (progressIndicator && progressIndicator.parentNode) {
                    console.log('Removing progress indicator...');
                    progressIndicator.remove();
                } else {
                    console.log('No progress indicator to remove');
                }
                
                console.log('Adding response message to chat...');
                console.log('Chat messages element:', chatMessages);
                console.log('Current chat messages count:', chatMessages.children.length);
                
                // Add the response message
                if (data.response && data.response.trim() !== '') {
                    addMessageToChat(data.response, 'bot-message');
                    console.log('Response message added successfully');
                } else {
                    console.error('Empty or undefined response received:', data.response);
                    addMessageToChat('Empty response received from server', 'bot-message error-message');
                }
                
                // Add metadata if available
                if (data.intent || data.plan) {
                    const metadata = [];
                    if (data.intent) metadata.push(`Intent: ${data.intent}`);
                    if (data.plan && data.plan.length > 0) metadata.push(`Plan: ${data.plan.join(', ')}`);
                    
                    const metadataDiv = document.createElement('div');
                    metadataDiv.className = 'message bot-message metadata';
                    metadataDiv.innerHTML = `<small>${metadata.join(' | ')}</small>`;
                    chatMessages.appendChild(metadataDiv);
                    console.log('Metadata added:', metadata);
                }
                break;
                
            case 'error':
                if (progressIndicator && progressIndicator.parentNode) {
                    progressIndicator.remove();
                }
                addMessageToChat(`Error: ${data.message}`, 'bot-message error-message');
                showStatus('Processing error', true);
                break;
                
            case 'complete':
                // Stream completed successfully
                console.log('Stream completed:', data.message);
                break;
                
            default:
                console.log('Unknown stream event:', data);
        }
        
        // Auto-scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Fallback to non-streaming for older browsers
    function sendMessageFallback() {
        const message = messageInput.value.trim();
        const requestTimestamp = new Date().toISOString();
        const conversationHash = getConversationHash();
        
        if (!message) {
            return;
        }
        
        // Add user message to chat
        addMessageToChat(message, 'user-message');
        messageInput.value = '';
        
        // Disable send button during processing
        sendBtn.disabled = true;
        sendBtn.textContent = 'Processing...';
        
        // Add a typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message typing-indicator';
        typingIndicator.textContent = 'Thinking... (this may take a while for complex research)';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        const startTime = Date.now();
        showStatus('Processing (fallback mode)...', false);
        
        // Send message to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_hash: conversationHash,
                message: message,
                request_timestamp: requestTimestamp
            })
        })
        .then(response => response.json())
        .then(data => {
            typingIndicator.remove();
            addMessageToChat(data.response, 'bot-message');
            
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            showStatus(`Completed in ${duration}s`, false);
            
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        })
        .catch(error => {
            console.error('Error:', error);
            typingIndicator.remove();
            addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot-message error-message');
            showStatus('Error processing request', true);
            
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        });
    }
    
    // Send to knowledge endpoints
    function sendToKnowledgeEndpoint(endpoint, message, startTime) {
        const userId = userIdInput.value.trim() || 'demo_user';
        const limit = parseInt(limitInput.value) || 10;
        
        let url, method, body = null, params = new URLSearchParams();
        
        // Build URL and parameters based on endpoint
        switch (endpoint) {
            case 'knowledge/search':
                url = '/api/knowledge/search';
                method = 'POST';
                body = JSON.stringify({
                    query: message,
                    limit: limit,
                });
                break;
                
            case 'knowledge/papers':
                url = `/api/knowledge/papers/${encodeURIComponent(message)}`;
                method = 'GET';
                params.append('limit', limit);
                break;
                
            case 'knowledge/insights':
                url = `/api/knowledge/insights/${encodeURIComponent(message)}`;
                method = 'GET';
                params.append('limit', limit);
                break;
                
            case 'knowledge/summary':
                url = `/api/knowledge/summary/${encodeURIComponent(message)}`;
                method = 'GET';
                break;
                
            case 'knowledge/memories':
                url = '/api/knowledge/memories';
                method = 'GET';
                params.append('limit', limit);
                break;
                
            default:
                console.error('Unknown endpoint:', endpoint);
                return;
        }
        
        // Add query parameters for GET requests
        if (method === 'GET' && params.toString()) {
            url += '?' + params.toString();
        }
        
        // Make the request
        const fetchOptions = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (body) {
            fetchOptions.body = body;
        }
        
        fetch(url, fetchOptions)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Format and display the response
                const formattedResponse = formatKnowledgeResponse(endpoint, data);
                addMessageToChat(formattedResponse, 'bot-message');
                
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                showStatus(`Completed in ${duration}s`, false);
                
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            })
            .catch(error => {
                console.error('Knowledge endpoint error:', error);
                addMessageToChat(`Error: ${error.message}`, 'bot-message error-message');
                showStatus('Error processing request', true);
                
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            });
    }
    
    // Format knowledge endpoint responses
    function formatKnowledgeResponse(endpoint, data) {
        switch (endpoint) {
            case 'knowledge/search':
                if (data.results && data.results.length > 0) {
                    let response = `## Search Results (${data.total_results} found)\n\n`;
                    data.results.forEach((result, index) => {
                        response += `### Result ${index + 1} (Score: ${result.relevance_score.toFixed(2)})\n`;
                        response += `${result.content}\n\n`;
                        if (result.metadata.type) {
                            response += `*Type: ${result.metadata.type}*\n\n`;
                        }
                    });
                    return response;
                } else {
                    return 'No search results found.';
                }
                
            case 'knowledge/papers':
                if (data.papers && data.papers.length > 0) {
                    let response = `## Research Papers (${data.total_papers} found)\n\n`;
                    data.papers.forEach((paper, index) => {
                        response += `### ${index + 1}. ${paper.title}\n`;
                        response += `**Authors:** ${paper.authors.join(', ')}\n`;
                        response += `**ArXiv ID:** ${paper.arxiv_id}\n`;
                        response += `**Categories:** ${paper.categories.join(', ')}\n`;
                        response += `**Source:** ${paper.source}\n`;
                        response += `**Score:** ${paper.relevance_score.toFixed(2)}\n\n`;
                        response += `${paper.content.substring(0, 200)}...\n\n`;
                    });
                    return response;
                } else {
                    return 'No research papers found.';
                }
                
            case 'knowledge/insights':
                if (data.insights && data.insights.length > 0) {
                    let response = `## Research Insights (${data.total_insights} found)\n\n`;
                    data.insights.forEach((insight, index) => {
                        response += `### Insight ${index + 1} - ${insight.topic}\n`;
                        response += `${insight.insight}\n`;
                        response += `*Added: ${insight.added_date}*\n`;
                        response += `*Score: ${insight.relevance_score.toFixed(2)}*\n\n`;
                    });
                    return response;
                } else {
                    return 'No research insights found.';
                }
                
            case 'knowledge/summary':
                let response = `## Knowledge Summary: ${data.topic}\n\n`;
                response += `**Total Papers:** ${data.total_papers}\n`;
                response += `**Total Insights:** ${data.total_insights}\n`;
                response += `**Total Knowledge Items:** ${data.total_knowledge_items}\n\n`;
                
                if (data.related_papers.length > 0) {
                    response += `### Related Papers\n`;
                    data.related_papers.forEach((paper, index) => {
                        response += `${index + 1}. **${paper.title}** by ${paper.authors.join(', ')}\n`;
                    });
                    response += '\n';
                }
                
                if (data.research_insights.length > 0) {
                    response += `### Research Insights\n`;
                    data.research_insights.forEach((insight, index) => {
                        response += `${index + 1}. ${insight.insight.substring(0, 100)}...\n`;
                    });
                    response += '\n';
                }
                
                return response;
                
            case 'knowledge/memories':
                if (data.memories && data.memories.length > 0) {
                    let response = `## All Memories (${data.total_memories} found)\n\n`;
                    data.memories.forEach((memory, index) => {
                        response += `### Memory ${index + 1}\n`;
                        response += `${memory.content}\n`;
                        if (memory.metadata.type) {
                            response += `*Type: ${memory.metadata.type}*\n`;
                        }
                        response += `*Created: ${memory.created_at}*\n\n`;
                    });
                    return response;
                } else {
                    return 'No memories found.';
                }
                
            default:
                return JSON.stringify(data, null, 2);
        }
    }
    
    // Simple markdown formatter
    function formatMarkdown(text) {
        // Escape HTML first to prevent XSS
        const escapeHtml = (unsafe) => {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        };
        
        let html = escapeHtml(text);
        
        // Format headers (### ## #)
        html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
        
        // Format bold (**text** or __text__)
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
        
        // Format italic (*text* or _text_)
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/_(.*?)_/g, '<em>$1</em>');
        
        // Format code blocks (```code```)
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Format inline code (`code`)
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Format links [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Format line breaks
        html = html.replace(/\n/g, '<br>');
        
        // Format lists
        html = html.replace(/^- (.*$)/gm, '<li>$1</li>');
        html = html.replace(/^(\d+)\. (.*$)/gm, '<li>$1. $2</li>');
        
        // Wrap consecutive list items in ul tags
        html = html.replace(/(<li>.*<\/li>)/g, function(match) {
            return match.replace(/(<li>.*?<\/li>)(?=<li>|$)/g, '$1');
        });
        
        // Simple list wrapping (basic implementation)
        html = html.replace(/(<li>.*?<\/li>)(?:\s*<li>.*?<\/li>)*/g, '<ul>$&</ul>');
        
        return html;
    }
    
    // Add message to chat
    function addMessageToChat(message, className) {
        console.log('Adding message to chat:', message?.substring(0, 100) + '...', 'with class:', className);
        
        if (!message || message.trim() === '') {
            console.warn('Attempted to add empty message, using fallback');
            message = 'Empty response received';
        }
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${className}`;
        
        // Check if this is a bot message and should be formatted as markdown
        if (className.includes('bot-message') && !className.includes('error-message')) {
            console.log('Formatting as markdown for bot message');
            const formattedHtml = formatMarkdown(message);
            console.log('Formatted HTML preview:', formattedHtml.substring(0, 200) + '...');
            messageElement.innerHTML = formattedHtml;
        } else {
            console.log('Using text content (no markdown formatting)');
            messageElement.textContent = message;
        }
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        console.log('Message added successfully, total messages:', chatMessages.children.length);
        console.log('Message element:', messageElement);
    }
    
    // Check if streaming is supported
    function supportsStreaming() {
        return typeof ReadableStream !== 'undefined' && typeof Response !== 'undefined';
    }
    
    // Event listeners
    sendBtn.addEventListener('click', function() {
        if (supportsStreaming()) {
            sendMessage();
        } else {
            sendMessageFallback();
        }
    });
    
    messageInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            if (supportsStreaming()) {
                sendMessage();
            } else {
                sendMessageFallback();
            }
        }
    });
    
    // Add connection status indicator
    function checkConnection() {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy') {
                    console.log('Connection healthy');
                    showStatus('🟢 Connected', false);
                } else {
                    console.warn('Connection issues detected');
                    showStatus('🟡 Connection issues', true);
                }
            })
            .catch(error => {
                console.error('Connection check failed:', error);
                showStatus('🔴 Connection failed', true);
            });
    }
    
    // Check connection on load
    checkConnection();
    
    // Periodic connection check
    setInterval(checkConnection, 30000); // Check every 30 seconds
    
    // Show streaming capability
    if (supportsStreaming()) {
        console.log('✅ Streaming support enabled');
    } else {
        console.log('⚠️ Fallback mode (no streaming support)');
        showStatus('⚠️ Limited browser support', true);
    }
}); 