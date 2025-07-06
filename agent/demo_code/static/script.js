document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const statusIndicator = document.getElementById('status-indicator');
    
    // Add welcome message
    addMessageToChat("Hello! I'm your research assistant. I can help you find and analyze academic papers, query knowledge, and answer questions. What would you like to explore today?", 'bot-message');
    
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
    
    // Send message with streaming
    function sendMessage() {
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
        
        // Add progress indicator
        const progressIndicator = createProgressIndicator();
        chatMessages.appendChild(progressIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        const startTime = Date.now();
        showStatus('Processing your request...', false);
        
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