// IVF Journey Tracker - AI Chatbot JavaScript

// Chatbot state management
const ChatBot = {
    isOpen: false,
    isTyping: false,
    messageHistory: [],
    currentConversation: null,
    
    // Initialize chatbot
    init: function() {
        this.bindEvents();
        this.loadChatHistory();
        console.log('AI Chatbot initialized');
    },
    
    // Bind chatbot events
    bindEvents: function() {
        // Chat header click to toggle
        const chatHeader = document.querySelector('.chat-header');
        if (chatHeader) {
            chatHeader.addEventListener('click', () => this.toggle());
        }
        
        // Send message on button click
        const sendButton = document.querySelector('.chat-input button');
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendMessage());
        }
        
        // Send message on Enter key
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Auto-resize input
            chatInput.addEventListener('input', this.autoResizeInput);
        }
        
        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
        
        // Click outside to close on mobile
        document.addEventListener('click', (e) => {
            const chatWidget = document.getElementById('chatWidget');
            if (chatWidget && !chatWidget.contains(e.target) && this.isOpen && window.innerWidth <= 768) {
                this.close();
            }
        });
    },
    
    // Toggle chat widget
    toggle: function() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    },
    
    // Open chat widget
    open: function() {
        const chatBody = document.getElementById('chatBody');
        const chatToggle = document.getElementById('chatToggle');
        
        if (chatBody && chatToggle) {
            chatBody.style.display = 'flex';
            chatToggle.classList.remove('fa-chevron-up');
            chatToggle.classList.add('fa-chevron-down');
            this.isOpen = true;
            
            // Focus input
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                setTimeout(() => chatInput.focus(), 100);
            }
            
            // Scroll to bottom
            this.scrollToBottom();
            
            // Load initial message if no history
            if (this.messageHistory.length === 0) {
                this.showWelcomeMessage();
            }
        }
    },
    
    // Close chat widget
    close: function() {
        const chatBody = document.getElementById('chatBody');
        const chatToggle = document.getElementById('chatToggle');
        
        if (chatBody && chatToggle) {
            chatBody.style.display = 'none';
            chatToggle.classList.remove('fa-chevron-down');
            chatToggle.classList.add('fa-chevron-up');
            this.isOpen = false;
        }
    },
    
    // Show welcome message
    showWelcomeMessage: function() {
        const welcomeMessages = [
            "Hello! I'm your AI assistant specialized in IVF support. How can I help you today?",
            "Hi there! I'm here to answer your questions about IVF procedures, medications, and wellness. What would you like to know?",
            "Welcome! I'm your personal IVF AI assistant. Feel free to ask me about your treatment, symptoms, or any concerns you have."
        ];
        
        const randomMessage = welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
        this.addMessage(randomMessage, 'bot');
    },
    
    // Send message
    sendMessage: async function() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput?.value?.trim();
        
        if (!message || this.isTyping) return;
        
        // Clear input and add user message
        chatInput.value = '';
        this.addMessage(message, 'user');
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send to backend
            const response = await this.sendToAPI(message);
            
            // Remove typing indicator and add response
            this.hideTypingIndicator();
            this.addMessage(response.response, 'bot');
            
            // Save to history
            this.saveMessageToHistory(message, response.response);
            
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            
            const errorMessage = this.getErrorMessage(error);
            this.addMessage(errorMessage, 'bot', true);
        }
        
        // Focus input
        chatInput.focus();
    },
    
    // Send message to API
    sendToAPI: async function(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    },
    
    // Add message to chat
    addMessage: function(content, sender, isError = false) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message ${isError ? 'error-message' : ''}`;
        
        if (sender === 'bot') {
            messageDiv.innerHTML = `
                <div class="d-flex align-items-start">
                    <i class="fas fa-robot me-2 mt-1"></i>
                    <div class="message-content">${this.formatMessage(content)}</div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="d-flex align-items-start justify-content-end">
                    <div class="message-content">${this.escapeHtml(content)}</div>
                    <i class="fas fa-user me-2 mt-1 ms-2"></i>
                </div>
            `;
        }
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
        messageDiv.appendChild(timestamp);
        
        messagesContainer.appendChild(messageDiv);
        
        // Add to message history
        this.messageHistory.push({
            content,
            sender,
            timestamp: new Date().toISOString(),
            isError
        });
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Animate message
        requestAnimationFrame(() => {
            messageDiv.classList.add('animate-fade-in');
        });
    },
    
    // Format bot message content
    formatMessage: function(content) {
        // Convert newlines to <br>
        content = content.replace(/\n/g, '<br>');
        
        // Convert **bold** to <strong>
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert *italic* to <em>
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Convert lists
        content = content.replace(/^- (.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Convert numbers lists
        content = content.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
        content = content.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
        
        return content;
    },
    
    // Escape HTML for user messages
    escapeHtml: function(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    // Show typing indicator
    showTypingIndicator: function() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.innerHTML = `
            <div class="d-flex align-items-start">
                <i class="fas fa-robot me-2 mt-1"></i>
                <div class="typing-animation">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        typingDiv.id = 'typingIndicator';
        
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    },
    
    // Hide typing indicator
    hideTypingIndicator: function() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.isTyping = false;
    },
    
    // Scroll to bottom of messages
    scrollToBottom: function() {
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            requestAnimationFrame(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            });
        }
    },
    
    // Auto-resize input
    autoResizeInput: function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    },
    
    // Get error message
    getErrorMessage: function(error) {
        if (error.message?.includes('HTTP 429')) {
            return "I'm getting a lot of questions right now. Please wait a moment and try again.";
        } else if (error.message?.includes('HTTP 500')) {
            return "I'm experiencing some technical difficulties. Please try again in a few minutes.";
        } else if (error.message?.includes('Failed to fetch')) {
            return "I can't connect to the server right now. Please check your internet connection and try again.";
        } else {
            return "I'm sorry, I encountered an error while processing your request. Please try rephrasing your question or try again later.";
        }
    },
    
    // Save message to history
    saveMessageToHistory: function(userMessage, botResponse) {
        const history = this.loadChatHistory();
        history.push({
            userMessage,
            botResponse,
            timestamp: new Date().toISOString()
        });
        
        // Keep only last 50 conversations
        if (history.length > 50) {
            history.splice(0, history.length - 50);
        }
        
        try {
            localStorage.setItem('ivf_chat_history', JSON.stringify(history));
        } catch (e) {
            console.warn('Could not save chat history:', e);
        }
    },
    
    // Load chat history
    loadChatHistory: function() {
        try {
            const history = localStorage.getItem('ivf_chat_history');
            return history ? JSON.parse(history) : [];
        } catch (e) {
            console.warn('Could not load chat history:', e);
            return [];
        }
    },
    
    // Clear chat history
    clearHistory: function() {
        this.messageHistory = [];
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            // Keep only the welcome message
            const messages = messagesContainer.querySelectorAll('.message:not(.welcome-message)');
            messages.forEach(msg => msg.remove());
        }
        
        try {
            localStorage.removeItem('ivf_chat_history');
        } catch (e) {
            console.warn('Could not clear chat history:', e);
        }
        
        this.showWelcomeMessage();
    },
    
    // Quick response buttons
    addQuickResponses: function(responses) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const quickResponseDiv = document.createElement('div');
        quickResponseDiv.className = 'quick-responses my-2';
        quickResponseDiv.innerHTML = `
            <div class="d-flex flex-wrap gap-2">
                ${responses.map(response => `
                    <button class="btn btn-sm btn-outline-primary quick-response-btn" 
                            data-message="${this.escapeHtml(response)}">
                        ${this.escapeHtml(response)}
                    </button>
                `).join('')}
            </div>
        `;
        
        // Add click handlers
        quickResponseDiv.querySelectorAll('.quick-response-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const message = btn.getAttribute('data-message');
                document.getElementById('chatInput').value = message;
                this.sendMessage();
                quickResponseDiv.remove(); // Remove quick responses after selection
            });
        });
        
        messagesContainer.appendChild(quickResponseDiv);
        this.scrollToBottom();
    },
    
    // Show common IVF questions as quick responses
    showCommonQuestions: function() {
        const commonQuestions = [
            "What should I expect during my first IVF consultation?",
            "How do I prepare for egg retrieval?",
            "What are the side effects of fertility medications?",
            "When should I take my trigger shot?",
            "What foods should I avoid during IVF treatment?",
            "How can I manage stress during treatment?",
            "What supplements should I take?",
            "When will I know if the transfer was successful?"
        ];
        
        this.addQuickResponses(commonQuestions.slice(0, 4)); // Show 4 random questions
    }
};

// Function to open chatbot in a small popup window
function openChatbotPopup() {
    const popupWidth = 400;
    const popupHeight = 600;
    const left = (window.innerWidth - popupWidth) / 2;
    const top = (window.innerHeight - popupHeight) / 2;

    const popup = window.open(
        '/chatbot',
        'chatbotPopup',
        `width=${popupWidth},height=${popupHeight},left=${left},top=${top},scrollbars=yes,resizable=yes`
    );

    if (popup) {
        popup.focus();
    } else {
        // Fallback if popup is blocked
        window.location.href = '/chatbot';
    }
}

// Quick access functions for global use
function toggleChat() {
    ChatBot.toggle();
}

function sendMessage() {
    ChatBot.sendMessage();
}

function clearChatHistory() {
    if (confirm('Are you sure you want to clear your chat history?')) {
        ChatBot.clearHistory();
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    ChatBot.init();

    // Only run this if the chat widget exists on the page
    if (document.getElementById('chatWidget')) {
        // Show common questions after a delay if no messages
        setTimeout(() => {
            if (ChatBot.messageHistory.length <= 1) { // Only welcome message
                ChatBot.addMessage("Here are some common questions I can help with:", 'bot');
                ChatBot.showCommonQuestions();
            }
        }, 3000);
    }
});

// Add CSS for typing animation and other chat styles
const chatStyles = `
<style>
.message {
    margin-bottom: 1rem;
    animation: fadeInUp 0.3s ease-out;
    max-width: 90%;
}

.user-message {
    margin-left: auto;
    text-align: right;
}

.bot-message {
    margin-right: auto;
}

.message-content {
    background: white;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    word-wrap: break-word;
    line-height: 1.4;
}

.user-message .message-content {
    background: var(--gradient-primary);
    color: white;
    border-bottom-right-radius: 0.25rem;
}

.bot-message .message-content {
    background: #f8f9fa;
    color: var(--dark-color);
    border-bottom-left-radius: 0.25rem;
    border: 1px solid #e9ecef;
}

.error-message .message-content {
    background: rgba(220, 53, 69, 0.1);
    border-color: var(--danger-color);
    color: var(--danger-color);
}

.message-timestamp {
    font-size: 0.7rem;
    color: var(--secondary-color);
    text-align: center;
    margin-top: 0.25rem;
}

.typing-indicator {
    opacity: 0.7;
}

.typing-animation {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
}

.typing-animation span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--secondary-color);
    display: inline-block;
    margin: 0 2px;
    animation: typing 1.4s infinite;
}

.typing-animation span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-animation span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
    }
    30% {
        transform: translateY(-10px);
        opacity: 1;
    }
}

.quick-responses {
    text-align: center;
    margin: 1rem 0;
}

.quick-response-btn {
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    white-space: nowrap;
    transition: all 0.2s ease;
}

.quick-response-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.chat-input textarea {
    resize: none;
    min-height: 40px;
    max-height: 120px;
    border-radius: 20px;
    padding: 0.75rem 1rem;
}

.chat-messages {
    scroll-behavior: smooth;
}

.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 10px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .chat-widget {
        width: calc(100vw - 20px);
        left: 10px;
        right: 10px;
        bottom: 10px;
    }
    
    .message {
        max-width: 95%;
    }
    
    .quick-response-btn {
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .bot-message .message-content {
        background: #2d3748;
        color: #e2e8f0;
        border-color: #4a5568;
    }
    
    .chat-messages {
        background-color: #1a202c;
    }
    
    .chat-input {
        background: #2d3748;
        border-color: #4a5568;
    }
}

/* Accessibility improvements */
.message:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

.quick-response-btn:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

/* High contrast mode */
@media (prefers-contrast: high) {
    .message-content {
        border: 2px solid #000;
    }
    
    .user-message .message-content {
        background: #000;
        color: #fff;
    }
    
    .bot-message .message-content {
        background: #fff;
        color: #000;
    }
}
</style>
`;

// Inject styles into document head
document.head.insertAdjacentHTML('beforeend', chatStyles);

// Export for use in other scripts
window.ChatBot = ChatBot;
