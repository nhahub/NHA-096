const messagesContainer = document.getElementById('messagesContainer');
      const messageInput = document.getElementById('messageInput');
      const sendButton = document.getElementById('sendButton');

      function formatTime(date) {
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        return `${hours}:${minutes}`;
      }

      function addMessage(text, isUser) {
        const message = document.createElement('div');
        message.className = `message ${isUser ? 'user' : 'bot'}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = isUser ? '👤' : '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;

        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = formatTime(new Date());

        content.appendChild(bubble);
        content.appendChild(time);
        message.appendChild(avatar);
        message.appendChild(content);

        messagesContainer.appendChild(message);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }

      function showTypingIndicator() {
        const message = document.createElement('div');
        message.className = 'message bot';
        message.id = 'typingIndicator';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';

        for (let i = 0; i < 3; i++) {
          const dot = document.createElement('div');
          dot.className = 'typing-dot';
          indicator.appendChild(dot);
        }

        content.appendChild(indicator);
        message.appendChild(avatar);
        message.appendChild(content);

        messagesContainer.appendChild(message);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }

      function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
          indicator.remove();
        }
      }

      function handleSend() {
        const text = messageInput.value.trim();
        if (!text) return;

        addMessage(text, true);
        messageInput.value = '';
        messageInput.style.height = 'auto';
        updateSendButtonState();

        showTypingIndicator();

        fetch('/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: text }),
        })
          .then((response) => response.json())
          .then((data) => {
            removeTypingIndicator();
            addMessage(data.reply ?? data.response ?? "No response from server.", false);
          })
          .catch((error) => {
            removeTypingIndicator();
            addMessage("Sorry, something went wrong. Please try again.", false);
            console.error('Error:', error);
          });
      }

      function updateSendButtonState() {
        const hasText = messageInput.value.trim().length > 0;
        sendButton.disabled = !hasText;
      }

      messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 6 * 16) + 'px';
        updateSendButtonState();
      });

      messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      });

      sendButton.addEventListener('click', handleSend);

      // Theme initialization and toggle (persist in localStorage)
      (function(){
        const themeToggle = document.getElementById('themeToggle');
        if (!themeToggle) return;
        const applyTheme = (t) => {
          document.documentElement.setAttribute('data-theme', t);
          localStorage.setItem('theme', t);
          themeToggle.textContent = t === 'dark' ? '🌞' : '🌙';
        };
        const saved = localStorage.getItem('theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        applyTheme(saved);
        themeToggle.addEventListener('click', () => {
          const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
          applyTheme(cur === 'dark' ? 'light' : 'dark');
        });
      })();

      updateSendButtonState();