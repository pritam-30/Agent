const chatForm = document.getElementById("chat-form");

const messageInput = document.getElementById("message-input");

const chatContainer = document.getElementById("chat-container");

const sendButton = document.getElementById("send-button");

const newChatButton = document.getElementById("new-chat-button");

const welcomeMessage = document.getElementById("welcome-message");

// =====================================================
// Conversation
// =====================================================

let conversationId = localStorage.getItem("conversation_id");

// =====================================================
// Add Message
// =====================================================

function addMessage(role, text, retrievedChunks = null) {
  const message = document.createElement("div");

  message.classList.add("message", role);

  const wrapper = document.createElement("div");

  wrapper.classList.add("message-wrapper");

  // ---------------------------------------------
  // Label
  // ---------------------------------------------

  const label = document.createElement("div");

  label.classList.add("message-label");

  label.textContent = role === "user" ? "You" : "Assistant";

  // ---------------------------------------------
  // Content
  // ---------------------------------------------

  const content = document.createElement("div");

  content.classList.add("message-content");

  if (role === "assistant") {
    // Render Markdown
    content.innerHTML = marked.parse(text);
  } else {
    content.textContent = text;
  }

  wrapper.appendChild(label);

  wrapper.appendChild(content);

  // ---------------------------------------------
  // Assistant actions
  // ---------------------------------------------

  if (role === "assistant") {
    const actions = document.createElement("div");

    actions.classList.add("message-actions");

    const copyButton = document.createElement("button");

    copyButton.classList.add("copy-button");

    copyButton.textContent = "Copy";

    copyButton.addEventListener("click", async () => {
      await navigator.clipboard.writeText(text);

      copyButton.textContent = "Copied";

      setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1500);
    });

    actions.appendChild(copyButton);

    wrapper.appendChild(actions);

    // -----------------------------------------
    // Retrieved chunks
    // -----------------------------------------

    if (retrievedChunks && retrievedChunks.length > 0) {
      const panel = document.createElement("details");

      panel.classList.add("retrieval-panel");

      const summary = document.createElement("summary");

      summary.classList.add("retrieval-summary");

      summary.textContent = `🔍 Retrieved ${retrievedChunks.length} chunks from the book`;

      const retrievalContent = document.createElement("div");

      retrievalContent.classList.add("retrieval-content");

      retrievedChunks.forEach((chunk, index) => {
        const chunkElement = document.createElement("div");

        chunkElement.classList.add("chunk");

        chunkElement.textContent = `Chunk ${index + 1}\n\n${chunk}`;

        retrievalContent.appendChild(chunkElement);
      });

      panel.appendChild(summary);

      panel.appendChild(retrievalContent);

      wrapper.appendChild(panel);
    }
  }

  message.appendChild(wrapper);

  chatContainer.appendChild(message);

  chatContainer.scrollTop = chatContainer.scrollHeight;

  return content;
}

// =====================================================
// Loading Message
// =====================================================

function addLoadingMessage() {
  const message = document.createElement("div");

  message.classList.add("message", "assistant");

  const wrapper = document.createElement("div");

  wrapper.classList.add("message-wrapper");

  const loading = document.createElement("div");

  loading.classList.add("loading");

  loading.innerHTML = `
        <span>🔍 Searching the book</span>

        <span class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;

  wrapper.appendChild(loading);

  message.appendChild(wrapper);

  chatContainer.appendChild(message);

  chatContainer.scrollTop = chatContainer.scrollHeight;

  return message;
}

// =====================================================
// Send Message
// =====================================================

async function sendMessage(message) {
  // Remove welcome screen
  if (welcomeMessage) {
    welcomeMessage.remove();
  }

  // User message
  addMessage("user", message);

  // Loading
  const loadingMessage = addLoadingMessage();

  sendButton.disabled = true;

  messageInput.disabled = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        message: message,
        conversation_id: conversationId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // -----------------------------------------
    // Save conversation ID
    // -----------------------------------------

    conversationId = data.conversation_id;

    localStorage.setItem("conversation_id", conversationId);

    // -----------------------------------------
    // Remove loading
    // -----------------------------------------

    loadingMessage.remove();

    // -----------------------------------------
    // Assistant answer
    // -----------------------------------------

    addMessage("assistant", data.answer, data.retrieved_chunks);
  } catch (error) {
    console.error(error);

    loadingMessage.remove();

    addMessage(
      "assistant",
      "Sorry, something went wrong while processing your question.",
    );
  } finally {
    sendButton.disabled = false;

    messageInput.disabled = false;

    messageInput.focus();
  }
}

// =====================================================
// Form Submit
// =====================================================

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();

  if (!message) {
    return;
  }

  messageInput.value = "";

  await sendMessage(message);
});

// =====================================================
// Enter to Send
// =====================================================

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();

    chatForm.requestSubmit();
  }
});

// =====================================================
// Auto Resize Textarea
// =====================================================

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";

  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
});

// =====================================================
// New Chat
// =====================================================

newChatButton.addEventListener("click", async () => {
  try {
    if (conversationId) {
      await fetch(`/conversation/${conversationId}`, {
        method: "DELETE",
      });
    }
  } catch (error) {
    console.error(error);
  }

  // -----------------------------------------
  // Clear conversation ID
  // -----------------------------------------

  conversationId = null;

  localStorage.removeItem("conversation_id");

  // -----------------------------------------
  // Clear UI
  // -----------------------------------------

  chatContainer.innerHTML = `
            <div
                id="welcome-message"
                class="welcome-message"
            >

                <div class="welcome-icon">
                    📖
                </div>

                <h2>
                    Ask me about the book
                </h2>

                <p>
                    I can answer questions using
                    the indexed book content.
                </p>

            </div>
        `;

  messageInput.focus();
});

// =====================================================
// Example Questions
// =====================================================

document.addEventListener("click", (event) => {
  if (event.target.classList.contains("example-question")) {
    const question = event.target.textContent.trim();

    messageInput.value = question;

    chatForm.requestSubmit();
  }
});
