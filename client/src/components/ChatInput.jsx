import React, { useState, useRef, useEffect } from 'react';
import { Send, X, Mic, Paperclip } from 'lucide-react';

const ChatInput = ({ onSendMessage, isLoading, isConnected, onCancel }) => {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputChange = (e) => {
    setMessage(e.target.value);
    
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  const handleCancel = () => {
    if (onCancel) {
      onCancel();
    }
  };

  // Auto-focus on mount
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  // Sample questions for quick start
  const sampleQuestions = [
    "What are the fundamental rights in the Indian Constitution?",
    "Explain Article 370 of the Indian Constitution",
    "What is Section 302 of the Indian Penal Code?",
    "Tell me about the Right to Information Act"
  ];

  const handleSampleClick = (question) => {
    setMessage(question);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <div className="chat-input-container">
      {/* {!isConnected && (
        <div className="offline-notice">
          <span>⚠️ Offline mode - Your messages will be saved locally</span>
        </div>
      )} */}
      
      {message === '' && !isFocused && (
        <div className="sample-questions">
          <p className="sample-label">Try asking:</p>
          <div className="sample-buttons">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                className="sample-button"
                onClick={() => handleSampleClick(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className={`input-wrapper ${isFocused ? 'focused' : ''}`}>
          <div className="input-actions-left">
            {/* Future: File attachment */}
            <button
              type="button"
              className="input-action-btn"
              title="Attach file (Coming soon)"
              disabled
            >
              <Paperclip size={18} />
            </button>
          </div>
          
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={
              isConnected 
                ? "Ask about Indian Constitution or IPC..." 
                : "Ask about Indian Constitution or IPC (offline mode)..."
            }
            disabled={isLoading}
            rows={1}
            className="message-input"
          />
          
          <div className="input-actions-right">
            {/* Future: Voice input */}
            <button
              type="button"
              className="input-action-btn"
              title="Voice input (Coming soon)"
              disabled
            >
              <Mic size={18} />
            </button>
            
            {isLoading ? (
              <button
                type="button"
                onClick={handleCancel}
                className="input-action-btn cancel"
                title="Cancel"
              >
                <X size={18} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!message.trim() || isLoading}
                className="send-button"
                title="Send message"
              >
                <Send size={18} />
              </button>
            )}
          </div>
        </div>
        
        <div className="input-footer">
          <span className="input-hint">
            Press Enter to send, Shift+Enter for new line
          </span>
          {isLoading && (
            <span className="loading-text">
              Consulting AI Legal Assistant...
            </span>
          )}
        </div>
      </form>
    </div>
  );
};

export default ChatInput; 