import React from 'react';
import { Bot } from 'lucide-react';

const TypingIndicator = () => {
  return (
    <div className="message-item agent-message typing-indicator">
      <div className="message-header">
        <div className="message-sender">
          <Bot size={20} className="message-icon agent" />
          <span className="sender-name">Constitutional AI</span>
        </div>
      </div>
      
      <div className="message-content">
        <div className="typing-animation">
          <div className="typing-dots">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
          <span className="typing-text">is typing...</span>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator; 