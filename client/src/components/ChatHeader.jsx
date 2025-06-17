import React from 'react';
import { 
  MessageSquare, 
  Download, 
  Trash2, 
  RefreshCw, 
  Scale,
  BookOpen
} from 'lucide-react';

const ChatHeader = ({ 
  connectionStatus, 
  onClearChat, 
  onExportChat, 
  onCheckConnection, 
  messageCount 
}) => {
  const handleClearChat = () => {
    if (messageCount > 0) {
      const confirmed = window.confirm(
        'Are you sure you want to clear all chat history? This action cannot be undone.'
      );
      if (confirmed) {
        onClearChat();
      }
    }
  };

  return (
    <header className="chat-header">
      <div className="header-left">
        <div className="logo">
          <Scale className="logo-icon" size={32} />
          <div className="logo-text">
            <h1>AI Legal Assistant</h1>
            <p>Indian Constitution & IPC Expert</p>
          </div>
        </div>
      </div>
      
      <div className="header-center">
        <div className="topic-indicators">
          <div className="topic-item">
            <BookOpen size={16} />
            <span>Constitution of India</span>
          </div>
          <div className="topic-item">
            <Scale size={16} />
            <span>Indian Penal Code</span>
          </div>
        </div>
      </div>
      
      <div className="header-right">
        <div className="chat-stats">
          <MessageSquare size={16} />
          <span>{messageCount} messages</span>
        </div>
        
        <div className="header-actions">
          <button
            className="header-btn"
            onClick={onCheckConnection}
            title="Check connection"
            disabled={connectionStatus === 'connecting'}
          >
            <RefreshCw 
              size={18} 
              className={connectionStatus === 'connecting' ? 'spinning' : ''} 
            />
          </button>
          
          <button
            className="header-btn"
            onClick={onExportChat}
            title="Export chat history"
            disabled={messageCount === 0}
          >
            <Download size={18} />
          </button>
          
          <button
            className="header-btn danger"
            onClick={handleClearChat}
            title="Clear chat history"
            disabled={messageCount === 0}
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>
    </header>
  );
};

export default ChatHeader; 