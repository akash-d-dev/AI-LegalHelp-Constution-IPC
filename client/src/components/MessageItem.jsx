import React from 'react';
import { User, Bot, AlertTriangle, Clock } from 'lucide-react';

const MessageItem = ({ message }) => {
  const { content, sender, timestamp, type } = message;

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getMessageIcon = () => {
    if (type === 'error') {
      return <AlertTriangle size={20} className="message-icon error" />;
    }
    
    return sender === 'user' ? (
      <User size={20} className="message-icon user" />
    ) : (
      <Bot size={20} className="message-icon agent" />
    );
  };

  const getMessageClass = () => {
    const baseClass = 'message-item';
    const senderClass = sender === 'user' ? 'user-message' : 'agent-message';
    const typeClass = type === 'error' ? 'error-message' : '';
    
    return `${baseClass} ${senderClass} ${typeClass}`.trim();
  };

  const renderContent = () => {
    if (type === 'error') {
      return (
        <div className="message-content error">
          <p>{content}</p>
        </div>
      );
    }

    return (
      <div className="message-content">
        <div className="message-text">
          {content.split('\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={getMessageClass()}>
      <div className="message-header">
        <div className="message-sender">
          {getMessageIcon()}
          <span className="sender-name">
            {sender === 'user' ? 'You' : 'Constitutional AI'}
          </span>
        </div>
        <div className="message-timestamp">
          <Clock size={12} />
          <span>{formatTime(timestamp)}</span>
        </div>
      </div>
      
      {renderContent()}
      
      <div className="message-actions">
        {/* Placeholder for future actions */}
      </div>
    </div>
  );
};

export default MessageItem; 