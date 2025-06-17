import React, { useEffect, useRef } from 'react';
import MessageItem from './MessageItem';
import TypingIndicator from './TypingIndicator';

const MessageList = ({ messages, isTyping }) => {
  const messagesEndRef = useRef(null);
  const containerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return date.toLocaleDateString('en-IN', { weekday: 'long' });
    } else {
      return date.toLocaleDateString('en-IN', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      });
    }
  };

  const groupMessagesByDate = (messages) => {
    const groups = {};
    
    messages.forEach(message => {
      const dateKey = formatDate(message.timestamp);
      if (!groups[dateKey]) {
        groups[dateKey] = [];
      }
      groups[dateKey].push(message);
    });

    return groups;
  };

  const messageGroups = groupMessagesByDate(messages);

  return (
    <div className="message-list" ref={containerRef}>
      {messages.length === 0 ? (
        <div className="welcome-message">
          <div className="welcome-content">
            <h2>Welcome to Constitutional AI</h2>
            <p>
              I'm here to help you understand the Indian Constitution and Indian Penal Code. 
              Ask me anything about:
            </p>
            <ul>
              <li>Constitutional articles and amendments</li>
              <li>Fundamental rights and duties</li>
              <li>Indian Penal Code sections</li>
              <li>Legal interpretations and precedents</li>
              <li>Government structure and procedures</li>
            </ul>
            <p>Start by typing your question below!</p>
          </div>
        </div>
      ) : (
        Object.entries(messageGroups).map(([date, dateMessages]) => (
          <div key={date} className="message-group">
            <div className="date-separator">
              <span>{date}</span>
            </div>
            {dateMessages.map((message) => (
              <MessageItem 
                key={message.id} 
                message={message} 
              />
            ))}
          </div>
        ))
      )}
      
      {isTyping && <TypingIndicator />}
      
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList; 