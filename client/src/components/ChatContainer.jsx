import React from 'react';
import { useChat } from '../hooks/useChat';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import ChatHeader from './ChatHeader';
import { AlertCircle, Wifi, WifiOff } from 'lucide-react';

const ChatContainer = () => {
  const {
    messages,
    isLoading,
    error,
    isTyping,
    connectionStatus,
    sendMessage,
    clearChat,
    exportChat,
    cancelCurrentRequest,
    checkConnection,
  } = useChat();

  return (
    <div className="chat-container">
      <ChatHeader
        connectionStatus={connectionStatus}
        onClearChat={clearChat}
        onExportChat={exportChat}
        onCheckConnection={checkConnection}
        messageCount={messages.length}
      />
      
      <div className="chat-content">
        {error && (
          <div className="error-banner">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}
        
        <MessageList 
          messages={messages} 
          isTyping={isTyping}
        />
      </div>
      
      <ChatInput
        onSendMessage={sendMessage}
        isLoading={isLoading}
        isConnected={connectionStatus === 'connected'}
        onCancel={cancelCurrentRequest}
      />
      
      {connectionStatus === 'disconnected' && (
        <div className="connection-status offline">
          <WifiOff size={16} />
          <span>Offline Mode - Messages saved locally</span>
        </div>
      )}
      
      {connectionStatus === 'connected' && (
        <div className="connection-status online">
          <Wifi size={16} />
          <span>Connected to Constitutional AI Agent</span>
        </div>
      )}
    </div>
  );
};

export default ChatContainer; 