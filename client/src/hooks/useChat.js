import { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/apiService';
import storageService from '../services/storageService';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const abortControllerRef = useRef(null);

  // Load chat history on mount
  useEffect(() => {
    const savedHistory = storageService.getChatHistory();
    setMessages(savedHistory);
    
    // Check backend connection
    checkConnection();
  }, []);

  // Save messages to local storage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      storageService.saveChatHistory(messages);
    }
  }, [messages]);

  const checkConnection = async () => {
    try {
      await apiService.healthCheck();
      setConnectionStatus('connected');
    } catch {
      setConnectionStatus('disconnected');
      console.warn('Backend not available, working in offline mode');
    }
  };

  const addMessage = useCallback((message) => {
    const newMessage = {
      id: storageService.generateMessageId(),
      ...message,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, newMessage]);
    return newMessage;
  }, []);

  const sendMessage = useCallback(async (content, type = 'text') => {
    if (!content.trim()) return;

    setError(null);
    setIsLoading(true);

    // Add user message
    addMessage({
      content: content.trim(),
      sender: 'user',
      type,
    });

    try {
      // Prepare chat history for API (exclude current message)
      const chatHistory = messages.map(msg => ({
        content: msg.content,
        sender: msg.sender,
        timestamp: msg.timestamp,
      }));

      // Cancel any previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      setIsTyping(true);

      // Call API
      const response = await apiService.sendMessage(content, chatHistory);
      
      // Add agent response
      addMessage({
        content: response.message || response.response || 'I received your message.',
        sender: 'agent',
        type: 'text',
        metadata: response.metadata || {},
      });

      setConnectionStatus('connected');
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }

      console.error('Error sending message:', error);
      setError(error.message);
      
      // Add error message
      addMessage({
        content: `Sorry, I'm having trouble connecting to the server. ${error.message}`,
        sender: 'agent',
        type: 'error',
      });

      setConnectionStatus('disconnected');
    } finally {
      setIsLoading(false);
      setIsTyping(false);
      abortControllerRef.current = null;
    }
  }, [messages, addMessage]);

  const resendMessage = useCallback(async (messageId) => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1) return;

    const message = messages[messageIndex];
    if (message.sender !== 'user') return;

    // Remove the message and any subsequent messages
    setMessages(prev => prev.slice(0, messageIndex));
    
    // Resend the message
    await sendMessage(message.content, message.type);
  }, [messages, sendMessage]);

  const clearChat = useCallback(() => {
    setMessages([]);
    storageService.clearChatHistory();
    setError(null);
  }, []);

  const deleteMessage = useCallback((messageId) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId));
  }, []);

  const exportChat = useCallback(() => {
    return storageService.exportChatHistory();
  }, []);

  const cancelCurrentRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
      setIsTyping(false);
    }
  }, []);

  // Get formatted chat history for API
  const getChatHistoryForAPI = useCallback(() => {
    return messages.map(msg => ({
      content: msg.content,
      sender: msg.sender,
      timestamp: msg.timestamp,
    }));
  }, [messages]);

  // Get statistics
  const getStatistics = useCallback(() => {
    const userMessages = messages.filter(msg => msg.sender === 'user').length;
    const agentMessages = messages.filter(msg => msg.sender === 'agent').length;
    const totalMessages = messages.length;
    
    return {
      totalMessages,
      userMessages,
      agentMessages,
      averageResponseTime: 0, // TODO: Implement response time tracking
    };
  }, [messages]);

  return {
    // State
    messages,
    isLoading,
    error,
    isTyping,
    connectionStatus,
    
    // Actions
    sendMessage,
    resendMessage,
    clearChat,
    deleteMessage,
    exportChat,
    cancelCurrentRequest,
    checkConnection,
    
    // Utilities
    getChatHistoryForAPI,
    getStatistics,
  };
}; 