class StorageService {
  constructor() {
    this.CHAT_HISTORY_KEY = 'constitutional_chat_history';
    this.USER_PREFERENCES_KEY = 'constitutional_user_preferences';
    this.SESSION_KEY = 'constitutional_session';
  }

  // Chat History Management
  getChatHistory() {
    try {
      const history = localStorage.getItem(this.CHAT_HISTORY_KEY);
      return history ? JSON.parse(history) : [];
    } catch (error) {
      console.error('Error retrieving chat history:', error);
      return [];
    }
  }

  saveChatHistory(history) {
    try {
      localStorage.setItem(this.CHAT_HISTORY_KEY, JSON.stringify(history));
      return true;
    } catch (error) {
      console.error('Error saving chat history:', error);
      return false;
    }
  }

  addMessage(message) {
    try {
      const history = this.getChatHistory();
      const newMessage = {
        id: this.generateMessageId(),
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
      };
      
      history.push(newMessage);
      this.saveChatHistory(history);
      return newMessage;
    } catch (error) {
      console.error('Error adding message:', error);
      return null;
    }
  }

  clearChatHistory() {
    try {
      localStorage.removeItem(this.CHAT_HISTORY_KEY);
      return true;
    } catch (error) {
      console.error('Error clearing chat history:', error);
      return false;
    }
  }

  // User Preferences Management
  getUserPreferences() {
    try {
      const preferences = localStorage.getItem(this.USER_PREFERENCES_KEY);
      return preferences ? JSON.parse(preferences) : {
        theme: 'light',
        language: 'en',
        autoSave: true,
        showTimestamps: true,
      };
    } catch (error) {
      console.error('Error retrieving user preferences:', error);
      return {};
    }
  }

  saveUserPreferences(preferences) {
    try {
      const currentPreferences = this.getUserPreferences();
      const updatedPreferences = { ...currentPreferences, ...preferences };
      localStorage.setItem(this.USER_PREFERENCES_KEY, JSON.stringify(updatedPreferences));
      return true;
    } catch (error) {
      console.error('Error saving user preferences:', error);
      return false;
    }
  }

  // Session Management
  getSession() {
    try {
      const session = localStorage.getItem(this.SESSION_KEY);
      return session ? JSON.parse(session) : null;
    } catch (error) {
      console.error('Error retrieving session:', error);
      return null;
    }
  }

  saveSession(sessionData) {
    try {
      const session = {
        ...sessionData,
        lastActivity: new Date().toISOString(),
      };
      localStorage.setItem(this.SESSION_KEY, JSON.stringify(session));
      return true;
    } catch (error) {
      console.error('Error saving session:', error);
      return false;
    }
  }

  clearSession() {
    try {
      localStorage.removeItem(this.SESSION_KEY);
      return true;
    } catch (error) {
      console.error('Error clearing session:', error);
      return false;
    }
  }

  // Utility Methods
  generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  exportChatHistory() {
    try {
      const history = this.getChatHistory();
      const dataStr = JSON.stringify(history, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `chat_history_${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      
      URL.revokeObjectURL(url);
      return true;
    } catch (error) {
      console.error('Error exporting chat history:', error);
      return false;
    }
  }

  getStorageUsage() {
    try {
      let totalSize = 0;
      for (let key in localStorage) {
        if (Object.prototype.hasOwnProperty.call(localStorage, key) && key.startsWith('constitutional_')) {
          totalSize += localStorage[key].length;
        }
      }
      return {
        totalSize,
        formattedSize: this.formatBytes(totalSize),
        chatHistorySize: localStorage.getItem(this.CHAT_HISTORY_KEY)?.length || 0,
      };
    } catch (error) {
      console.error('Error calculating storage usage:', error);
      return { totalSize: 0, formattedSize: '0 B', chatHistorySize: 0 };
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Future: MongoDB integration methods (placeholders)
  async syncWithMongoDB() {
    // Placeholder for future MongoDB sync functionality
    console.log('MongoDB sync not implemented yet');
    return false;
  }

  async backupToCloud() {
    // Placeholder for future cloud backup functionality
    console.log('Cloud backup not implemented yet');
    return false;
  }
}

export default new StorageService(); 