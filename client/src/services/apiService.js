import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000, // 30 seconds timeout for agent responses
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for future authentication
    this.axiosInstance.interceptors.request.use(
      (config) => {
        // Future: Add authentication token here
        // const token = localStorage.getItem('authToken');
        // if (token) {
        //   config.headers.Authorization = `Bearer ${token}`;
        // }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  async sendMessage(message, chatHistory = []) {
    try {
      const response = await this.axiosInstance.post('/chat', {
        message,
        chat_history: chatHistory,
        timestamp: new Date().toISOString(),
      });
      
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        error.message || 
        'Failed to send message'
      );
    }
  }

  // Future: Authentication methods
  async login(credentials) {
    try {
      const response = await this.axiosInstance.post('/auth/login', credentials);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Login failed'
      );
    }
  }

  async logout() {
    try {
      await this.axiosInstance.post('/auth/logout');
      localStorage.removeItem('authToken');
    } catch (error) {
      console.error('Logout error:', error);
    }
  }

  // Health check for the backend
  async healthCheck() {
    try {
      const response = await this.axiosInstance.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Backend is not available: ${error.message}`);
    }
  }
}

export default new ApiService(); 