# Constitutional AI Chat Application

A modern React chat interface for interacting with Indian Constitution and IPC (Indian Penal Code) AI agents.

## Features

- 🏛️ **Constitutional Law Expertise**: Ask questions about the Indian Constitution
- ⚖️ **IPC Knowledge**: Get information about Indian Penal Code sections
- 💬 **Real-time Chat**: Modern chat interface with typing indicators
- 💾 **Local Storage**: Chat history saved locally (MongoDB integration ready)
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🔄 **Offline Support**: Works offline with local storage
- 🎨 **Modern UI**: Clean, accessible design with smooth animations
- 🚀 **Future Ready**: Built for easy extension with login, voice input, etc.

## Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: Custom CSS with modern design principles
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Storage**: LocalStorage (MongoDB ready)
- **State Management**: React Hooks + Custom Hooks

## Project Structure

```
client/
├── src/
│   ├── components/          # React components
│   │   ├── ChatContainer.jsx    # Main chat wrapper
│   │   ├── ChatHeader.jsx       # Header with logo and actions
│   │   ├── MessageList.jsx      # Message list container
│   │   ├── MessageItem.jsx      # Individual message component
│   │   ├── TypingIndicator.jsx  # AI typing animation
│   │   └── ChatInput.jsx        # Message input with features
│   ├── hooks/               # Custom React hooks
│   │   └── useChat.js           # Chat state management
│   ├── services/            # Business logic
│   │   ├── apiService.js        # Backend API communication
│   │   └── storageService.js    # Local storage management
│   ├── utils/               # Utility functions
│   └── App.jsx              # Main app component
├── public/                  # Static assets
└── package.json            # Dependencies and scripts
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the client directory:
   ```bash
   cd client
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   # Create .env file based on .env.example
   cp .env.example .env
   ```

4. Configure environment variables in `.env`:
   ```env
   # API Configuration
   VITE_API_BASE_URL=http://localhost:8000
   
   # App Configuration
   VITE_APP_NAME=Constitutional AI Chat
   VITE_APP_VERSION=1.0.0
   ```

5. Start the development server:
   ```bash
   npm run dev
   ```

6. Open your browser and navigate to `http://localhost:5173`

## Configuration

### Environment Variables

Create a `.env` file in the client directory:

```env
# Required: Backend API URL
VITE_API_BASE_URL=http://localhost:8000

# Optional: App metadata
VITE_APP_NAME=Constitutional AI Chat
VITE_APP_VERSION=1.0.0

# Optional: Feature flags for future features
VITE_ENABLE_VOICE_INPUT=false
VITE_ENABLE_FILE_UPLOAD=false
VITE_ENABLE_DARK_MODE=false

# Optional: Debug settings
VITE_DEBUG_MODE=false
VITE_LOG_LEVEL=info
```

### API Integration

The app expects the backend API to have the following endpoints:

1. **Health Check** (GET `/health`):
   ```json
   {
     "status": "ok",
     "message": "Service is running"
   }
   ```

2. **Send Message** (POST `/chat`):
   ```json
   {
     "message": "What is Article 370?",
     "chat_history": [
       {
         "content": "Previous message",
         "sender": "user",
         "timestamp": "2024-01-01T00:00:00Z"
       }
     ],
     "timestamp": "2024-01-01T00:00:00Z"
   }
   ```

   Response:
   ```json
   {
     "message": "Article 370 was a provision...",
     "metadata": {
       "sources": ["Constitution of India"],
       "confidence": 0.95
     }
   }
   ```

## Features

### Current Features

- ✅ Modern chat interface
- ✅ Real-time message exchange
- ✅ Local storage for chat history
- ✅ Typing indicators
- ✅ Error handling and offline support
- ✅ Responsive design
- ✅ Sample questions for quick start
- ✅ Message timestamps and grouping
- ✅ Connection status indicators

### Future Features (Ready for Implementation)

- 🔄 User authentication and login
- 🔄 MongoDB integration for chat storage
- 🔄 Voice input support
- 🔄 File upload and document analysis
- 🔄 Dark mode theme
- 🔄 Message search and filtering
- 🔄 Export chat history
- 🔄 Multi-language support

## Customization

### Styling

The app uses custom CSS with CSS variables for easy theming. Key style files:

- `src/App.css` - Main application styles
- Component-specific styles are included in the main CSS file

### Adding New Features

The architecture is designed for easy extension:

1. **New Components**: Add to `src/components/`
2. **API Methods**: Extend `src/services/apiService.js`
3. **Storage**: Extend `src/services/storageService.js`
4. **State Logic**: Extend `src/hooks/useChat.js`

### Example: Adding a New API Endpoint

```javascript
// In src/services/apiService.js
async getDocuments() {
  try {
    const response = await this.axiosInstance.get('/documents');
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch documents');
  }
}
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style

- Use functional components with hooks
- Follow React best practices
- Use meaningful component and variable names
- Add proper PropTypes or TypeScript (future)

## Production Deployment

### Build for Production

```bash
npm run build
```

### Environment Variables for Production

```env
VITE_API_BASE_URL=https://your-backend-api.com
VITE_APP_NAME=Constitutional AI
VITE_DEBUG_MODE=false
```

### Deployment Options

- **Vercel**: `vercel --prod`
- **Netlify**: Connect repository and set build command to `npm run build`
- **Static Hosting**: Serve the `dist` folder after building

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes and demonstrates AI agent development skills.

## Support

For questions or issues, please refer to the project documentation or create an issue in the repository.
