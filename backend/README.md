# Constitutional AI Chat API

FastAPI backend for the Constitutional AI Chat application, providing endpoints to interact with Indian Constitution and IPC AI agents.

## Features

- 🤖 **AI Agent Integration**: Direct integration with Constitutional AI agents
- 💬 **Chat API**: RESTful endpoints for real-time chat functionality
- 📚 **Legal Expertise**: Specialized in Indian Constitution and IPC
- 🔍 **Chat History Management**: Intelligent context management (last 5 interactions)
- 🏥 **Health Monitoring**: Comprehensive health check endpoints
- 🔧 **Future Ready**: Prepared for MongoDB integration and authentication
- 📊 **Logging**: Comprehensive logging and monitoring
- 🌐 **CORS Enabled**: Ready for frontend integration

## Quick Start

### Prerequisites

- Python 3.8+
- All dependencies from `requirements.txt`

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   python start_server.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Access the API:
   - **API Base**: http://localhost:8000
   - **Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Core Endpoints

#### 🏥 Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Service is running",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "agent_status": "healthy"
}
```

#### 💬 Send Chat Message
```http
POST /api/v1/chat
```

**Request:**
```json
{
  "message": "What are fundamental rights in Indian Constitution?",
  "chat_history": [
    {
      "content": "Hello",
      "sender": "user",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "content": "Hello! How can I help you with Indian law today?",
      "sender": "agent",
      "timestamp": "2024-01-01T12:00:01Z"
    }
  ],
  "timestamp": "2024-01-01T12:00:30Z"
}
```

**Response:**
```json
{
  "message": "Fundamental rights are basic human rights guaranteed by the Indian Constitution under Part III (Articles 12-35)...",
  "metadata": {
    "model": "constitutional_ai_agent",
    "chat_history_length": 2,
    "agent_history_length": 2
  },
  "timestamp": "2024-01-01T12:00:35Z",
  "processing_time": 2.34
}
```

### Utility Endpoints

#### 📋 Chat History Format
```http
GET /api/v1/chat/history/format
```

#### 🔍 Detailed Health Check
```http
GET /api/v1/health/detailed
```

### Future Endpoints (Prepared)

- `GET /api/v1/chat/sessions` - Get user chat sessions
- `POST /api/v1/chat/sessions` - Create new chat session  
- `GET /api/v1/chat/sessions/{session_id}` - Get specific session
- `DELETE /api/v1/chat/sessions/{session_id}` - Delete session

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Application Settings
ENVIRONMENT=development
DEBUG=true

# CORS Settings (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Agent Settings
MAX_CHAT_HISTORY=5
AGENT_TIMEOUT=30

# Future: Database Settings
DATABASE_URL=mongodb://localhost:27017/constitutional_ai

# Future: Authentication
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
```

### Configuration Details

- **MAX_CHAT_HISTORY**: Number of previous interactions sent to AI (default: 5)
- **AGENT_TIMEOUT**: Timeout for agent responses in seconds (default: 30)
- **ALLOWED_ORIGINS**: Frontend URLs allowed for CORS

## Architecture

### Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── start_server.py         # Server startup script
├── requirements.txt        # Dependencies
├── app/
│   ├── __init__.py
│   ├── core/              # Core configuration
│   │   ├── config.py      # Settings and configuration
│   │   └── logging_config.py # Logging setup
│   ├── api/               # API layer
│   │   └── routes/        # API endpoints
│   │       ├── chat.py    # Chat endpoints
│   │       └── health.py  # Health endpoints
│   ├── models/            # Pydantic models
│   │   └── chat.py        # Request/response models
│   ├── services/          # Business logic
│   │   └── chat_service.py # Chat processing service
│   └── db/                # Future: Database layer
└── agent_system/          # Existing AI agent system
```

### Key Components

1. **FastAPI Application** (`main.py`): Main application with middleware and routing
2. **Chat Service** (`chat_service.py`): Handles agent integration and message processing
3. **Pydantic Models** (`models/chat.py`): Request/response validation
4. **Configuration** (`core/config.py`): Centralized settings management
5. **API Routes** (`api/routes/`): RESTful endpoint definitions

## Integration with Agent System

The API integrates with your existing agent system:

```python
from agent_system.ai_agent.agent_graph import run_agent

# Convert frontend chat history to agent format
agent_history = [
    {"role": "user", "content": "What is Article 370?"},
    {"role": "assistant", "content": "Article 370 was..."}
]

# Get agent response
response = run_agent(user_message, agent_history)
```

### Chat History Management

- Frontend sends complete chat history
- Backend limits to last **5 interactions** (configurable)
- Converts between frontend and agent formats automatically
- Maintains conversation context efficiently

## Development

### Running in Development

```bash
# Start with auto-reload
python start_server.py

# Or with custom settings
ENVIRONMENT=development uvicorn main:app --reload --port 8000
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Logging

Comprehensive logging is configured:

```python
from app.core.logging_config import get_logger

logger = get_logger("my_module")
logger.info("Processing request")
```

## Production Deployment

### Environment Setup

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### Running in Production

```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or direct uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Future Enhancements

### 🔐 Authentication System
- JWT token-based authentication
- User registration and login
- Role-based access control

### 📊 MongoDB Integration
- Persistent chat sessions
- User chat history
- Analytics and usage tracking

### 🚀 Advanced Features
- WebSocket support for real-time chat
- Rate limiting and throttling
- Caching layer for common queries
- Multi-language support

## Testing

### Manual Testing

```bash
# Test health check
curl http://localhost:8000/api/v1/health

# Test chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are fundamental rights?",
    "chat_history": []
  }'
```

### Integration with Frontend

The API is designed to work seamlessly with the React frontend:

```javascript
// Frontend API call
const response = await apiService.sendMessage(
  "What is Article 370?",
  chatHistory
);
```

## Error Handling

The API provides consistent error responses:

```json
{
  "detail": "Message cannot be empty",
  "error": "validation_error",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Support

For issues or questions:

1. Check the API documentation at `/docs`
2. Review logs for error details
3. Verify agent system configuration
4. Test health endpoints

---

**Built with FastAPI for the Constitutional AI Chat Application**
*Showcasing AI agent development and modern web API design* 