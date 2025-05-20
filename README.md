# Constitution of India API

A FastAPI-based backend service for accessing and managing the Constitution of India.

## Setup

1. Create and activate virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./constitution.db
```

4. Run the application:
```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Project Structure

```
.
├── backend/
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration settings
│   └── models.py        # Database models
├── venv/                # Virtual environment
├── requirements.txt     # Project dependencies
└── README.md           # This file
``` 