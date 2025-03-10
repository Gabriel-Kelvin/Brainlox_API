# Brainlox Courses API

A Flask RESTful API that extracts technical course data from Brainlox, creates embeddings, and provides a conversational interface to interact with the course information.

## Overview

This application crawls the Brainlox technical courses page, extracts course information, creates vector embeddings using Google's Gemini AI, and stores them in a persistent Chroma vector database. The API then allows users to have natural language conversations about the available courses.

## Features

- Extracts detailed course information from Brainlox
- Creates and stores embeddings in a persistent Chroma DB
- Provides a conversational interface using Gemini AI
- Maintains conversation history for context
- Includes API endpoints for initialization, chat, and conversation reset

## Requirements

- Python 3.8+
- Flask and Flask-RESTful
- LangChain libraries
- Google Gemini API key
- ChromaDB
- BeautifulSoup4 and Requests

## Installation

1. Clone the repository or download the source code

2. Install the required dependencies:
   ```bash
   pip install flask flask-restful langchain langchain-google-genai langchain-community python-dotenv chromadb beautifulsoup4 requests
   ```

3. Create a `.env` file in the project root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. The server will run on `http://127.0.0.1:5000` by default

3. API Endpoints:
   - `GET /` - Displays API information and available endpoints
   - `GET /initialize` - Extracts data from Brainlox and creates embeddings (only needed once)
   - `POST /chat` - Sends a query to chat with the course data
   - `POST /reset` - Resets the conversation history

## Example Requests

### Initialize the API
```bash
curl http://127.0.0.1:5000/initialize
```

### Chat with the API
```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What technical courses are available?"}'
```

### Reset Conversation
```bash
curl -X POST http://127.0.0.1:5000/reset
```

## Data Persistence

The application stores embeddings in a local `chroma_db` directory. After initialization, data persists between application restarts, so there's no need to re-extract and re-embed data each time you start the app.

## Purpose

This API serves as an intelligent interface to Brainlox's technical course offerings. It can be used to:
- Provide a natural language search interface for courses
- Create a chatbot for the Brainlox website
- Build a recommendation system based on user queries
- Integrate course information with other applications

## Technical Implementation

- Web scraping using LangChain's WebBaseLoader and BeautifulSoup
- Vector embeddings using Google's Gemini AI
- Vector storage using ChromaDB
- Conversational retrieval chain from LangChain
- RESTful API built with Flask and Flask-RESTful