import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Set your Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set")

# URLs
BASE_URL = "https://brainlox.com/courses/category/technical"

# Configure Chroma DB path
CHROMA_DB_DIRECTORY = "chroma_db"
if not os.path.exists(CHROMA_DB_DIRECTORY):
    os.makedirs(CHROMA_DB_DIRECTORY)

# Initialize global variables for storage
conversation_chain = None
chat_history = []


def extract_and_process_data():
    """Extract data from Brainlox, create embeddings and store in Chroma vector database"""
    global conversation_chain

    # Step 1: Extract course data using LangChain's WebBaseLoader
    print("Loading data from Brainlox...")
    loader = WebBaseLoader(BASE_URL)
    data = loader.load()

    # Additional detailed extraction using BeautifulSoup for specific course details
    print("Extracting detailed course information...")
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract course details
    courses = []
    course_elements = soup.select('.course-item')

    for course in course_elements:
        title_elem = course.select_one('.course-title')
        desc_elem = course.select_one('.course-description')
        link_elem = course.select_one('a')

        title = title_elem.text.strip() if title_elem else "Unknown Title"
        description = desc_elem.text.strip() if desc_elem else "No description available"
        link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else None

        if link and not link.startswith('http'):
            link = f"https://brainlox.com{link}"

        courses.append({
            "title": title,
            "description": description,
            "link": link
        })

    # Enhance the data from WebBaseLoader with the detailed information
    for i, course in enumerate(courses):
        if i < len(data):
            data[
                i].page_content += f"\nTitle: {course['title']}\nDescription: {course['description']}\nLink: {course['link']}"

    # Step 2: Split the documents into chunks
    print("Splitting content into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    splits = text_splitter.split_documents(data)

    # Step 3: Create embeddings and store in Chroma DB using Google Gemini
    print("Creating embeddings with Gemini and storing in Chroma DB...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Store in Chroma DB (persistent)
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )

    # Persist the vector store to disk
    vector_store.persist()

    # Create a conversational retrieval chain with Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    print("Data processing complete!")
    return {"status": "success", "courses_processed": len(courses)}


def load_from_chroma():
    """Load the vector store from Chroma DB if it exists"""
    global conversation_chain

    if os.path.exists(CHROMA_DB_DIRECTORY) and os.listdir(CHROMA_DB_DIRECTORY):
        try:
            print("Loading existing vector store from Chroma DB...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            # Load from existing Chroma DB
            vector_store = Chroma(
                persist_directory=CHROMA_DB_DIRECTORY,
                embedding_function=embeddings
            )

            # Create the conversation chain
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )

            return True
        except Exception as e:
            print(f"Error loading from Chroma DB: {e}")
            return False
    return False


class Index(Resource):
    def get(self):
        return jsonify({
            "status": "success",
            "message": "Brainlox Courses API is running",
            "endpoints": {
                "/": "GET - This help information",
                "/initialize": "GET - Extract data and create embeddings",
                "/chat": "POST - Chat with the data (requires JSON body with 'query')",
                "/reset": "POST - Reset conversation history"
            }
        })


class InitializeAPI(Resource):
    def get(self):
        try:
            # Try to load from existing DB first
            if load_from_chroma():
                return jsonify({"status": "success", "message": "Loaded existing data from Chroma DB"})

            # If loading failed or no DB exists, extract new data
            result = extract_and_process_data()
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


class Conversation(Resource):
    def post(self):
        global conversation_chain, chat_history

        if not conversation_chain:
            # Try to load from existing DB if not initialized
            if not load_from_chroma():
                return jsonify({"status": "error", "message": "API not initialized. Call /initialize first."}), 400

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Missing 'query' in request body"}), 400

        query = data['query']

        try:
            # Process query using the conversation chain
            response = conversation_chain({"question": query, "chat_history": chat_history})

            # Extract answer and sources
            answer = response['answer']
            sources = [doc.page_content for doc in response.get('source_documents', [])]

            # Update chat history
            chat_history.append((query, answer))
            if len(chat_history) > 10:  # Keep only last 10 exchanges for context
                chat_history = chat_history[-10:]

            return jsonify({
                "status": "success",
                "answer": answer,
                "sources": sources[:3]  # Limit to top 3 sources for brevity
            })

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


class ResetConversation(Resource):
    def post(self):
        global chat_history
        chat_history = []
        return jsonify({"status": "success", "message": "Conversation history reset"})


# Register resources
api.add_resource(Index, '/')
api.add_resource(InitializeAPI, '/initialize')
api.add_resource(Conversation, '/chat')
api.add_resource(ResetConversation, '/reset')

if __name__ == '__main__':
    # Try to load existing data on startup
    load_from_chroma()
    app.run(debug=True, port=5000)