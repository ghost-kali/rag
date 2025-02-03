from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from brain import generate_quiz_questions, get_index_for_pdf

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client using the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Global variable to store the vectordb
vectordb = None

def create_vectordb(files, filenames):
    global vectordb
    vectordb = get_index_for_pdf(files, filenames, api_key)




@app.route('/quiz', methods=['POST'])
def get_quiz():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        sem = data.get('sem', '')
        dept = data.get('dept', '')
        level = data.get('level', '')
        subject = data.get('subject', '')
        unit = data.get('unit', '')
        dept = dept.upper()
        
        pdf_directory = f"D:/rag/pdfs/{dept}Sem-{sem}/{subject}/unit{unit}"
        print(f"Looking for PDFs in: {pdf_directory}")
        
        try:
            pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
            pdf_file_names = [os.path.basename(f) for f in pdf_files]
        except FileNotFoundError as e:
            return jsonify({"error": f"Directory not found: {pdf_directory}"}), 404
        except Exception as e:
            return jsonify({"error": f"Error accessing directory: {str(e)}"}), 500

        if not pdf_files:
            return jsonify({"error": "No PDF files found"}), 404

        try:
            with app.app_context():
                create_vectordb([open(f, 'rb').read() for f in pdf_files], pdf_file_names)
        except Exception as e:
            return jsonify({"error": f"Error creating vector database: {str(e)}"}), 500

        if not vectordb:
            return jsonify({"error": "Vector database initialization failed"}), 500

        search_results = vectordb.similarity_search("questions of the entire pdf?", k=3)
        pdf_extract = "\n".join([result.page_content for result in search_results])
        
        try:
            result = generate_quiz_questions(client=client, model="gpt-4o-2024-08-06", pdf_extract=pdf_extract)
            print(result)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Error generating questions: {str(e)}"}), 500

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
 
    app.run(debug=True)