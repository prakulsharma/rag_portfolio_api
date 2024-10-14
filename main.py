import os
import ollama
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Define the persist directory for ChromaDB
persist_directory = "./tmp/chromadb"


# Load documents and set up embeddings and vector store
def load_documents(folder_path: str):
    logging.info(f"Loading documents from {folder_path}...")
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r") as file:
                content = file.read()
                docs.append(
                    Document(page_content=content, metadata={"source": filename})
                )
    logging.info(f"Loaded {len(docs)} documents.")
    return docs


# Initialize the embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Initialize the vectorstore based on whether it's the first run or a subsequent one
if os.path.exists(persist_directory):
    logging.info("Loading vectorstore from persisted directory...")
    vectorstore = Chroma(
        collection_name="test",
        persist_directory=persist_directory,  # Load persisted data
        embedding_function=embedding_model,
    )
else:
    # Load and chunk the documents
    folder_path = "./data"
    documents = load_documents(folder_path)

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Generated {len(chunks)} chunks from the documents.")

    logging.info("Creating vectorstore and embedding documents...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name="test",
        embedding=embedding_model,  # Use embedding model directly
        persist_directory=persist_directory,  # Specify the directory to persist data
    )

retriever = vectorstore.as_retriever()

logging.info("Vectorstore initialized and retriever set up.")


# Combine retrieved documents into a context string
def combine_docs(docs):
    logging.info(f"Combining {len(docs)} retrieved documents into context...")
    combined_text = ""
    for doc in docs:
        combined_text += f"Source: {doc.metadata['source']}\n\n{doc.page_content}\n\n"
    return combined_text


# Chatbot interaction (search + generation)
def query_search(question: str, context: str, style: str):
    logging.info(f"Generating {style} response with Ollama...")

    # Define different prompt styles
    prompt_templates = {
        "concise": (
            f"Answer the following question as briefly and clearly as possible.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Answer concisely."
        ),
        "detailed": (
            f"Provide a detailed and thorough answer to the following question.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Answer in a detailed manner, covering all relevant points."
        ),
        "casual": (
            f"Respond to the following question in a friendly and conversational tone.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Make the answer informal and engaging."
        ),
    }

    # Instruction to handle irrelevant questions
    relevance_instruction = (
        f"Do not mention the presence of .txt files."
        f"If the question is unrelated to Prakul's experience, biography, or projects, respond with:\n"
        f'"I can only provide information about Prakul\'s experience, biography, or projects. Please ask something related."\n\n'
        f"Otherwise, answer the question according to the style and context provided."
    )

    # Get the appropriate prompt template based on the style
    formatted_prompt = (
        prompt_templates.get(style, prompt_templates[style])
        + "\n\n"
        + relevance_instruction
    )

    # Call Ollama's LLM to generate the response
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": formatted_prompt}]
    )

    if response and "message" in response and response["message"].get("content"):
        logging.info("Response generated successfully.")
        return response["message"]["content"]

    logging.warning("No response generated, fallback message returned.")
    return "Sorry, I couldn't generate a response."


def rag_chain(question, style="concise"):
    try:
        logging.info(f"Received question: {question} with style: {style}")
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            logging.warning("No relevant information found.")
            return "Sorry, I couldn't find any relevant information."
        formatted_context = combine_docs(retrieved_docs)
        return query_search(question, formatted_context, style)
    except Exception as e:
        logging.error(f"Error during retrieval or generation: {str(e)}")
        return f"Error in retrieval or generation: {str(e)}"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    # Pass the question from the request to the rag_chain function
    result = rag_chain(request.question)

    # Return the response as JSON
    return QueryResponse(answer=result)


# if __name__ == "__main__":
#     print("Greetings from PrakulAI. Ask me anything about myself. Type 'exit' to stop.")

#     while True:
#         query = input("You: ").strip()
#         if query.lower() == "exit":
#             break

#         result = rag_chain(query)
#         print(f"Bot: {result}\n")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
