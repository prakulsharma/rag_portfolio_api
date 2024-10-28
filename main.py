import os
import logging
import time
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uvicorn

# initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

load_dotenv()

# load allowed origins from environment
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve all Hugging Face tokens from the environment variable
HF_TOKENS = os.getenv("HF_TOKENS", "").split(",")
if not HF_TOKENS:
    raise ValueError("No Hugging Face tokens found in the environment variable 'HF_TOKENS'")

# Create config folder if it doesn't exist
config_folder = "./config"
Path(config_folder).mkdir(parents=True, exist_ok=True)

# Path to JSON file for storing token usage times
token_usage_file = os.path.join(config_folder, "token_usage.json")


# Load token usage from JSON file if it exists
def load_token_usage():
    if os.path.exists(token_usage_file):
        with open(token_usage_file, "r") as file:
            return json.load(file)
    return {}


# Save token usage to JSON file
def save_token_usage(token_usage_data):
    with open(token_usage_file, "w") as file:
        json.dump(token_usage_data, file)


# Initialize the token usage dictionary
token_usage = load_token_usage()

# Ensure all tokens in the environment are in the token usage dictionary
for token in HF_TOKENS:
    if token not in token_usage:
        token_usage[token] = {"count": 0, "last_used": 0}

# Save the initialized token usage
save_token_usage(token_usage)

# Track the usage of tokens
token_index = 0


def get_current_token():
    """Return the current token based on the index and rotate if needed."""
    global token_index
    current_time = time.time()

    for _ in range(len(HF_TOKENS)):
        current_token = HF_TOKENS[token_index]
        last_used = token_usage.get(current_token, {}).get("last_used", 0)
        count = token_usage.get(current_token, {}).get("count", 0)

        if count < 3 or (current_time - last_used) > 3600:
            # Reset count if more than an hour has passed
            if (current_time - last_used) > 3600:
                token_usage[current_token]["count"] = 0

            return current_token
        else:
            # Rotate to the next available token
            token_index = (token_index + 1) % len(HF_TOKENS)

    # If all tokens are in cooldown, return None
    return None


def switch_client():
    """Switch the client based on the next available token."""
    global client
    current_token = get_current_token()
    if current_token is None:
        return None
    client = InferenceClient(api_key=current_token, timeout=10)
    logging.info(f"Switched to token: {current_token}")
    return current_token


# Initialize client with the first available token
if switch_client() is None:
    raise ValueError("No available tokens to initialize the client.")

# define the persist directory for chromadb
persist_directory = "./chromadb"


def load_documents(folder_path: str):
    """
    Load text documents from the specified folder path.

    Args:
        folder_path (str): The path to the folder containing text files.

    Returns:
        List[Document]: A list of Document objects containing the content and metadata of each file.
    """
    try:
        logging.info(f"Loading documents from '{folder_path}'...")
        docs = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r") as file:
                    content = file.read()
                    docs.append(
                        Document(page_content=content, metadata={"source": filename})
                    )
        logging.info(f"Loaded {len(docs)} documents successfully.")
        return docs
    except FileNotFoundError:
        logging.error(f"Folder '{folder_path}' not found.")
        raise FileNotFoundError(f"Folder '{folder_path}' not found.")
    except Exception as e:
        logging.error(f"Error while loading documents: {str(e)}")
        raise RuntimeError(f"Error while loading documents: {str(e)}")


# initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initialize the vectorstore based on whether it's the first run or a subsequent one
try:
    if os.path.exists(persist_directory):
        logging.info("Loading vectorstore from persisted directory...")
        vectorstore = Chroma(
            collection_name="test",
            persist_directory=persist_directory,  # load persisted data
            embedding_function=embedding_model,
        )
    else:
        # load and chunk the documents
        folder_path = "./data"
        documents = load_documents(folder_path)

        logging.info("Splitting documents into manageable chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Generated {len(chunks)} document chunks.")

        logging.info("Creating vectorstore and embedding documents...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            collection_name="test",
            embedding=embedding_model,  # use embedding model directly
            persist_directory=persist_directory,  # specify the directory to persist data
        )
    retriever = vectorstore.as_retriever()
    logging.info("Vectorstore initialized and retriever set up successfully.")
except Exception as e:
    logging.error(f"Error initializing vectorstore: {str(e)}")
    raise RuntimeError(f"Error initializing vectorstore: {str(e)}")


def combine_docs(docs):
    """
    Combine multiple Document objects into a single formatted string.

    Args:
        docs (List[Document]): A list of Document objects to be combined.

    Returns:
        str: A formatted string combining the content and metadata of each document.
    """
    try:
        logging.info(f"Combining {len(docs)} retrieved documents into context...")
        combined_text = ""
        for doc in docs:
            combined_text += f"Source: {doc.metadata['source']}\n\n{doc.page_content}\n\n"
        return combined_text
    except Exception as e:
        logging.error(f"Error combining documents: {str(e)}")
        raise RuntimeError(f"Error combining documents: {str(e)}")


def query_search(question: str, context: str, style: str):
    """
    Generate a response based on the given question, context, and style.

    Args:
        question (str): The question to be answered.
        context (str): The context information to help generate the response.
        style (str): The style of the response ("concise", "detailed", "casual").

    Returns:
        str: The generated response or a fallback message if no response is produced.
    """
    global client, token_usage

    current_token = get_current_token()
    if current_token is None:
        cooldown_time = 3600 - (time.time() - min(token_usage[token]["last_used"] for token in HF_TOKENS))
        cooldown_minutes = int(cooldown_time // 60)
        cooldown_seconds = int(cooldown_time % 60)
        return (f"All my tokens are in their cooldown period since I am using the HuggingFace Inference API. "
                f"Try again in about {cooldown_minutes} minutes.")

    try:
        # Update token usage before making the API call
        token_usage[current_token]["count"] += 1
        token_usage[current_token]["last_used"] = time.time()
        save_token_usage(token_usage)

        logging.info(f"Generating response with style: '{style}' using token '{current_token}'...")

        # define different prompt styles
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

        relevance_instruction = (
            f"Do not mention the presence of .txt files."
            f"If the question is unrelated to Prakul's experience, biography, or projects, respond with:\n"
            f'"I can only provide information about Prakul\'s experience, biography, or projects. Please ask something related."\n\n'
            f"Otherwise, answer the question according to the style and context provided. Give the answer directly without any other verbose."
        )

        formatted_prompt = (
                prompt_templates.get(style, prompt_templates['concise'])
                + "\n\n"
                + relevance_instruction
        )

        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": formatted_prompt}]
        )

        output = response.choices[0].message.content

        if output:
            logging.info(f"Response generated successfully.")
            return output

        logging.warning("No response generated, returning fallback message.")
        return "Sorry, I couldn't generate a response."
    except Exception as e:
        logging.error(f"Error generating response with current token: {str(e)}")
        # Switch to the next token and retry
        switch_client()
        return query_search(question, context, style)


def rag_chain(question, style="concise"):
    """
    Process a question by retrieving relevant documents and generating a response.

    Args:
        question (str): The question to be answered.
        style (str): The style of the response ("concise", "detailed", "casual").

    Returns:
        str: The generated response or an error message if an issue occurs.
    """
    try:
        logging.info(f"Received question: '{question}' with style: '{style}'")
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            logging.warning("No relevant information found.")
            return "Sorry, I couldn't find any relevant information."
        formatted_context = combine_docs(retrieved_docs)
        return query_search(question, formatted_context, style)
    except Exception as e:
        logging.error(f"Error during retrieval or response generation: {str(e)}")
        return f"Error in retrieval or generation: {str(e)}"

class QuestionValidationError(HTTPException):
    """
    Custom exception for question validation errors.
    """
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

class QueryRequest(BaseModel):
    """
    A model representing a request for querying information.
    """
    question: str

    @field_validator("question", mode="before")
    def check_question_length(cls, v):
        if not v.strip():
            raise QuestionValidationError(detail="Your question cannot be empty. Please enter a valid question.")
        if len(v) > 100:
            raise QuestionValidationError(detail="Your question is too long. Please limit it to 100 characters.")
        return v


class QueryResponse(BaseModel):
    """
    A model representing the response to a query request.
    """
    answer: str


@app.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to process a query request and return a generated response.

    Args:
        request (QueryRequest): The query request containing the question.

    Returns:
        QueryResponse: The response with the generated answer.
    """
    try:
        result = rag_chain(request.question)
        return QueryResponse(answer=result)
    except QuestionValidationError as ve:
        # Catch the custom validation error and return the message
        logging.error(f"Validation error: {ve.detail}")
        raise HTTPException(status_code=400, detail=ve.detail)
    except Exception as e:
        logging.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# if __name__ == "__main__":
#     print("Greetings from PrakulAI. Ask me anything about myself. Type 'exit' to stop.")
#
#     while True:
#         query = input("You: ").strip()
#         if query.lower() == "exit":
#             break
#
#         result = rag_chain(query)
#         print(f"Bot: {result}\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
