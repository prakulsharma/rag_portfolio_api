# RAG Portfolio API
Welcome to the RAG Portfolio API (being used at https://prakulsharma.github.io/), a demonstration of using Retrieval-Augmented Generation (RAG) to interactively answer questions and present information about my projects and experience. This project utilizes state-of-the-art natural language processing techniques combined with a well-structured document retrieval system to provide concise, detailed, or conversational responses based on your queries.

## What is RAG?
Retrieval-Augmented Generation (RAG) is a hybrid approach that combines information retrieval with generative text models. It augments the generative model's output with relevant, contextually retrieved documents, resulting in more accurate and context-aware responses. This technique is particularly useful in scenarios where we want to provide precise answers based on a pre-defined set of documents, such as personal portfolios, knowledge bases, or FAQs.

## Project Overview
In this project, I have built a FastAPI-based web service that serves as a dynamic portfolio assistant. You can ask questions about my experience, projects, and expertise, and the API responds with relevant information retrieved from a document store.

The key feature of this implementation is its use of RAG to retrieve relevant documents from a vector store and generate answers using a language model hosted on Hugging Face’s Inference API.

## Features
- Dynamic Document Retrieval
- Adaptive Question Responses
- Token Management with Cooldown
- Flexible Deployment
  
## Technologies Used
- FastAPI
- LangChain
- ChromaDB
- Hugging Face Inference API
- Python-Dotenv
- Uvicorn

## Implementation Details
The project works in the following way:

- Document Loading and Splitting: Loads text documents from a specified directory, splits them into manageable chunks, and stores them in a vector database.
- Dynamic Embedding Generation: Uses the HuggingFaceEmbeddings model to generate vector representations of document chunks.
- Document Retrieval: Retrieves relevant document chunks based on query similarity using ChromaDB.
- RAG Integration: Combines retrieved documents into context and sends them to the Hugging Face language model to generate responses.
- Token Management: Rotates between multiple API tokens to handle Hugging Face API rate limits, ensuring consistent service availability.

## Demo
https://github.com/user-attachments/assets/1c039d6d-361f-4923-b59d-6e11ff8ef858

This project showcases the power of combining retrieval-based methods with generative models to create an interactive and informative portfolio. It has been a fascinating journey to bring this technology to life, and I hope it provides useful insights into my work. Feel free to explore the code, ask questions, and check out the GitHub repository for more details.

Happy exploring!

If you have any feedback or suggestions, please feel free to open an issue or get in touch. 😊

