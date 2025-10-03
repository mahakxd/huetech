Mini Agentic Pipeline
Overview

This project is a Mini Agentic Pipeline that combines a small document knowledge base, a CSV tool for product lookups, and the Google Gemini API. The pipeline demonstrates how a large language model (LLM) can decide whether to:

Retrieve an answer from the document knowledge base

Call an external tool (CSV lookup)

Answer directly using Gemini

The application runs through a simple Gradio interface for testing and demonstration.

Setup Instructions

Install dependencies:

pip install -r requirements.txt


Prepare your documents in the docs/ folder. Example files:

ai_basics.txt
ml_vs_dl.txt
data_science.txt
cloud_computing.txt
iot_overview.txt
cybersecurity.txt
robotics.txt
software_engineering.txt


Add a prices.csv file with product data. Example:

product,price,currency
burger,120,INR
fries,60,INR
coke,40,INR
pizza,250,INR
sandwich,80,INR


Set your Gemini API key:

export GEMINI_API_KEY="your_api_key_here"


Run the application:

python app.py

Features

Retriever: Embeds documents using textembedding-gecko-001 and finds the closest match with cosine similarity.

Actor: Looks up product prices from the CSV.

Reasoner: Uses Gemini (gemini-2.5-flash) to decide which path to follow.

Controller: Manages the workflow from knowledge base → CSV → fallback to Gemini.

User Interface: Gradio interface with sample questions and a large answer box.

Example Queries

“Tell me about cybersecurity” → Retrieved from knowledge base

“Price of pizza” → Retrieved from product database

“What is quantum computing?” → Answered directly by Gemini

Known Limitations

Limited to 8 documents and one CSV file.

Gemini fallback may generate inaccurate or hallucinated answers.

CSV is static, no live API integration.
