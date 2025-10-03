import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from google import genai

# Load Gemini API key from Hugging Face secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in Hugging Face Spaces secrets.")

client = genai.Client(api_key=GEMINI_API_KEY)

docs_folder = "docs/"
csv_file = "prices.csv"
embeddings_file = "doc_embeddings.pkl"

doc_names = [
    "ai_basics.txt",
    "ml_vs_dl.txt",
    "data_science.txt",
    "cloud_computing.txt",
    "iot_overview.txt",
    "cybersecurity.txt",
    "robotics.txt",
    "software_engineering.txt"
]

def load_or_create_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            doc_texts, doc_embeddings = pickle.load(f)
        return doc_texts, doc_embeddings

    doc_texts = []
    for name in doc_names:
        try:
            with open(f"{docs_folder}{name}", "r", encoding="utf-8") as f:
                doc_texts.append(f.read())
        except FileNotFoundError:
            doc_texts.append("")

    doc_embeddings = []
    for txt in doc_texts:
        if txt.strip():
            try:
                emb_resp = client.models.embed_text(
                    model="textembedding-gecko-001",
                    input=[txt]
                )
                doc_embeddings.append(emb_resp.data[0].embedding)
            except Exception:
                doc_embeddings.append(np.zeros(1024))
        else:
            doc_embeddings.append(np.zeros(1024))

    doc_embeddings = np.array(doc_embeddings)
    with open(embeddings_file, 'wb') as f:
        pickle.dump((doc_texts, doc_embeddings), f)

    return doc_texts, doc_embeddings

doc_texts, doc_embeddings = load_or_create_embeddings()

try:
    price_data = pd.read_csv(csv_file)
except FileNotFoundError:
    price_data = pd.DataFrame(columns=["product", "price", "currency"])

def search_docs(user_question):
    try:
        q_emb_resp = client.models.embed_text(
            model="textembedding-gecko-001",
            input=[user_question]
        )
        q_emb = np.array(q_emb_resp.data[0].embedding).reshape(1, -1)
        sims = cosine_similarity(q_emb, doc_embeddings)
        best_idx = np.argmax(sims)
        if sims[0][best_idx] > 0.55:
            return doc_texts[best_idx]
        if "cybersecurity" in user_question.lower():
            for doc in doc_texts:
                if "cybersecurity" in doc.lower():
                    return doc
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def call_csv_tool(user_question):
    user_question = user_question.lower()
    for idx, row in price_data.iterrows():
        if row["product"].lower() in user_question:
            return f"{row['product'].title()}: {row['price']} {row['currency']}"
    return None

def ask_gemini(user_question, context=""):
    if context:
        prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer based on context:"
    else:
        prompt = user_question

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )
    return response.text

def run_query(user_question):
    doc_result = search_docs(user_question)
    if doc_result:
        answer = ask_gemini(user_question, context=doc_result[:3000])
        trace = "Retrieved from knowledge base"
        return answer, trace

    csv_result = call_csv_tool(user_question)
    if csv_result:
        answer = csv_result
        trace = "Retrieved from product database"
        return answer, trace

    answer = ask_gemini(user_question)
    trace = "Answered directly by Gemini"
    return answer, trace

def gradio_interface(user_input):
    answer, trace = run_query(user_input)
    return answer, trace

sample_questions = [
    "Tell me about cybersecurity",
    "Explain AI basics",
    "Price of pizza",
    "What is machine learning?",
    "How does cloud computing work?"
]

demo = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs=[gr.Textbox(lines=10, label="Answer"), "text"],
    title="Mini Agentic Pipeline",
    description="Ask questions about AI topics or product prices",
    examples=[[q] for q in sample_questions]
)

if __name__ == "__main__":
    demo.launch()
