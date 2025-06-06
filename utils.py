from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
import openai

# PDF Handling
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

def create_conversational_chain(vectorstore, api_key):
    return ConversationalRetrievalChain.from_llm(
        llm=lambda prompt: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )['choices'][0]['message']['content'],
        retriever=vectorstore.as_retriever()
    )

# Hydration Logic
def analyze_hydration(water_glasses):
    if water_glasses < 5:
        return "You may be dehydrated. Try to drink at least 8 glasses of water daily."
    elif 5 <= water_glasses <= 10:
        return "You're doing well on hydration!"
    else:
        return "Excellent! Keep staying hydrated."

# Sleep Logic
def analyze_sleep(hours):
    if hours < 6:
        return "Try to get at least 7–8 hours of sleep for better health."
    elif 6 <= hours <= 9:
        return "You're sleeping well. Great job!"
    else:
        return "Oversleeping sometimes causes fatigue. Maintain 7–9 hours."

# Diet Suggestion Logic
def get_diet_suggestions(water_glasses, sleep_hours):
    tips = []

    # Hydration effect on diet
    if water_glasses < 5:
        tips.append("🔹 Increase your water intake. Dehydration can reduce metabolism.")
    elif water_glasses > 10:
        tips.append("🔹 Great hydration! Maintain fruits like watermelon and cucumber.")

    # Sleep effect on diet
    if sleep_hours < 6:
        tips.append("🔹 Avoid caffeine and fried food. Focus on oats, bananas, and almonds.")
    elif sleep_hours >= 8:
        tips.append("🔹 You’re well-rested! Add lean proteins, leafy greens, and healthy fats.")

    # General
    tips.append("🔹 Eat balanced meals (carbs + protein + veggies).")
    tips.append("🔹 Avoid late-night snacks.")

    return "\n".join(tips)
