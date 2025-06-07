from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

import tempfile
import time

# --------------------------
# Gemini-based Conversational Chain
# --------------------------

def create_conversational_chain(vectorstore, gemini_api_key=None):
    if gemini_api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",  # or "gemini-pro"
                google_api_key=gemini_api_key,
                convert_system_message_to_human=True,
                temperature=0.7
            )
        except Exception as e:
            print(f"Gemini error: {e}")
            llm = get_dummy_llm()
    else:
        llm = get_dummy_llm()

    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

# Dummy fallback LLM
def get_dummy_llm():
    from langchain.schema import LLM
    class DummyLLM(LLM):
        def _call(self, prompt, stop=None):
            return "⚠️ No Gemini API key provided."
        @property
        def _identifying_params(self):
            return {}
        @property
        def _llm_type(self):
            return "dummy"
    return DummyLLM()

# --------------------------
# PDF Processing Functions
# --------------------------

def process_pdf(uploaded_file):
    """Reads and splits PDF into chunks for vector storage."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    return chunks

def get_vector_store(chunks, api_key=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)

# --------------------------
# Health Trackers
# --------------------------

def analyze_hydration(water_glasses):
    if water_glasses < 5:
        return "🚰 You may be dehydrated. Try to drink at least 8 glasses of water daily."
    elif 5 <= water_glasses <= 10:
        return "✅ You're doing well on hydration!"
    else:
        return "💧 Excellent! Keep staying hydrated."

def analyze_sleep(hours):
    if hours < 6:
        return "😴 Try to get at least 7–8 hours of sleep for better health."
    elif 6 <= hours <= 9:
        return "🌙 You're sleeping well. Great job!"
    else:
        return "⏰ Oversleeping sometimes causes fatigue. Maintain 7–9 hours."

def get_diet_suggestions(water_glasses, sleep_hours):
    tips = []
    if water_glasses < 5:
        tips.append("🔹 Increase your water intake. Dehydration can reduce metabolism.")
    elif water_glasses > 10:
        tips.append("🔹 Great hydration! Include hydrating foods like watermelon and cucumber.")

    if sleep_hours < 6:
        tips.append("🔹 Avoid caffeine and fried food. Focus on oats, bananas, and almonds.")
    elif sleep_hours >= 8:
        tips.append("🔹 You're well-rested! Include lean proteins, leafy greens, and healthy fats.")

    tips.append("🔹 Eat balanced meals (carbs + protein + veggies).")
    tips.append("🔹 Avoid late-night snacks.")

    return "\n".join(tips)
