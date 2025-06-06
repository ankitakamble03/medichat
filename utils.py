from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
from langchain.chat_models import ChatOpenAI
import openai
import time
import openai
from openai.error import RateLimitError, OpenAIError

def openai_chat_completion_with_fallback(prompt, api_key, max_retries=3, wait_seconds=10):
    openai.api_key = api_key
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return response['choices'][0]['message']['content']
        except RateLimitError:
            print(f"Rate limit exceeded. Waiting {wait_seconds}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_seconds)
        except OpenAIError as e:
            # Other OpenAI errors
            print(f"OpenAI error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    # If all retries fail, return fallback message
    return "⚠️ Sorry, OpenAI quota exceeded or service unavailable. Please try again later."

# Example usage inside your chain or function:
def create_conversational_chain_with_retry(vectorstore, api_key):
    def llm_function(prompt):
        return openai_chat_completion_with_fallback(prompt, api_key)

    from langchain.chains import ConversationalRetrievalChain
    return ConversationalRetrievalChain.from_llm(llm=llm_function, retriever=vectorstore.as_retriever())


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
    # You can optionally use api_key inside here if needed.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)




def create_conversational_chain(vectorstore, api_key=None):
    if api_key:
        llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
    else:
        # fallback dummy LLM
        from langchain.schema import LLM
        class DummyLLM(LLM):
            def _call(self, prompt, stop=None):
                return "⚠️ No OpenAI key provided."
            @property
            def _identifying_params(self):
                return {}
            @property
            def _llm_type(self):
                return "dummy"
        llm = DummyLLM()

    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

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
