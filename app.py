import streamlit as st
import openai
from utils import (
    process_pdf, get_vector_store, create_conversational_chain,
    analyze_hydration, analyze_sleep, get_diet_suggestions
)

st.set_page_config(page_title="MediChat AI", layout="wide")

# Sidebar: OpenAI Key
api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# Sidebar Navigation with icons (using emojis)
st.sidebar.title("📋 Navigation")
menu = {
    "🏥 General Health Chat": "general",
    "💧 Hydration Tracker": "hydration",
    "😴 Sleep Tracker": "sleep",
    "🥗 Diet Recommendation": "diet",
    "📄 Medical Report (PDF) Chat": "pdf"
}
choice = st.sidebar.radio("Go to:", list(menu.keys()))

# Map choice to function
page = menu[choice]

st.title("🤖 MediChat AI - Your Health Companion")

if page == "general":
    st.header("🏥 General Health Chat")
    query = st.text_input("Ask your health question:")
    if query and api_key:
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}]
            )
            st.success(response.choices[0].message.content)

elif page == "hydration":
    st.header("💧 Hydration Tracker")
    glasses = st.number_input("How many glasses of water did you drink today?", 0, 20, 8)
    if st.button("Analyze Hydration"):
        feedback = analyze_hydration(glasses)
        st.success(feedback)

elif page == "sleep":
    st.header("😴 Sleep Tracker")
    hours = st.slider("How many hours did you sleep last night?", 0, 24, 7)
    if st.button("Analyze Sleep"):
        feedback = analyze_sleep(hours)
        st.success(feedback)

elif page == "diet":
    st.header("🥗 Diet Recommendation")
    glasses = st.number_input("Water intake (glasses)", 0, 20, 8, key="diet_water")
    sleep = st.slider("Sleep duration (hours)", 0, 24, 7, key="diet_sleep")
    if st.button("Get Diet Plan"):
        diet_advice = get_diet_suggestions(glasses, sleep)
        st.success(diet_advice)

elif page == "pdf":
    st.header("📄 Medical Report (PDF) Chat")
    pdf_file = st.file_uploader("Upload your medical report (PDF)", type="pdf")
    if pdf_file and api_key:
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(pdf_file)
            vector_store = get_vector_store(chunks)
            chain = create_conversational_chain(vector_store, api_key)
        question = st.text_input("Ask something about your report:")
        if question:
            with st.spinner("Analyzing..."):
                response = chain.invoke({"question": question, "chat_history": []})
                st.success(response["answer"])
