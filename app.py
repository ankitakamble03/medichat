import openai
import streamlit as st
from utils.utils import (
    analyze_hydration,
    analyze_sleep,
    get_diet_suggestions,
    process_pdf,
    get_vector_store,
    create_conversational_chain
)


# Page config
st.set_page_config(page_title="MediChat AI", layout="wide")

# Sidebar: API key input
st.sidebar.title("🔐 API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Sidebar: Navigation
st.sidebar.title("🩺 MediChat Navigation")
page = st.sidebar.radio("Go to", ["💧 Hydration", "🌙 Sleep", "🥗 Diet", "📄 PDF Chat", "📊 BMI Calculator"])

# --- Hydration Tab ---
if page == "💧 Hydration":
    st.title("💧 Hydration Tracker")
    glasses = st.slider("How many glasses of water did you drink today?", 0, 20, 8)
    if st.button("Analyze Hydration"):
        result = analyze_hydration(glasses)
        st.success(result)

# --- Sleep Tab ---
elif page == "🌙 Sleep":
    st.title("🌙 Sleep Tracker")
    hours = st.slider("How many hours did you sleep last night?", 0, 12, 7)
    if st.button("Analyze Sleep"):
        result = analyze_sleep(hours)
        st.success(result)

# --- Diet Tab ---
elif page == "🥗 Diet":
    st.title("🥗 Diet Suggestions")
    st.write("Get personalized diet tips based on hydration and sleep.")
    water = st.slider("Water Intake (glasses)", 0, 20, 8, key="diet_water")
    sleep = st.slider("Sleep Duration (hours)", 0, 12, 7, key="diet_sleep")
    if st.button("Get Diet Tips"):
        tips = get_diet_suggestions(water, sleep)
        st.info(tips)

# --- PDF Chat Tab ---
elif page == "📄 PDF Chat":
    st.title("📄 Medical Report Assistant")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    else:
        uploaded_file = st.file_uploader("Upload your medical PDF", type="pdf")
        if uploaded_file:
            st.success("✅ PDF uploaded and processed.")
            chunks = process_pdf(uploaded_file)
            vectorstore = get_vector_store(chunks, api_key)
            qa_chain = create_conversational_chain(vectorstore, api_key)

            st.write("💬 Ask questions about your medical report:")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Helper to remove emojis and non-ASCII
def clean_text(text):
    return ''.join(c for c in text if ord(c) < 128)

query = st.text_input("Ask a question")
if query:
    with st.spinner("Thinking..."):
        # Clean query and chat history
        clean_query = clean_text(query)
        clean_history = [(clean_text(q), clean_text(a)) for q, a in st.session_state.chat_history]

        # Run clean input
        answer = qa_chain.run({"question": clean_query, "chat_history": clean_history})

        # Append original (not cleaned) query and answer to display
        st.session_state.chat_history.append((query, answer))
        st.success(answer)

# --- BMI Tab ---
elif page == "📊 BMI Calculator":
    st.title("📊 Body Mass Index (BMI) Calculator")

    height_cm = st.number_input("Enter your height (in cm)", min_value=50, max_value=250, value=165)
    weight_kg = st.number_input("Enter your weight (in kg)", min_value=20, max_value=300, value=60)

    if st.button("Calculate BMI"):
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        st.info(f"Your BMI is **{bmi:.2f}**")

        if bmi < 18.5:
            st.warning("You are underweight.")
        elif 18.5 <= bmi < 24.9:
            st.success("You have a healthy weight.")
        elif 25 <= bmi < 29.9:
            st.warning("You are overweight.")
        else:
            st.error("You are in the obese range.")
