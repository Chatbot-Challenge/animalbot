import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Sustainabot",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
    /* Hide Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Style the input box */
    .stTextInput>div>div>input {
        background-color: #f0f2f6 !important;
        border: 4px solid #4B5320 !important; /* dark military green */
        border-radius: 4px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        outline: none !important;
        box-shadow: 0 0 8px 2px #4B5320 !important;
        -webkit-appearance: none !important;
        appearance: none !important;
    }

    /* On focus: lighter military green with stronger glow */
    .stTextInput>div>div>input:focus,
    .stTextInput>div>div>input:focus-visible {
        border: 4px solid #6B8E23 !important; /* lighter military green */
        box-shadow: 0 0 12px 4px #6B8E23 !important;
        outline: none !important;
    }

    /* Autofill and invalid inputs override */
    input:-webkit-autofill,
    input:-webkit-autofill:focus,
    input:-webkit-autofill:active,
    input:invalid,
    input:invalid:focus {
        border: 4px solid #6B8E23 !important;
        box-shadow: 0 0 12px 4px #6B8E23 !important;
        -webkit-box-shadow: 0 0 12px 4px #6B8E23 inset !important;
        background-color: #f0f2f6 !important;
        color: black !important;
        outline: none !important;
    }

    /* Remove browser red validation icons/arrows */
    input::-webkit-validation-bubble-arrow,
    input::-webkit-validation-bubble-arrow-body {
        display: none !important;
    }

    /* Chat styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_input = ""
    st.session_state.input_key = 0
    st.session_state.is_loading = False
    st.session_state.is_creating_roadmap = False  # NEW: track roadmap generation state
    # Initial bot greeting
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": st.session_state.last_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages],
                "generate_roadmap": st.session_state.is_creating_roadmap  # ✅ add this
            }
        )
        response_data = response.json()
        st.session_state.messages.append({"role": "bot", "content": response_data["response"]})
    except Exception as e:
        st.error(f"Fehler bei der Initialkommunikation mit dem Server: {str(e)}")

# Titel und Beschreibung
st.title("♻️ Sustainabot")
st.markdown("""
🌱 Welcome to the Sustainable Packaging Consultant! 🌎\n
I'll help you find eco-friendly packaging solutions for your business.
""")

# Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div>👤 <b>You:</b></div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div>🤖 <b>Bot:</b></div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)


if st.session_state.is_loading:
    if st.session_state.is_creating_roadmap:
        st.markdown("🛠️ Roadmap is being created... This might take a moment ⏳")
    else:
        st.markdown("💬 Sustainabot is typing...")

user_input = st.text_input("Your message:", key=f"user_input_{st.session_state.input_key}")


if not st.session_state.is_loading:
    if user_input and user_input != st.session_state.last_input:
        st.session_state.is_loading = True
        st.session_state.is_creating_roadmap = False  
        st.session_state.last_input = user_input
        st.session_state.input_key += 1  
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.experimental_rerun()  


if st.session_state.is_loading:
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": st.session_state.last_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages],
                "generate_roadmap": st.session_state.is_creating_roadmap  
            }
        )

        response_data = response.json()

        # If backend signals roadmap creation, set the flag
        if response_data.get("is_loading", False):
            st.session_state.is_creating_roadmap = True
            st.experimental_rerun()
        else:
            st.session_state.is_creating_roadmap = False

            if "response" in response_data:
                st.session_state.messages.append({"role": "bot", "content": response_data["response"]})

            roadmap_items = response_data.get("roadmap")
            if not isinstance(roadmap_items, list):
                roadmap_items = []


            st.session_state.is_loading = False
            st.session_state.input_key += 1
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}")
        st.session_state.is_loading = False
        st.session_state.is_creating_roadmap = False