import streamlit as st
from front import show_front_page
from PIL import Image
import base64
from pathlib import Path

st.set_page_config(
    page_title="ARGO AI Ocean Explorer",
    layout="wide",
page_icon="FloatChat.png")


logo_path = Path(__file__).parent / "FloatChat.png"
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = "landing"

# Styling
st.markdown("""
<style>
.stApp {
    background-image: url('https://i.pinimg.com/736x/86/f8/4b/86f84bf00e07cc71f76151118f764234.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed; 
}

.hero-title { 
    text-align:left; 
    font-size:70px; 
    margin-top: 10px;
    margin-left:170px;
    color: white;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
}

.title { 
    text-align:left; 
    font-size:100px; 
    margin-top:0px;
    color: white;
    margin-left:150px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
}

.hero-sub { 
    text-align:left;
    font-size:30px; 
    margin-top:20px;
    color: white;
    margin-left:150px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
}

/* Style for the button container */
.button-container {
    display: flex;
    justify-content: center;
    margin-top: 60px;
    
}

.stButton > button {
    padding: 20px 60px;
    font-size: 80px;
    border-radius: 50px;
    background-color: #001528;
    color: white;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-left:150px;
}

.stButton > button:hover {
  
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <style>
    .custom-logo {{
        position: absolute;   
        top: 20px;
        left: 5px;
        width: 120px;        /* control size */
        z-index: 9999;
        margin-top:0px; 
        margin-left:0px;
        margin-right:900px;
    }}
    </style>

    <img src="data:image/png;base64,{logo_b64}" class="custom-logo" />
""", unsafe_allow_html=True)

if st.session_state.page == "landing":
    
    st.markdown('<div class="hero-title">ARGO</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">FloatChat</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explore the Oceans with ARGO AI</div>', unsafe_allow_html=True)
  
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Let's Dive!", key="dive_button"):
        st.session_state.page = "front"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "front":
    show_front_page()
