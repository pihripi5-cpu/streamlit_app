import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# Load environment variables (e.g., from a .env file)
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="PragyanAI Career Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State Management
# -----------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def clear_form():
    """Clears the session memory and reruns the app to reset state."""
    st.session_state.store = {}
    # Streamlit rerun to refresh the UI
    st.rerun()

# -----------------------------
# UI Header
# -----------------------------
# You can replace this URL with your local "PragyanAI_Transperent.png" if it's in the same folder
try:
    st.image("PragyanAI_Transperent.png", width=100)
except FileNotFoundError:
    pass # Skips image if not found locally

st.title("PragyanAI AI LinkedIn Career Coach & Profile Analyzer")
st.markdown("""
Analyze LinkedIn profiles, compare with Job Descriptions, estimate ATS score, 
identify skill gaps, generate interview questions, career roadmap, and LinkedIn posts.
""")

# -----------------------------
# Main Layout (Two Columns)
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Details")
    
    profile = st.text_area(
        "LinkedIn Profile / Resume / About Section", 
        height=250, 
        placeholder="Paste your LinkedIn profile, resume, or About section here..."
    )
    
    purposes = [
        "Professional Summary", "LinkedIn Analysis", "ATS Analysis", "Job Match",
        "Career Advice", "Resume Review", "Interview Preparation", "Career Roadmap",
        "Skill Gap Analysis", "Recruiter Review", "Headline Optimizer",
        "About Section Rewriter", "Experience Improvement", "LinkedIn Post Generator"
    ]
    purpose = st.selectbox("Purpose", options=purposes, index=1)
    
    company = st.text_input("Target Company", placeholder="Google, Microsoft, Amazon...")
    role = st.text_input("Target Job Role", placeholder="AI Engineer, Data Scientist...")
    
    experience = st.radio(
        "Experience Level", 
        options=["Student", "Fresher", "1-3 Years", "3-5 Years", "5+ Years"],
        index=0,
        horizontal=True
    )
    
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    jd = st.text_area(
        "Job Description (Optional)", 
        height=150, 
        placeholder="Paste Job Description here..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_btn = st.button("Analyze Profile", type="primary")
    with col_btn2:
        clear_btn = st.button("Clear History", on_click=clear_form)

with col2:
    st.subheader("PragyanAI Career Report")
    
    # Placeholder for output
    output_container = st.empty()
    
    if analyze_btn:
        if not profile.strip():
            output_container.error("Please provide a LinkedIn Profile, Resume, or About Section to analyze.")
        else:
            with st.spinner("Analyzing profile..."):
                # Retrieve API Key
                groq_api_key = os.environ.get("GROQ_API_KEY")
                
                if not groq_api_key:
                    output_container.error("GROQ_API_KEY not found. Please set it in your `.env` file or system environment variables.")
                else:
                    try:
                        # 1. Initialize LLM dynamically with the user's selected temperature
                        llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name="llama-3.3-70b-versatile",
                            temperature=temperature
                        )

                        # 2. Build the Prompt
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """
                            You are an experienced
                            • HR Recruiter
                            • LinkedIn Top Voice
                            • ATS Resume Expert
                            • Career Coach
                            • Hiring Manager

                            Analyze the LinkedIn Profile according to the selected purpose.
                            Always produce a professional report.

                            The report should contain wherever applicable:
                            1. Executive Summary
                            2. Overall Score (Out of 10)
                            3. Strengths
                            4. Weaknesses
                            5. Missing Skills
                            6. ATS Score
                            7. Recruiter Impression
                            8. Improvement Suggestions
                            9. Recommended Certifications
                            10. Recommended Projects
                            11. Interview Questions
                            12. Career Roadmap
                            13. LinkedIn Improvements
                            14. Resume Improvements
                            15. Final Recommendation

                            If the purpose is LinkedIn Post Generator, generate an engaging LinkedIn post.
                            """),
                            MessagesPlaceholder(variable_name="history"),
                            ("human", """
                            LinkedIn Profile:
                            {profile}

                            Purpose: {purpose}
                            Target Company: {company}
                            Target Job Role: {role}
                            Experience Level: {experience}
                            Job Description: {jd}
                            """)
                        ])

                        # 3. Create the Chain
                        parser = StrOutputParser()
                        chain = prompt | llm | parser

                        chain_with_history = RunnableWithMessageHistory(
                            chain,
                            get_session_history,
                            input_messages_key="profile",
                            history_messages_key="history"
                        )

                        # 4. Execute the Chain
                        response = chain_with_history.invoke(
                            {
                                "profile": profile,
                                "purpose": purpose,
                                "company": company,
                                "role": role,
                                "experience": experience,
                                "jd": jd
                            },
                            config={"configurable": {"session_id": "career"}}
                        )

                        # Display the result
                        output_container.markdown(response)
                        
                    except Exception as e:
                        output_container.error(f"An error occurred during analysis: {str(e)}")
    else:
        output_container.info("Enter your details on the left and click **Analyze Profile** to generate your report.")
