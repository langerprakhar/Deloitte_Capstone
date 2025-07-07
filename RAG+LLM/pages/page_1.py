import streamlit as st

st.set_page_config(page_title="Deloitte Landing", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:green;'>Deloitte Health Dashboard</h1>
    <p style='text-align:center;'>Welcome to the landing page! Click below to enter the dashboard.</p>
""", unsafe_allow_html=True)

# Button as HTML link to the dashboard page
dashboard_url = "http://localhost:8501/StreamlitWithRagTry"  # adjust to your local URL

st.markdown(f"""
    <div style='text-align:center; margin-top:50px;'>
        <a href="{dashboard_url}">
            <button style='padding: 20px 60px; font-size: 24px;'>🚀 Enter Dashboard</button>
        </a>
    </div>
""", unsafe_allow_html=True)
