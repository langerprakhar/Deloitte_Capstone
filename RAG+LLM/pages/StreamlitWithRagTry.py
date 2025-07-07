# ✅ Combined FHE-Compatible Dynamic Model + Streamlit + Gemini RAG Chatbot

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from concrete.ml.sklearn import XGBRegressor, XGBClassifier
from concrete.ml.deployment import FHEModelClient
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

# --- Environment setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        folder_path="/home/prakharlanger/Deloitte_Capstone_Project/RAG+LLM/faiss_store",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

def retrieve_docs(query: str, k: int = 4) -> list[str]:
    vs = load_vectorstore()
    results: list[Document] = vs.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def generate_answer(query: str) -> str:
    context = "\n\n".join(retrieve_docs(query))
    prompt = f"""
You are a helpful assistant with expertise in:
- Sleep, heart rate, fitness
- Privacy-preserving AI (FHE)

Use this context to answer:
{context}

User: {query}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# --- Streamlit Layout ---
st.set_page_config(page_title="Health & Fitness Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("💪 Personalized Health Metrics Dashboard + 🤖 Gemini")
# --- User Input: Dynamic Activity Goal ---
st.sidebar.header("⚙️ Personal Goal Settings")
activity_goal = st.sidebar.slider(
    "Daily Active Minutes Goal",
    min_value=10,
    max_value=120,
    value=30,
    step=5,
    help="Set your daily target for very active minutes."
)

# --- Page Title ---
st.subheader("💬 Ask the Gemini Health Assistant")

# --- Initialize Session State ---
if "chat" not in st.session_state:
    st.session_state.chat = []

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat = []

# --- Input Form ---
with st.form("chat_form", clear_on_submit=False):  # Input stays after submit
    user_input = st.text_input("Ask your health question...", key="chat_input")
    submitted = st.form_submit_button("Send")

# --- Process Input ---
if submitted and user_input.strip():
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = generate_answer(user_input)
    st.session_state.chat.append({"role": "bot", "content": response})

# --- Display Chat Only If Messages Exist ---
if st.session_state.chat:
    chat_html = """
    <div style='max-height: 300px; overflow-y: auto; padding: 1rem; border: 1px solid #444; border-radius: 10px; background-color: #111;'>
    """

    for msg in st.session_state.chat:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Gemini"
        chat_html += f"<p><strong>{role}:</strong> {msg['content']}</p>"

    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

# --- Load Data ---

# --- Load Data ---
@st.cache_data
def load_data():
    st.sidebar.header("📂 Upload Your Fitbit Data")

uploaded_daily = st.sidebar.file_uploader("Upload Daily Activity CSV", type="csv")
uploaded_hr = st.sidebar.file_uploader("Upload Heart Rate CSV", type="csv")
uploaded_hourly = st.sidebar.file_uploader("Upload Hourly Activity CSV", type="csv")

@st.cache_data
def load_data(uploaded_daily, uploaded_hr, uploaded_hourly):
    if uploaded_daily is not None:
        daily = pd.read_csv(uploaded_daily)
    else:
        daily = pd.read_csv("/home/prakharlanger/Deloitte_Capstone_Project/Dataset/dailyActivity_merged.csv")
    
    if uploaded_hr is not None:
        hr = pd.read_csv(uploaded_hr)
    else:
        hr = pd.read_csv("/home/prakharlanger/Deloitte_Capstone_Project/Dataset/heartrate_minutes_avg.csv")

    if uploaded_hourly is not None:
        hourly = pd.read_csv(uploaded_hourly)
    else:
        hourly = pd.read_csv("/home/prakharlanger/Deloitte_Capstone_Project/Dataset/hourlyActivity_merged.csv")

    return daily, hr, hourly

daily, hr, hourly = load_data(uploaded_daily, uploaded_hr, uploaded_hourly)

# --- Preprocessing ---
hr['ActivityMinute'] = pd.to_datetime(hr['ActivityMinute'])
hr['Date'] = hr['ActivityMinute'].dt.date
hr = hr.dropna(subset=['Avg_HeartRate'])

scaler = StandardScaler()
hr['HR_Scaled'] = scaler.fit_transform(hr[['Avg_HeartRate']])
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
hr['IsAnomaly'] = (iso.fit_predict(hr[['HR_Scaled']]) == -1).astype(int)

hr_daily = hr.groupby(['Id', 'Date']).agg({
    'Avg_HeartRate': ['mean', 'max', 'min', 'std'],
    'IsAnomaly': 'mean'
}).reset_index()
hr_daily.columns = ['Id', 'ActivityDate', 'HR_Mean', 'HR_Max', 'HR_Min', 'HR_Std', 'HR_RiskScore']
hr_daily['ActivityDate'] = pd.to_datetime(hr_daily['ActivityDate'])

daily['ActivityDate'] = pd.to_datetime(daily['ActivityDate'])
df = daily.merge(hr_daily, on=['Id', 'ActivityDate'], how='left')

hourly['ActivityHour'] = pd.to_datetime(hourly['ActivityHour'])
hourly['Hour'] = hourly['ActivityHour'].dt.hour
hourly['Date'] = hourly['ActivityHour'].dt.date

# Evening Activity: 8 PM to 11 PM
evening_df = hourly[(hourly['Hour'] >= 20) & (hourly['Hour'] <= 23)]
evening_agg = evening_df.groupby(['Id', 'Date']).agg({
    'TotalIntensity': 'sum',
    'StepTotal': 'sum'
}).reset_index()
evening_agg.columns = ['Id', 'ActivityDate', 'EveningIntensity', 'EveningSteps']
evening_agg['ActivityDate'] = pd.to_datetime(evening_agg['ActivityDate'])

# Night Heart Rate: 11 PM to 6 AM
hr['Hour'] = hr['ActivityMinute'].dt.hour
night_hr = hr[(hr['Hour'] >= 23) | (hr['Hour'] <= 6)]
night_agg = night_hr.groupby(['Id', 'Date']).agg({
    'Avg_HeartRate': ['mean', 'std']
}).reset_index()
night_agg.columns = ['Id', 'ActivityDate', 'NightHR_Mean', 'NightHR_Std']
night_agg['ActivityDate'] = pd.to_datetime(night_agg['ActivityDate'])

df = df.merge(evening_agg, on=['Id', 'ActivityDate'], how='left')
df = df.merge(night_agg, on=['Id', 'ActivityDate'], how='left')
df[['EveningIntensity', 'EveningSteps', 'NightHR_Mean', 'NightHR_Std']] = df[
    ['EveningIntensity', 'EveningSteps', 'NightHR_Mean', 'NightHR_Std']
].fillna(0)

df = df.dropna(subset=['Calories', 'TotalMinutesAsleep', 'TotalTimeInBed', 'VeryActiveMinutes'])
df['SleepQuality'] = df['TotalMinutesAsleep'] / df['TotalTimeInBed']
df['MetActiveGoal'] = (df['VeryActiveMinutes'] >= activity_goal).astype(int)
df['ActiveRatio'] = df['VeryActiveMinutes'] / (df['VeryActiveMinutes'] + df['SedentaryMinutes'] + 1)
df['DistancePerStep'] = df['TotalDistance'] / (df['TotalSteps'] + 1)
df['ActiveIntensity'] = df['TotalSteps'] / (df['FairlyActiveMinutes'] + df['VeryActiveMinutes'] + 1)
df['UserID'] = df['Id'].astype("category").cat.codes
df = df[df['Calories'] < 5000]
df = df[df['SleepQuality'] <= 1.0]

features = [
    'TotalSteps', 'TotalDistance', 'TrackerDistance', 'LoggedActivitiesDistance',
    'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance',
    'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes',
    'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalSleepRecords',
    'TotalMinutesAsleep', 'TotalTimeInBed', 'ActiveRatio', 'DistancePerStep',
    'ActiveIntensity', 'HR_Mean', 'HR_Max', 'HR_Min', 'HR_Std', 'HR_RiskScore',
    'EveningIntensity', 'EveningSteps', 'NightHR_Mean', 'NightHR_Std', 'UserID'
]

X = df[features].fillna(0).round(2)
y_calories = df['Calories']
y_sleep_quality = df['SleepQuality']
y_met_goal = df['MetActiveGoal']

X_train, X_test, y_cal_train, y_cal_test = train_test_split(X, y_calories, test_size=0.2, random_state=42)
_, _, y_sleep_train, y_sleep_test = train_test_split(X, y_sleep_quality, test_size=0.2, random_state=42)
_, _, y_goal_train, y_goal_test = train_test_split(X, y_met_goal, test_size=0.2, random_state=42)

model_cal = XGBRegressor(n_estimators=100, max_depth=3)
model_sleep = XGBRegressor(n_estimators=100, max_depth=3)
model_goal = XGBClassifier(n_estimators=100, max_depth=3)

model_cal.fit(X_train, y_cal_train)
model_sleep.fit(X_train, y_sleep_train)
model_goal.fit(X_train, y_goal_train)

df['PredictedCalories'] = model_cal.predict(X)
df['PredictedSleepQuality'] = model_sleep.predict(X)
df['PredictedMetActiveGoal'] = model_goal.predict(X)


# 🔮 Daily Predictions
df['PredictedCalories'] = model_cal.predict(X)
df['PredictedSleepQuality'] = model_sleep.predict(X)
df['PredictedMetActiveGoal'] = model_goal.predict(X)

# 📌 Summary Metrics
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Predicted Calories", f"{df['PredictedCalories'].mean():.0f} kcal")
col2.metric("Avg Sleep Quality", f"{df['PredictedSleepQuality'].mean():.2f}")
col3.metric(f"% Days Met Goal (≥{activity_goal} mins)", f"{df['MetActiveGoal'].mean() * 100:.1f}%")
# --- Visualizations ---

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("🎯 Activity Goal Distribution")
    fig1, ax1 = plt.subplots()
    labels = ['Not Met', 'Met']
    values = df['PredictedMetActiveGoal'].value_counts().sort_index()
    colors = ["black", "green"]
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

with row1_col2:
    st.subheader("🧠 Heart Rate Anomaly Timeline")
    hr['Timestamp'] = pd.to_datetime(hr['ActivityMinute'])
    fig2, ax2 = plt.subplots(figsize=(8, 4))  # Wider helps too!
    hr_sample = hr.sample(n=1000, random_state=42).sort_values("Timestamp")

    colors = np.where(hr_sample['IsAnomaly'] == 1, 'red', 'lime')
    ax2.scatter(hr_sample['Timestamp'], hr_sample['Avg_HeartRate'], c=colors, alpha=0.6)

    ax2.set_title("Heart Rate Anomalies (Sample)", color='white')
    ax2.set_xlabel("Time", color='white')
    ax2.set_ylabel("Avg Heart Rate", color='white')
    ax2.set_facecolor('black')
    fig2.patch.set_facecolor('black')

    # Rotate date labels
    ax2.tick_params(axis='x', rotation=45)

    # Optional: Set date format for better spacing
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_color('lime')

    fig2.tight_layout()  # Avoid clipping rotated labels
    st.pyplot(fig2)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    corr = df[features + ['PredictedCalories']].corr()
    sns.heatmap(corr, cmap="Greens", linewidths=0.5, ax=ax3)
    ax3.set_title("Correlation Heatmap", color='white')
    st.pyplot(fig3)

with row2_col2:
    st.subheader("😴 Calories vs Sleep Quality")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.scatter(df['PredictedCalories'], df['PredictedSleepQuality'], c='lime', alpha=0.6)
    ax4.set_xlabel("Calories", color='white')
    ax4.set_ylabel("Sleep Quality", color='white')
    ax4.set_facecolor('black')
    fig4.patch.set_facecolor('black')
    for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
        label.set_color('lime')
    ax4.set_title("Calories vs Sleep Quality", color='white')
    st.pyplot(fig4)

row3_col1, _ = st.columns(2)
with row3_col1:
    st.subheader("🏃‍♂️ Active Intensity Distribution")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['ActiveIntensity'], kde=True, bins=30, color='lime', ax=ax5)
    ax5.set_facecolor('black')
    fig5.patch.set_facecolor('black')
    ax5.set_title("Active Intensity Histogram", color='white')
    ax5.set_xlabel("Active Intensity", color='white')
    ax5.set_ylabel("Frequency", color='white')
    for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
        label.set_color('lime')
    st.pyplot(fig5)

# --- Suggestions ---
st.subheader("💡 Possible Anomaly Causes & Suggestions")
st.markdown("""
- 🔴 **Detected anomalies** in heart rate may relate to:
    - Sleep deprivation / caffeine / dehydration
    - Stress / illness / overexertion
- ✅ Suggested Actions:
    - Sleep 7–9 hrs daily
    - Hydrate well
    - Limit stimulants
    - Rest after workouts
""")

# --- Data Table ---
st.subheader("📋 Explore FHE-Protected Data")
st.dataframe(df[['ActivityDate', 'Id', 'PredictedCalories', 'PredictedSleepQuality', 'PredictedMetActiveGoal']].sort_values('ActivityDate'), use_container_width=True)
