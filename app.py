import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

st.title("📊 Customer Churn Prediction App")

# ------------------------------
# Instructions
# ------------------------------
st.sidebar.header("📌 Upload Your Dataset")
st.sidebar.markdown("""
Upload a CSV file with the following columns:

- gender  
- SeniorCitizen  
- tenure  
- MonthlyCharges  
- Churn  

Example:
Male,0,12,50.5,Yes
""")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------
# Validate Columns
# ------------------------------
required_cols = ["gender", "SeniorCitizen", "tenure", "MonthlyCharges", "Churn"]

if not all(col in df.columns for col in required_cols):
    st.error("❌ Uploaded file does not have required columns")
    st.stop()

# ------------------------------
# Show Data
# ------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# Visualization
# ------------------------------
st.subheader("📊 Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# ------------------------------
# Model Training
# ------------------------------
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)

X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

# ------------------------------
# Prediction Section
# ------------------------------
st.subheader("🔮 Predict Customer Churn")

gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

if st.button("Predict"):
    sample = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [0],
        "tenure": [tenure],
        "MonthlyCharges": [monthly]
    })

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")
