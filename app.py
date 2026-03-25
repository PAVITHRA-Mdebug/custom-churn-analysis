import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

st.title("📊 Customer Churn Analysis App")

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
data = [
    ["7590-VHVEG","Female",0,"Yes","No",1,"No","No phone service","DSL","No","Yes","No","No","No","No","Month-to-month","Yes","Electronic check",29.85,29.85,"No"],
    ["5575-GNVDE","Male",0,"No","No",34,"Yes","No","DSL","Yes","No","Yes","No","No","No","One year","No","Mailed check",56.95,1889.5,"No"],
    ["3668-QPYBK","Male",0,"No","No",2,"Yes","No","DSL","Yes","Yes","No","Yes","No","No","Month-to-month","Yes","Mailed check",53.85,108.15,"Yes"],
    ["7795-CFOCW","Male",0,"No","No",45,"No","No phone service","DSL","Yes","Yes","No","No","Yes","No","One year","No","Bank transfer (automatic)",42.3,1840.75,"No"],
    ["9237-HQITU","Female",0,"No","No",2,"Yes","No","Fiber optic","No","No","No","No","No","No","Month-to-month","Yes","Electronic check",70.7,151.65,"Yes"]
]

columns = ["customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
           "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
           "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
           "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"]

df = pd.DataFrame(data, columns=columns)

st.subheader("Dataset Preview")
st.dataframe(df)

# ------------------------------
# Step 2: Visualization
# ------------------------------
st.subheader("Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# ------------------------------
# Step 3: Model Training
# ------------------------------
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,72], labels=['0-12','12-24','24-48','48-60','60-72'])

X = pd.get_dummies(df.drop(['customerID','Churn'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

model = RandomForestClassifier()
model.fit(X, y)

# ------------------------------
# Step 4: Prediction Section
# ------------------------------
st.subheader("Predict Churn")

gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

if st.button("Predict"):
    sample = df.iloc[[0]].copy()
    sample['gender'] = gender
    sample['tenure'] = tenure
    sample['MonthlyCharges'] = monthly

    sample = pd.get_dummies(sample.drop(['customerID','Churn'], axis=1))
    sample = sample.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")
