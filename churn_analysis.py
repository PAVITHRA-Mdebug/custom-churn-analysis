# churn_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Step 1: Define dataset directly
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

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# ------------------------------
# Step 2: Basic EDA
# ------------------------------
print("First 5 rows of dataset:\n", df.head())
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# ------------------------------
# Step 3: Feature Engineering
# ------------------------------
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,72], labels=['0-12','12-24','24-48','48-60','60-72'])
X = pd.get_dummies(df.drop(['customerID','Churn'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# ------------------------------
# Step 4: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 5: Train Model
# ------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------
# Step 6: Results
# ------------------------------
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.head(10).plot(kind='bar')
plt.title("Top Features Influencing Churn")
plt.show()

# ------------------------------
# Step 7: Optional Risk Score
# ------------------------------
y_prob = model.predict_proba(X_test)[:,1]
risk_score = (y_prob*100).astype(int)
print("\nCustomer Risk Scores:\n", pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Risk Score": risk_score}))