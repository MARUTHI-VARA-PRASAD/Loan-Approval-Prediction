# Step 1: Install required packages
!pip install pandas scikit-learn joblib ipywidgets --quiet

# Step 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import ipywidgets as widgets
from IPython.display import display

# Step 3: Load dataset
df = pd.read_csv('/content/Loan data set.csv')

# Step 4: Preprocess data
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(float)
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
df.fillna({'Gender':'Male','Married':'No','Self_Employed':'No'}, inplace=True)

# Features and target
X = df.drop(columns=['Loan_ID','Loan_Status'])
y = df['Loan_Status'].map({'Y':1,'N':0})

# Preprocessing
numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Dependents']
categorical_features = ['Gender','Married','Education','Self_Employed','Property_Area']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save model
joblib.dump(model, 'loan_model.pkl')
print("Model saved as loan_model.pkl")

# ------------------------------------------------
# Step 5: Interactive widgets for manual input
# ------------------------------------------------

# Create widgets
gender_widget = widgets.Dropdown(options=['Male','Female'], description='Gender:')
married_widget = widgets.Dropdown(options=['Yes','No'], description='Married:')
dependents_widget = widgets.IntSlider(min=0, max=5, description='Dependents:')
education_widget = widgets.Dropdown(options=['Graduate','Not Graduate'], description='Education:')
self_employed_widget = widgets.Dropdown(options=['Yes','No'], description='Self Employed:')
app_income_widget = widgets.IntText(value=0, description='Applicant Income:')
coapp_income_widget = widgets.IntText(value=0, description='Coapplicant Income:')
loan_amt_widget = widgets.IntText(value=0, description='Loan Amount:')
loan_term_widget = widgets.IntText(value=360, description='Loan Term:')
credit_hist_widget = widgets.IntSlider(min=0, max=1, description='Credit History:')
property_widget = widgets.Dropdown(options=['Urban','Semiurban','Rural'], description='Property Area:')

predict_button = widgets.Button(description="Predict Loan Status")
output = widgets.Output()

# Prediction function
def on_button_click(b):
    with output:
        output.clear_output()
        data = {
            'Gender': gender_widget.value,
            'Married': married_widget.value,
            'Dependents': dependents_widget.value,
            'Education': education_widget.value,
            'Self_Employed': self_employed_widget.value,
            'ApplicantIncome': app_income_widget.value,
            'CoapplicantIncome': coapp_income_widget.value,
            'LoanAmount': loan_amt_widget.value,
            'Loan_Amount_Term': loan_term_widget.value,
            'Credit_History': credit_hist_widget.value,
            'Property_Area': property_widget.value
        }
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        result = 'Approved' if prediction == 1 else 'Rejected'
        print(f"Loan Status Prediction: {result}")

predict_button.on_click(on_button_click)

# Display widgets
display(gender_widget, married_widget, dependents_widget, education_widget, self_employed_widget,
        app_income_widget, coapp_income_widget, loan_amt_widget, loan_term_widget,
        credit_hist_widget, property_widget, predict_button, output)
