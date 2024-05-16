import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import time
# os.chdir('/Users/peter/Desktop/credit-score-prediction')

@st.cache_resource  
def load_model():
    
    with open('rf_classifier_random.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        return model

columns = ['Age',
       'Annual_Income', 'Num_Bank_Accounts',
       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
       'Credit_History_Age',
       'Payment_of_Min_Amount', 'Occupation',
        'Payment_Behaviour', 'Monthly_Balance']

occupations = [
        'Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer', 'Media Manager', 
        'Doctor', 'Journalist', 'Manager', 'Mechanic', 'Writer', 'Accountant', 
        'Architect', 'Musician', 'Developer'
    ]

payment_behaviors = ['High_spent_Large_value_payments',
       'High_spent_Medium_value_payments',
       'High_spent_Small_value_payments',
       'Low_spent_Large_value_payments',
       'Low_spent_Medium_value_payments',
       'Low_spent_Small_value_payments']

credit_mixs = ['Standard', 'Good', 'Bad']

Payment_of_Min_Amount = ['Yes', 'No']


# Load the model
model = load_model()

@st.cache_resource  
def predict_credit_score(data):
    occupations = data['Occupation'].unique()
    payment_behaviors = data['Payment_Behaviour'].unique()
    credit_mixs = data['Credit_Mix'].unique()

    for occupation in occupations:
        data[f'Occupation_{occupation}'] = (data['Occupation'] == occupation)

    data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].astype(str)
    payment_of_min_amount_categories = ['Yes', 'No'] 
    for category in payment_of_min_amount_categories:
        data[f'Payment_of_Min_Amount_{category}'] = (data['Payment_of_Min_Amount'] == category)

    # Create dummy variables for 'Payment_Behaviour'
    for payment_behavior in payment_behaviors:
        data[f'Payment_Behaviour_{payment_behavior}'] = (data['Payment_Behaviour'] == payment_behavior)

    # Create dummy variables for 'Credit_Mix'
    for credit_mix in credit_mixs:
        data[f'Credit_Mix_{credit_mix}'] = (data['Credit_Mix'] == credit_mix)

    # Ensure the data has all the columns expected by the model
    model_columns = model.feature_names_in_
    data = data.reindex(columns=model_columns, fill_value=0)

    # Make predictions
    prediction = model.predict(data)

    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

    return prediction, feature_importances

#### Cleaning column function ####
def clean_col(col):
    col = pd.to_numeric(col, errors='coerce')
    # Replace negative values with NaN
    col = col.apply(lambda x: x if x > 0 else np.nan)
    if col.name == 'Age':
        col = col.apply(lambda x: x if 1 <= x <= 100 else np.nan)
    return col

# Data cleaning (missing data + outliers)
def clean_data(df):
    columns = ['Age','Annual_Income', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
        'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
        'Credit_History_Age',
        'Payment_of_Min_Amount', 'Occupation',
        'Payment_Behaviour', 'Monthly_Balance','Credit_Score']

    df = df[[column for column in df.columns if column in columns]]
    
    # Age
    df['Age'] = clean_col(df['Age'])

    # Occupation
    df = df[df['Occupation'].str.contains('_______') == False]

    # Annual Income
    df['Annual_Income'] = clean_col(df['Annual_Income'])

    # Num Bank Account
    df['Num_Bank_Accounts'] = clean_col(df['Num_Bank_Accounts'])

    # Num Credit Card
    df['Num_Credit_Card'] = clean_col(df['Num_Credit_Card'])
    
    # Interest Rate
    df['Interest_Rate'] = clean_col(df['Interest_Rate'])

    # Num of Loan
    df['Num_of_Loan'] = clean_col(df['Num_of_Loan'])

    # Delay from due date
    df['Delay_from_due_date'] = clean_col(df['Delay_from_due_date'])

    # Num of Delated Payment
    df['Num_of_Delayed_Payment'] = clean_col(df['Num_of_Delayed_Payment'])

    # Change Credit Limit
    df['Changed_Credit_Limit'] = clean_col(df['Changed_Credit_Limit'])

    # Num Credit Inquiries
    df['Num_Credit_Inquiries'] = clean_col(df['Num_Credit_Inquiries'])
    
    # Credit Mix 
    df = df[df['Credit_Mix'].str.contains('_') == False]

    # Outstanding Debt
    df['Outstanding_Debt'] = clean_col(df['Outstanding_Debt'])

    
    # Credit History Age
    df['Credit_History_Age'] = df['Credit_History_Age'].astype(str).str.replace(' Years and ','.')
    df['Credit_History_Age'] = df['Credit_History_Age'].astype(str).str.replace('Months','')
    df['Credit_History_Age'] = df['Credit_History_Age'].astype(str).str.replace('nan','NaN')

    # Payment of Min Amount
    df = df[df['Payment_of_Min_Amount'] != 'NM']

    # Payment behavior
    df = df[df['Payment_Behaviour'] != '!@9#%8']

    # Monthly Balance 
    df['Monthly_Balance'] = clean_col(df['Monthly_Balance'])

    df['Age'] = df['Age'].astype('Int64')
    df['Interest_Rate'] = df['Interest_Rate'].astype('Int64')
    df['Delay_from_due_date'] = df['Delay_from_due_date'].astype('Int64')
    df['Credit_History_Age'] = df['Credit_History_Age'].astype('float')
    df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].astype('Int64')
    
    ## Convert to 2 decimal
    df['Monthly_Balance'] = df['Monthly_Balance'].round(2)
    
    ## Convert target variable into numerical
    df['Credit_Score'] = df['Credit_Score'].str.replace('Good', '2', n=-1)
    df['Credit_Score'] = df['Credit_Score'].str.replace('Standard', '1', n=-1)
    df['Credit_Score'] = df['Credit_Score'].str.replace('Poor', '0', n=-1)
    df['Credit_Score'] = df[['Credit_Score']].apply(pd.to_numeric)

    na_stats = df.isna().sum()
    st.write("Sum of the Nan values")
    st.write(na_stats)
    
    # Calculate the percentage of NaN values for each column
    nan_percentage = df.isna().mean() * 100
    
    # Sort the columns based on the percentage of NaN values in descending order
    nan_percentage_sorted = nan_percentage.sort_values(ascending=False)
    
    # Remove columns with 0 percent NaN values
    nan_percentage_sorted = nan_percentage_sorted[nan_percentage_sorted != 0]
    
    st.write("Percentage of the Nan values")
    st.write(nan_percentage_sorted)

    st.write("Boxplots for outliers")
    # Assuming df_cleaned is your DataFrame
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Set up a 3x3 subplot grid
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 12))
    # Flatten the 2D array of subplots into a 1D array
    axes = axes.flatten()
    
    # Loop through each numeric column and create a box plot
    for i, column in enumerate(numeric_columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Box Plot for {column}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    st.pyplot(fig)
    
    # Loop through each numeric column and drop outliers
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Drop outliers
        df = df.drop(df.loc[df[column] > (Q3 + 1.5 * IQR)].index)
        df = df.drop(df.loc[df[column] < (Q1 - 1.5 * IQR)].index)
    
    df.dropna(inplace=True)
    return df


# Start Streamlit
st.title('Credit Score Prediction App')
image_path = "https://i.imgur.com/iFV05Nd.jpeg"
st.image(image_path, use_column_width=True)

# Data file importer 
st.title("Import data file")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # To read a CSV file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    # To read an Excel file
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        
    df = clean_data(df)
    st.write("File successfully uploaded and cleaned!")
    st.write(df.head())



# User inputs
with st.sidebar:
    st.header('Customer Features')

    age = st.number_input('Age', min_value=18, max_value=100, value=21, step=1)
    occupation = st.selectbox("Occupation", [
        'Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer', 'Media Manager', 
        'Doctor', 'Journalist', 'Manager', 'Mechanic', 'Writer', 'Accountant', 
        'Architect', 'Musician', 'Developer'
    ])
    annual_income = st.number_input('Annual Income ($)', min_value=10000, max_value=1000000000, value=20000, step=5000)
    payment_of_min_amount = st.selectbox('Minimum Credit Card Payment Made?', ['Yes', 'No'])
    payment_behavior = st.selectbox("Payment Behavior", [
        'High_spent_Large_value_payments', 'High_spent_Medium_value_payments',
        'High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
        'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments'
    ])

# Main page layout for additional inputs
st.header('Additional Inputs')
col1, col2, col3 = st.columns(3)
with col1:
    num_of_bank_acc = st.number_input('Number of Bank Accounts Owned', min_value=1, max_value=12, value=1, step=1)
    num_credit_card = st.number_input('Number of Credit Cards Owned', min_value=1, max_value=12, value=1, step=1)
    outstanding_debt = st.number_input('Outstanding Debt ($)', min_value=0, max_value=100000, value=0, step=500)
    interest = st.number_input('Credit Cards Interest rate (%)', min_value=1.0, max_value=33.0, value=1.0, step=1.0)
with col2:
    num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=100, value=0, step=1)
    delay_from_due_date = st.number_input('Days Delayed Since Due Date for Payment', min_value=0, max_value=90, value=1, step=5)
    credit_history = st.number_input('Credit History Age (Years)', min_value=0, max_value=34, value=0, step=1)
    num_of_delayed = st.number_input("Number of Delayed Payments", min_value=0, max_value=4000, value=1, step=5)
with col3:
    changed_credit_limit = st.number_input("Percentage Change in Credit Card Limit", min_value=0, max_value=40, value=1, step=5)
    num_of_credit_card_inquiries = st.number_input("Number of Credit Card Inquiries", min_value=0, max_value=2600, value=1, step=5)
    credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])
    monthly_balance = st.number_input('Monthly Balance ($)', min_value=0, max_value=3000, value=0, step=100)

# Create a DataFrame from the user input
user_data = pd.DataFrame({
    'Age': [age],
    'Occupation': [occupation],
    'Annual_Income': [annual_income],
    'Num_Bank_Accounts': [num_of_bank_acc],
    'Num_Credit_Card': [num_credit_card],
    'Interest_Rate': [interest],
    'Num_of_Loan': [num_of_loan],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed],
    'Changed_Credit_Limit': [changed_credit_limit],
    'Num_Credit_Inquiries': [num_of_credit_card_inquiries],
    'Credit_Mix': [credit_mix],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_History_Age': [credit_history],
    'Monthly_Balance': [monthly_balance],
    'Payment_Behaviour': [payment_behavior],
    'Payment_of_Min_Amount': [payment_of_min_amount == 'True']
}, index=[0])

# Predict button and results
if st.button('Predict Credit Score'):
    # Display a spinner during the prediction process
    with st.spinner("Analyzing your data and predicting credit score..."):
        time.sleep(2)
        prediction, feature_importances = predict_credit_score(user_data)
        progress_bar = st.progress(0)
        
        # Check if a prediction was successfully made
        if prediction is not None:
            # Define labels for the predicted classes
            class_labels = {0: 'Poor Credit Score', 1: 'Average Credit Score', 2: 'Good Credit Score'}
            predicted_class = class_labels[prediction[0]]
            
            # Define styles for displaying the prediction result
            style = {
                0: "color: red; font-size: 24px;",
                1: "color: orange; font-size: 24px;",
                2: "color: green; font-size: 24px;"
            }[prediction[0]]
            
            # Update progress bar to 50% after prediction
            progress_bar.progress(50)
            time.sleep(2)
            
            # Display the prediction result with appropriate styling
            st.markdown(f"<h3 style='{style}'>Predicted Credit Score: {predicted_class}</h3>", unsafe_allow_html=True)
            
            # Provide a detailed explanation based on the prediction
            explanations = {
                0: "This indicates a high risk of defaulting on credit obligations.",
                1: "This indicates a moderate risk and moderate reliability in managing credit.",
                2: "This indicates a low risk and high reliability in managing credit."
            }
            st.write(f"### Explanation: {explanations[prediction[0]]}")
            
            # Display the user input features as a table for review
            st.subheader('Review Your Input Features')
            st.dataframe(user_data)
            
            # Complete the progress bar to 100%
            time.sleep(2)
            progress_bar.progress(100)
            st.balloons()
