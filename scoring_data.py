import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_model():
    with open('rf_classifier_random.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        return model


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

    model_columns = model.feature_names_in_
    data = data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(data)

    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

    return prediction, feature_importances

def main():
    model = load_model()
    
    data = pd.read_csv("cleaned-test.csv")
    
    predictions = predict_credit_score(data)
    
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()