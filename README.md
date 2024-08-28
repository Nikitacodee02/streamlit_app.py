# streamlit_app.py
import pickle
import pandas as pd

# Load the trained model (assuming the model is saved as 'logistic_model.pkl')
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input data
example_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # Female
    'Age': [29],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [32],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

# Make a prediction
prediction = model.predict(example_data)
probability = model.predict_proba(example_data)[:, 1]

print(f'Predicted Survival: {"Survived" if prediction[0] == 1 else "Did Not Survive"}')
print(f'Probability of Survival: {probability[0]:.2f}')
