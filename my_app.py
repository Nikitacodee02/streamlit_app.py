import streamlit as st
import pickle

# Save the model
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Streamlit app
st.title("Titanic Survival Prediction")
st.write("Enter the details of the passenger:")

Pclass = st.selectbox('Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 29)
SibSp = st.number_input('Number of Siblings/Spouses aboard', min_value=0, max_value=8, value=0)
Parch = st.number_input('Number of Parents/Children aboard', min_value=0, max_value=6, value=0)
Fare = st.number_input('Fare', min_value=0, max_value=500, value=32)
Embarked_S = st.selectbox('Embarked from Southampton?', ['No', 'Yes'])
Embarked_Q = st.selectbox('Embarked from Queenstown?', ['No', 'Yes'])

# Convert inputs to appropriate format
Sex = 1 if Sex == 'female' else 0
Embarked_S = 1 if Embarked_S == 'Yes' else 0
Embarked_Q = 1 if Embarked_Q == 'Yes' else 0

# Load the model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the feature vector
features = [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]]

# Predict
prediction = model.predict(features)
probability = model.predict_proba(features)[:, 1]

st.write(f"The predicted survival outcome is: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
st.write(f"Probability of survival: {probability[0]:.2f}")

streamlit run app.py
