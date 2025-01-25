import streamlit as st
import joblib
import numpy as np

# Path to the pre-trained model
MODEL_PATH = "svm_model.pkl"

# Load the pre-trained SVM model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function for making predictions
def make_prediction(model, input_data):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit Application UI
st.set_page_config(page_title="SVM Prediction App", layout="centered", page_icon="🔍")

# App title and description
st.title("SVM Prediction App")
st.markdown(
    """<style>h1 {text-align: center;}</style>",
    unsafe_allow_html=True
)
st.markdown(
    """Welcome to the **SVM Prediction App**! Enter your details below to find out if a purchase is likely.""",
    unsafe_allow_html=True
)

# Load the model
st.divider()
svm_model = load_model()
st.divider()

if svm_model:
    with st.form("prediction_form"):
        st.header("Input Features")

        # Input fields
        gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)
        age = st.slider("Age", min_value=0, max_value=100, value=25, step=1)
        salary = st.number_input("Estimated Salary", min_value=0, max_value=500000, value=50000, step=1000)

        # Submit button
        submitted = st.form_submit_button("Predict")

        # If form is submitted
        if submitted:
            # Convert Gender to numerical values
            gender_encoded = 1 if gender == "Male" else 0
            user_input = [gender_encoded, age, salary]

            # Make prediction
            prediction = make_prediction(svm_model, user_input)

            if prediction is not None:
                result = "**Purchased**" if prediction == 1 else "**Not Purchased**"
                st.success(f"Prediction: {result}")
                st.balloons()
else:
    st.error("Model could not be loaded. Please check the logs.")
