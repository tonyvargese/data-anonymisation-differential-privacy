import streamlit as st
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from diffprivlib.models import GaussianNB
import base64

# Adjust the path to the correct location
sys.path.insert(0, 'C:/Users/tonyv/OneDrive/Desktop/Main Project')

# Ignore all warnings (use with caution)
warnings.filterwarnings("ignore")

# Disable the specific warning related to pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define a function for login
def login():
    st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "tony" and password == "tony":
            return True
        else:
            st.error("Invalid credentials. Please try again.")
            return False
    return False

# Check if the user is logged in
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

# If not logged in, show login page
if not st.session_state.is_logged_in:
    st.session_state.is_logged_in = login()

# If logged in, proceed with the app
if st.session_state.is_logged_in:
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (.csv file)", type="csv")

    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)

        # Perform an 80/20 train/test split
        target_variable_column_name = df.columns[-1]  # Get the name of the last column assuming it's the target variable
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(target_variable_column_name, axis=1),
            df[target_variable_column_name],
            test_size=0.2
        )

        # Sidebar with epsilon selection
        epsilon = st.sidebar.slider('Select epsilon', min_value=0.01, max_value=100.0, value=1.0, step=0.01)

        # Define a function to update the model and return necessary data
        def update_model_and_visualizations():
            # Train a differentially private naive Bayes classifier with specified bounds
            clf_dp = GaussianNB(bounds=None, epsilon=epsilon)
            clf_dp.fit(X_train, y_train)

            # Evaluate the accuracy of the model for the selected epsilon
            accuracy = clf_dp.score(X_test, y_test)

            return accuracy

        # Call the function to update the model and return accuracy
        accuracy = update_model_and_visualizations()

        # Display key statistics of the original dataset before differential privacy
        st.subheader("Key Statistics of Original Dataset Before Differential Privacy:")
        st.write(df.describe())

        # Train a differentially private naive Bayes classifier with specified bounds
        clf_dp = GaussianNB(bounds=None, epsilon=epsilon)
        clf_dp.fit(X_train, y_train)

        # Get private predictions
        private_predictions = clf_dp.predict(X_train)

        # Store the dataset after differential privacy
        dp_stats = pd.DataFrame(X_train, columns=df.columns[:-1])  # Initialize DataFrame with features
        dp_stats['Private_Predictions'] = private_predictions  # Add a new column for predictions

        # Display key statistics of the dataset after differential privacy
        st.subheader("Key Statistics of Dataset After Differential Privacy:")
        st.write(dp_stats.describe())

        # Save the feature distributions of the dataset after differential privacy to a CSV file
        dp_stats.to_csv("feature_distributions_after_dp.csv", index=False)

        # Visualize feature distributions of the dataset after differential privacy
        st.subheader("Distribution of Features After Differential Privacy:")
        for feature in df.columns[:-1]:
            plt.figure(figsize=(12, 8))
            sns.histplot(data=dp_stats, x=feature, hue='Private_Predictions', kde=True, multiple="stack")
            st.pyplot()

        # Display model accuracy
        st.subheader("Model Accuracy:")
        st.write(f"The accuracy of the model for epsilon {epsilon}: {accuracy:.2f}")

        # Provide a link to download the modified dataset
        st.subheader("Download Modified Dataset:")
        modified_csv = dp_stats.to_csv(index=False)
        b64 = base64.b64encode(modified_csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="modified_dataset.csv" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px; border: none; cursor: pointer;">Download Modified Dataset</a>'
        st.markdown(href, unsafe_allow_html=True)
