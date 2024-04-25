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

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (.csv file)", type="csv")

if uploaded_file is not None:
    # Load the uploaded dataset
    diabetes_df = pd.read_csv(uploaded_file)

    # Sidebar with epsilon selection
    epsilon = st.sidebar.slider('Select epsilon', min_value=0.01, max_value=100.0, value=1.0, step=0.01)

    # Define a function to update the model and return necessary data
    def update_model_and_visualizations():
        # Perform an 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes_df.drop('Outcome', axis=1),
            diabetes_df['Outcome'],
            test_size=0.2
        )

        # Train a differentially private naive Bayes classifier with specified bounds
        clf_dp_diabetes = GaussianNB(bounds=([0, 0, 0, 0, 0, 0, 0.078, 21], [17, 199, 122, 99, 846, 67.1, 2.42, 81]), epsilon=epsilon)
        clf_dp_diabetes.fit(X_train, y_train)

        # Evaluate the accuracy of the model for the selected epsilon
        accuracy_diabetes = clf_dp_diabetes.score(X_test, y_test)

        # Get private predictions
        private_predictions_diabetes = clf_dp_diabetes.predict(X_train)

        # Store the dataset after differential privacy
        dp_stats_diabetes = pd.DataFrame(X_train, columns=diabetes_df.columns[:-1])  # Initialize DataFrame with features
        dp_stats_diabetes['Private_Predictions'] = private_predictions_diabetes  # Add a new column for predictions

        return X_test, y_test, dp_stats_diabetes, accuracy_diabetes

    # Call the function to update the model and return necessary data
    X_test, y_test, dp_stats_diabetes, accuracy_diabetes = update_model_and_visualizations()

    # Display key statistics of the original dataset before differential privacy
    st.subheader("Key Statistics of Original Dataset Before Differential Privacy:")
    st.write(diabetes_df.describe())

    # Display key statistics of the dataset after differential privacy
    st.subheader("Key Statistics of Dataset After Differential Privacy:")
    st.write(dp_stats_diabetes.describe())

    # Save the feature distributions of the dataset after differential privacy to a CSV file
    dp_stats_diabetes.to_csv("feature_distributions_after_dp_diabetes.csv", index=False)

    # Visualize feature distributions of the dataset after differential privacy
    st.subheader("Distribution of Features After Differential Privacy:")
    for i, feature in enumerate(diabetes_df.columns[:-1]):
        plt.figure(figsize=(12, 8))
        sns.histplot(data=dp_stats_diabetes, x=feature, hue='Private_Predictions', kde=True, multiple="stack")
        st.pyplot()

    # Provide a link to download the modified dataset
    st.subheader("Download Modified Dataset:")
    csv_file_diabetes = dp_stats_diabetes.to_csv(index=False)
    b64_diabetes = base64.b64encode(csv_file_diabetes.encode()).decode()
    href_diabetes = f'<a href="data:file/csv;base64,{b64_diabetes}" download="modified_dataset_diabetes.csv">Download CSV</a>'
    st.markdown(href_diabetes, unsafe_allow_html=True)

    # Plotting accuracy
    st.subheader("Differentially Private Naive Bayes Accuracy on Diabetes Dataset:")
    st.write(f"Test accuracy for epsilon {epsilon} on diabetes dataset: {accuracy_diabetes}")
#