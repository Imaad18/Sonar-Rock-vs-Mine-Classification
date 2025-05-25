import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS styling
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .title {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    
    .sidebar .sidebar-content .sidebar-title {
        color: white;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Slider styling */
    .stSlider .thumb {
        background-color: #3498db !important;
    }
    
    .stSlider .track {
        background-color: #bdc3c7 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Card styling for prediction results */
    .prediction-card {
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .mine-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    
    .rock-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    
    /* Progress bar styling */
    .stProgress>div>div>div {
        background-color: #3498db;
    }
    
    /* Radio button styling */
    .stRadio>div>label {
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Sonar Rock vs Mine Classifier",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.markdown('<h1 class="title">Sonar Rock vs Mine Classification</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #e9f7fe; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    This app uses machine learning to classify sonar signals as either a rock (R) or a mine (M).
    The model is trained on the sonar dataset which contains patterns obtained by bouncing sonar signals
    off either rocks or metal cylinders (mines) at various angles and conditions.
</div>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("sonar data.csv", header=None)
        return data
    except FileNotFoundError:
        st.error("Sonar data file not found. Please upload the data file.")
        return None

# Function to preprocess data
def preprocess_data(data):
    X = data.drop(60, axis=1)  # Features (first 60 columns)
    y = data[60]  # Labels ("M" or "R")
    return X, y

# Function to split and scale data
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train model
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate model
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)
    class_report = classification_report(y, predictions, output_dict=True)
    return accuracy, conf_matrix, class_report

# Function to predict
def predict_input(model, scaler, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    label = "Mine" if prediction[0] == "M" else "Rock"
    return label, probability

# Sidebar for navigation
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "Make Prediction"])

# File uploader
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-title">Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload sonar data CSV file", type=["csv"], label_visibility="collapsed")

# Initialize session state to store model and data
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Load and train the model
if uploaded_file is not None:
    # Load data from uploaded file
    data = pd.read_csv(uploaded_file, header=None)
    st.session_state.data = data
    
    # Preprocess data
    X, y = preprocess_data(data)
    st.session_state.X = X
    st.session_state.y = y
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)
    st.session_state.scaler = scaler
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    st.session_state.model = model
    
    st.sidebar.success("Model trained successfully!")
else:
    # Try to load the default data
    data = load_data()
    if data is not None:
        st.session_state.data = data
        
        # Preprocess data
        X, y = preprocess_data(data)
        st.session_state.X = X
        st.session_state.y = y
        
        # Split and scale data
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)
        st.session_state.scaler = scaler
        
        # Train model
        model = train_model(X_train_scaled, y_train)
        st.session_state.model = model

# Home page
if page == "Home":
    st.header("Rock vs Mine Classification using Sonar Data")
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3>How it works</h3>
        <p>This application uses a Logistic Regression model to classify sonar signals as either rocks or mines.</p>
        
        <h3>The Data</h3>
        <p>The dataset consists of 208 patterns obtained by bouncing sonar signals off a metal cylinder (mine) 
        and rocks under various conditions. Each pattern is a set of 60 numbers in the range 0.0 to 1.0.
        Each number represents the energy within a particular frequency band, integrated over a certain period of time.</p>
        
        <h3>Instructions</h3>
        <ol>
            <li>Upload the sonar data CSV file using the sidebar</li>
            <li>Explore the data in the "Data Exploration" section</li>
            <li>Check model performance in the "Model Performance" section</li>
            <li>Make predictions with your own inputs in the "Make Prediction" section</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        st.write("### Sample Data")
        st.dataframe(st.session_state.data.head().style.set_properties(**{
            'background-color': '#f8f9fa',
            'border': '1px solid #dee2e6'
        }))
    
# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if st.session_state.data is not None:
        # Display data info
        st.subheader("Dataset Information")
        
        # Get basic statistics
        buffer = []
        buffer.append(f"Number of samples: {len(st.session_state.data)}")
        
        # Count label distribution
        label_counts = st.session_state.data[60].value_counts()
        buffer.append(f"Number of Mines (M): {label_counts.get('M', 0)}")
        buffer.append(f"Number of Rocks (R): {label_counts.get('R', 0)}")
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            {"<br>".join(buffer)}
        </div>
        """, unsafe_allow_html=True)
        
        # Display class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=st.session_state.data[60], ax=ax, palette=["#3498db", "#2ecc71"])
        plt.title("Distribution of Rocks and Mines")
        plt.xlabel("Class")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        # Display feature visualization
        st.subheader("Feature Visualization")
        
        # Average feature values by class
        mines = st.session_state.data[st.session_state.data[60] == "M"].drop(60, axis=1).mean()
        rocks = st.session_state.data[st.session_state.data[60] == "R"].drop(60, axis=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(mines.index, mines.values, 'r-', label='Mine')
        plt.plot(rocks.index, rocks.values, 'b-', label='Rock')
        plt.title("Average Feature Values by Class")
        plt.xlabel("Feature Index")
        plt.ylabel("Average Value")
        plt.legend()
        st.pyplot(fig)
        
        # Display correlation heatmap
        st.subheader("Feature Correlation")
        
        # Select a subset of features to display in the heatmap (to avoid overcrowding)
        corr_features = st.session_state.X.iloc[:, :10]  # First 10 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_features.corr(), cmap="viridis", annot=False, ax=ax)
        plt.title("Correlation Heatmap of First 10 Features")
        st.pyplot(fig)
        
    else:
        st.warning("Please upload the sonar dataset to explore data.")

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance")
    
    if st.session_state.model is not None:
        # Split data for evaluation
        X, y = st.session_state.X, st.session_state.y
        X_train_scaled, X_test_scaled, y_train, y_test, _ = split_and_scale(X, y)
        
        # Evaluate on training data
        train_accuracy, train_conf_matrix, train_report = evaluate_model(st.session_state.model, X_train_scaled, y_train)
        
        # Evaluate on test data
        test_accuracy, test_conf_matrix, test_report = evaluate_model(st.session_state.model, X_test_scaled, y_test)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Results")
            st.markdown(f"""
            <div style="background-color: #e9f7fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4>Accuracy: {train_accuracy:.4f}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Rock', 'Mine'], yticklabels=['Rock', 'Mine'], ax=ax)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Training Confusion Matrix')
            st.pyplot(fig)
            
            # Display classification report
            st.write("Classification Report:")
            train_report_df = pd.DataFrame(train_report).transpose()
            st.dataframe(train_report_df.style.format({
                'precision': '{:.2f}', 
                'recall': '{:.2f}', 
                'f1-score': '{:.2f}', 
                'support': '{:.0f}'
            }).set_properties(**{'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}))
        
        with col2:
            st.subheader("Testing Results")
            st.markdown(f"""
            <div style="background-color: #e9f7fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4>Accuracy: {test_accuracy:.4f}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Rock', 'Mine'], yticklabels=['Rock', 'Mine'], ax=ax)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Testing Confusion Matrix')
            st.pyplot(fig)
            
            # Display classification report
            st.write("Classification Report:")
            test_report_df = pd.DataFrame(test_report).transpose()
            st.dataframe(test_report_df.style.format({
                'precision': '{:.2f}', 
                'recall': '{:.2f}', 
                'f1-score': '{:.2f}', 
                'support': '{:.0f}'
            }).set_properties(**{'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}))
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': [f'Feature {i+1}' for i in range(len(st.session_state.model.coef_[0]))],
            'Importance': np.abs(st.session_state.model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), 
                   palette='viridis', ax=ax)
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.warning("Please upload the sonar dataset to train the model.")

# Make Prediction page
elif page == "Make Prediction":
    st.header("Make Prediction")
    
    if st.session_state.model is not None and st.session_state.scaler is not None:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
            <h3>Enter Sonar Readings</h3>
            <p>Input the 60 sonar frequency readings (values between 0.0 and 1.0) to classify the object as a rock or a mine.</p>
            <p>You can either:</p>
            <ol>
                <li>Enter values manually using the sliders</li>
                <li>Upload a CSV file with a single row of 60 values</li>
                <li>Use a sample from the dataset</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio("Choose input method:", ["Use Sample", "Manual Input", "Upload CSV"])
        
        input_values = []
        
        if input_method == "Use Sample":
            # Select a random sample from the dataset
            if st.button("Generate Random Sample"):
                sample_idx = np.random.randint(0, len(st.session_state.data))
                sample = st.session_state.data.iloc[sample_idx].drop(60)
                input_values = sample.values.tolist()
                
                # Display the sample class
                true_class = st.session_state.data.iloc[sample_idx][60]
                st.markdown(f"""
                <div style="background-color: #e9f7fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong>True class of this sample:</strong> {'Mine' if true_class == 'M' else 'Rock'}
                </div>
                """, unsafe_allow_html=True)
                
                # Display the sample values
                fig, ax = plt.subplots(figsize=(12, 4))
                plt.plot(range(60), input_values)
                plt.title("Sample Sonar Readings")
                plt.xlabel("Feature Index")
                plt.ylabel("Value")
                st.pyplot(fig)
        
        elif input_method == "Manual Input":
            # Allow users to enter values with sliders
            st.write("Adjust the sliders to set the sonar frequency values:")
            
            # Create 6 rows of 10 sliders each for better organization
            for i in range(6):
                cols = st.columns(10)
                for j in range(10):
                    feature_idx = i * 10 + j
                    with cols[j]:
                        value = st.slider(f"F{feature_idx+1}", 0.0, 1.0, 0.5, key=f"slider_{feature_idx}")
                        input_values.append(value)
        
        elif input_method == "Upload CSV":
            # Allow users to upload a CSV file with a single row of 60 values
            uploaded_prediction_file = st.file_uploader("Upload CSV with a single row of 60 values", type=["csv"])
            
            if uploaded_prediction_file is not None:
                try:
                    prediction_data = pd.read_csv(uploaded_prediction_file, header=None)
                    if prediction_data.shape[1] < 60:
                        st.error(f"CSV file must contain 60 values. Found only {prediction_data.shape[1]} values.")
                    else:
                        input_values = prediction_data.iloc[0, :60].values.tolist()
                        
                        # Display the uploaded values
                        fig, ax = plt.subplots(figsize=(12, 4))
                        plt.plot(range(60), input_values)
                        plt.title("Uploaded Sonar Readings")
                        plt.xlabel("Feature Index")
                        plt.ylabel("Value")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        # Make prediction
        if len(input_values) == 60:
            if st.button("Predict", key="predict_button"):
                label, probability = predict_input(st.session_state.model, st.session_state.scaler, input_values)
                
                # Display prediction with probability
                mine_prob = probability[0][list(st.session_state.model.classes_).index("M")] if "M" in st.session_state.model.classes_ else probability[0][1]
                rock_prob = probability[0][list(st.session_state.model.classes_).index("R")] if "R" in st.session_state.model.classes_ else probability[0][0]
                
                # Create a visually appealing display for the prediction
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction Result:")
                    card_class = "mine-card" if label == "Mine" else "rock-card"
                    st.markdown(
                        f"<div class='prediction-card {card_class}'>"
                        f"<h2 style='color: {'#dc3545' if label == 'Mine' else '#28a745'}; text-align: center;'>{label}</h2>"
                        f"</div>", unsafe_allow_html=True
                    )
                
                with col2:
                    # Display confidence percentages
                    st.subheader("Prediction Confidence:")
                    
                    # Mine probability
                    st.write("Mine:")
                    mine_pct = mine_prob * 100
                    st.progress(mine_prob)
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>{mine_pct:.2f}%</div>", unsafe_allow_html=True)
                    
                    # Rock probability
                    st.write("Rock:")
                    rock_pct = rock_prob * 100
                    st.progress(rock_prob)
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>{rock_pct:.2f}%</div>", unsafe_allow_html=True)
                
                # Feature visualization
                st.subheader("Feature Visualization")
                fig, ax = plt.subplots(figsize=(12, 4))
                plt.plot(range(60), input_values)
                plt.title(f"Sonar Reading Pattern (Predicted: {label})")
                plt.xlabel("Feature Index")
                plt.ylabel("Value")
                st.pyplot(fig)
                
                # Display feature contribution
                if hasattr(st.session_state.model, 'coef_'):
                    st.subheader("Feature Contribution to Prediction")
                    
                    # Calculate feature contribution
                    scaled_input = st.session_state.scaler.transform(np.array(input_values).reshape(1, -1))[0]
                    coef = st.session_state.model.coef_[0]
                    contributions = scaled_input * coef
                    
                    # Create a DataFrame for feature contributions
                    contribution_df = pd.DataFrame({
                        'Feature': [f'Feature {i+1}' for i in range(60)],
                        'Contribution': contributions,
                        'Absolute': np.abs(contributions)
                    }).sort_values('Absolute', ascending=False)
                    
                    # Display top contributing features
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_features = contribution_df.head(15)
                    sns.barplot(x='Contribution', y='Feature', data=top_features, 
                               palette=['#dc3545' if x < 0 else '#28a745' for x in top_features['Contribution']], ax=ax)
                    plt.title('Top 15 Features Contributing to Prediction')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        
    else:
        st.warning("Please upload the sonar dataset to train the model.")

# Add footer
st.markdown("""
<div class="footer">
    <h4>About</h4>
    <p>This app demonstrates how machine learning can be used for sonar-based rock vs mine classification.
    Built with Streamlit, scikit-learn, and Python.</p>
</div>
""", unsafe_allow_html=True)
