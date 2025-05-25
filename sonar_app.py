import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Sonar Rock vs Mine Classifier",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #e2e8f0;
        --border-color: #e5e7eb;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        background: var(--bg-primary);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-lg);
        margin: 2rem auto;
        border: 1px solid var(--border-color);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    /* Headers */
    h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    h2 {
        font-size: 1.875rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
        color: var(--secondary-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border-right: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
    }
    
    .css-1d391kg .stRadio > label {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .stRadio > div:hover {
        background: var(--bg-primary);
        box-shadow: var(--shadow-sm);
        transform: translateY(-1px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--bg-secondary);
        border: 2px dashed var(--primary-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: var(--bg-primary);
        border-color: var(--secondary-color);
        transform: scale(1.02);
    }
    
    /* Metrics and info boxes */
    .stMetric {
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        padding: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: var(--radius-sm);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        padding: 0.5rem;
    }
    
    .stSlider > div > div > div > div {
        color: var(--primary-color);
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-color), #34d399);
        color: white;
        border-radius: var(--radius-md);
        padding: 1rem;
        border: none;
        box-shadow: var(--shadow-md);
    }
    
    .stWarning {
        background: linear-gradient(135deg, var(--warning-color), #fbbf24);
        color: white;
        border-radius: var(--radius-md);
        padding: 1rem;
        border: none;
        box-shadow: var(--shadow-md);
    }
    
    .stError {
        background: linear-gradient(135deg, var(--error-color), #f87171);
        color: white;
        border-radius: var(--radius-md);
        padding: 1rem;
        border: none;
        box-shadow: var(--shadow-md);
    }
    
    /* Custom cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: var(--radius-xl);
        text-align: center;
        box-shadow: var(--shadow-lg);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-color);
    }
    
    /* Columns */
    .stColumns {
        gap: 2rem;
    }
    
    /* Text areas and inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-md);
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Matplotlib figure styling */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        background: var(--bg-primary);
        padding: 1rem;
        border: 1px solid var(--border-color);
    }
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
            margin: 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        .stColumns {
            gap: 1rem;
        }
    }
    
    /* Footer styling */
    .footer {
        background: var(--bg-secondary);
        border-top: 1px solid var(--border-color);
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: var(--radius-lg);
        color: var(--text-secondary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Sonar Rock vs Mine Classification")
st.markdown("""
<div class="animate-fade-in">
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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "Make Prediction"])

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

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload sonar data CSV file", type=["csv"])

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
    
    st.write("""
    ### How it works
    This application uses a Logistic Regression model to classify sonar signals as either rocks or mines.
    
    ### The Data
    The dataset consists of 208 patterns obtained by bouncing sonar signals off a metal cylinder (mine) 
    and rocks under various conditions. Each pattern is a set of 60 numbers in the range 0.0 to 1.0.
    Each number represents the energy within a particular frequency band, integrated over a certain period of time.
    
    ### Instructions
    1. Upload the sonar data CSV file using the sidebar
    2. Explore the data in the "Data Exploration" section
    3. Check model performance in the "Model Performance" section
    4. Make predictions with your own inputs in the "Make Prediction" section
    """)
    
    if st.session_state.data is not None:
        st.write("### Sample Data")
        st.dataframe(st.session_state.data.head())
    
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
        
        st.write("\n".join(buffer))
        
        # Display class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.figure(figsize=(6, 4)), plt.subplot(111)
        sns.countplot(x=st.session_state.data[60], ax=ax)
        plt.title("Distribution of Rocks and Mines")
        plt.xlabel("Class")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        # Display feature visualization
        st.subheader("Feature Visualization")
        
        # Average feature values by class
        mines = st.session_state.data[st.session_state.data[60] == "M"].drop(60, axis=1).mean()
        rocks = st.session_state.data[st.session_state.data[60] == "R"].drop(60, axis=1).mean()
        
        fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
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
        
        fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
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
            st.write(f"Accuracy: {train_accuracy:.4f}")
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.figure(figsize=(8, 6)), plt.subplot(111)
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
            }))
        
        with col2:
            st.subheader("Testing Results")
            st.write(f"Accuracy: {test_accuracy:.4f}")
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.figure(figsize=(8, 6)), plt.subplot(111)
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
            }))
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': [f'Feature {i}' for i in range(len(st.session_state.model.coef_[0]))],
            'Importance': np.abs(st.session_state.model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), ax=ax)
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.warning("Please upload the sonar dataset to train the model.")

# Make Prediction page
elif page == "Make Prediction":
    st.header("Make Prediction")
    
    if st.session_state.model is not None and st.session_state.scaler is not None:
        st.write("""
        ### Enter Sonar Readings
        
        Input the 60 sonar frequency readings (values between 0.0 and 1.0) to classify the object as a rock or a mine.
        
        You can either:
        1. Enter values manually using the sliders
        2. Upload a CSV file with a single row of 60 values
        3. Use a sample from the dataset
        """)
        
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
                st.write(f"True class of this sample: {'Mine' if true_class == 'M' else 'Rock'}")
                
                # Display the sample values
                fig, ax = plt.figure(figsize=(12, 4)), plt.subplot(111)
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
                        fig, ax = plt.figure(figsize=(12, 4)), plt.subplot(111)
                        plt.plot(range(60), input_values)
                        plt.title("Uploaded Sonar Readings")
                        plt.xlabel("Feature Index")
                        plt.ylabel("Value")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        # Make prediction
        if len(input_values) == 60:
            if st.button("Predict"):
                label, probability = predict_input(st.session_state.model, st.session_state.scaler, input_values)
                
                # Display prediction with probability
                mine_prob = probability[0][list(st.session_state.model.classes_).index("M")] if "M" in st.session_state.model.classes_ else probability[0][1]
                rock_prob = probability[0][list(st.session_state.model.classes_).index("R")] if "R" in st.session_state.model.classes_ else probability[0][0]
                
                # Create a visually appealing display for the prediction
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction Result:")
                    st.markdown(
                        f"<div class='prediction-card'>"
                        f"<h2>{label}</h2>"
                        f"</div>", unsafe_allow_html=True
                    )
                
                with col2:
                    # Display confidence percentages
                    st.subheader("Prediction Confidence:")
                    
                    # Mine probability
                    st.write("Mine:")
                    mine_pct = mine_prob * 100
                    st.progress(mine_prob)
                    st.write(f"{mine_pct:.2f}%")
                    
                    # Rock probability
                    st.write("Rock:")
                    rock_pct = rock_prob * 100
                    st.progress(rock_prob)
                    st.write(f"{rock_pct:.2f}%")
                
                # Feature visualization
                st.subheader("Feature Visualization")
                fig, ax = plt.figure(figsize=(12, 4)), plt.subplot(111)
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
                    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
                    top_features = contribution_df.head(15)
                    sns.barplot(x='Contribution', y='Feature', data=top_features, 
                               palette=['red' if x < 0 else 'green' for x in top_features['Contribution']], ax=ax)
                    plt.title('Top 15 Features Contributing to Prediction')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        
    else:
        st.warning("Please upload the sonar dataset to train the model.")

# Add footer
st.markdown("""
<div class="footer">
<h3>About</h3>
<p>This app demonstrates how machine learning can be used for sonar-based rock vs mine classification.</p>
<p>Built with Streamlit, scikit-learn, and Python.</p>
</div>
""", unsafe_allow_html=True)
