import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import warnings
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import time
warnings.filterwarnings('ignore')

# Set page config for black and white theme
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Animation URLs
diamond_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_t9gkkhz4.json")
loading_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_szlepvdj.json")
success_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# Custom CSS for black and white theme with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

    .main {
        background: #000000;
        color: #FFFFFF;
        font-family: 'Orbitron', monospace;
    }

    .stApp {
        background: #000000;
        color: #FFFFFF;
    }

    /* Animated title */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        color: #FFFFFF;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 1rem;
    }

    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8), 0 0 40px rgba(255, 255, 255, 0.6); }
    }

    /* Diamond logo animation */
    .diamond-logo {
        width: 80px;
        height: 80px;
        margin: 0 auto 1.5rem;
        position: relative;
        animation: rotate 3s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .diamond-shape {
        width: 0;
        height: 0;
        border-left: 40px solid transparent;
        border-right: 40px solid transparent;
        border-bottom: 24px solid #FFFFFF;
        position: relative;
        margin: 0 auto;
        filter: drop-shadow(0 0 15px rgba(255, 255, 255, 0.6));
    }

    .diamond-shape:after {
        content: '';
        position: absolute;
        left: -40px;
        top: 24px;
        width: 0;
        height: 0;
        border-left: 40px solid transparent;
        border-right: 40px solid transparent;
        border-top: 56px solid #FFFFFF;
    }

    /* Animated cards */
    .metric-card {
        background: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 24px rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 255, 255, 0.2);
        border-color: #FFFFFF;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Pulse animation for buttons */
    .stButton > button {
        background: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 20px;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        padding: 0.6rem 1.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
    }

    .stButton > button:hover {
        background: #FFFFFF;
        color: #000000;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.4);
        animation: pulse 0.6s ease-in-out;
    }

    @keyframes pulse {
        0% { transform: scale(1) translateY(-2px); }
        50% { transform: scale(1.05) translateY(-2px); }
        100% { transform: scale(1) translateY(-2px); }
    }

    /* File uploader styling */
    .stFileUploader > div {
        background: #1a1a1a;
        border: 2px dashed #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: #FFFFFF;
        background: #2d2d2d;
        transform: scale(1.02);
    }

    /* Progress bar animation */
    .stProgress > div > div > div {
        background: #FFFFFF;
        animation: progressGlow 1.5s ease-in-out infinite alternate;
    }

    @keyframes progressGlow {
        from { box-shadow: 0 0 5px rgba(255, 255, 255, 0.5); }
        to { box-shadow: 0 0 20px rgba(255, 255, 255, 0.8); }
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #0a0a0a;
        border-right: 2px solid #FFFFFF;
    }

    /* Data table styling */
    .stDataFrame {
        background: #1a1a1a;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Metric styling */
    .stMetric {
        background: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(255, 255, 255, 0.2);
    }

    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Orbitron', monospace !important;
    }

    .stMarkdown {
        color: #FFFFFF;
        font-family: 'Orbitron', monospace;
    }

    /* Alert styling */
    .stAlert {
        background: #1a1a1a;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        color: #FFFFFF;
    }

    /* Floating elements */
    .floating {
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Sparkle effect */
    .sparkle {
        position: relative;
        overflow: hidden;
    }

    .sparkle::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: sparkle 2s linear infinite;
    }

    @keyframes sparkle {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare model
@st.cache_data
def load_training_data():
    """Load and preprocess the training data"""
    try:
        df = pd.read_csv('Diamond/diamonds.csv')

        # Data preprocessing steps from the notebook
        data = df.drop('Unnamed: 0', axis=1)
        data = data.drop_duplicates()

        # Replace zero values with median
        data['x'] = data['x'].replace(0, data['x'].median())
        data['y'] = data['y'].replace(0, data['y'].median())
        data['z'] = data['z'].replace(0, data['z'].median())

        # Mapping categorical variables
        clarity_map = {
            'I1': 1, 'SI2': 2, 'SI1': 3,
            'VS2': 4, 'VS1': 5,
            'VVS2': 6, 'VVS1': 7, 'IF': 8
        }
        data['clarity'] = data['clarity'].map(clarity_map)

        cut_map = {
            'Fair': 1, 'Good': 2, 'Very Good': 3,
            'Premium': 4, 'Ideal': 5
        }
        data['cut'] = data['cut'].map(cut_map)

        color_map = {
            'J': 1, 'I': 2, 'H': 3,
            'G': 4, 'F': 5, 'E': 6, 'D': 7
        }
        data['color'] = data['color'].map(color_map)

        # Remove depth column
        data = data.drop('depth', axis=1)

        # Remove outliers
        data = data[data['carat'] < 1.9]
        data = data[(data['table'] > 53) & (data['table'] < 61)]
        data = data[data['x'] < 9.2]
        data = data[data['y'] < 9.2]
        data = data[(data['z'] < 5.8) & (data['z'] > 1.2)]

        return data
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return None

@st.cache_resource
def train_knn_model():
    """Train the KNN model"""
    data = load_training_data()
    if data is None:
        return None

    X = data.drop(['price'], axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle=True)

    model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
    model.fit(X_train, y_train)

    return model, X_train.columns.tolist()

def preprocess_uploaded_data(df):
    """Preprocess uploaded CSV data"""
    try:
        # Remove unnamed columns if they exist
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # Check if required columns exist
        required_cols = ['carat', 'cut', 'color', 'clarity', 'table', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None

        # Remove duplicates
        df = df.drop_duplicates()

        # Replace zero values with median
        df['x'] = df['x'].replace(0, df['x'].median())
        df['y'] = df['y'].replace(0, df['y'].median())
        df['z'] = df['z'].replace(0, df['z'].median())

        # Mapping categorical variables
        clarity_map = {
            'I1': 1, 'SI2': 2, 'SI1': 3,
            'VS2': 4, 'VS1': 5,
            'VVS2': 6, 'VVS1': 7, 'IF': 8
        }

        cut_map = {
            'Fair': 1, 'Good': 2, 'Very Good': 3,
            'Premium': 4, 'Ideal': 5
        }

        color_map = {
            'J': 1, 'I': 2, 'H': 3,
            'G': 4, 'F': 5, 'E': 6, 'D': 7
        }

        # Apply mappings
        df['clarity'] = df['clarity'].map(clarity_map)
        df['cut'] = df['cut'].map(cut_map)
        df['color'] = df['color'].map(color_map)

        # Remove depth column if it exists
        if 'depth' in df.columns:
            df = df.drop('depth', axis=1)

        # Remove outliers
        df = df[df['carat'] < 1.9]
        df = df[(df['table'] > 53) & (df['table'] < 61)]
        df = df[df['x'] < 9.2]
        df = df[df['y'] < 9.2]
        df = df[(df['z'] < 5.8) & (df['z'] > 1.2)]

        # Remove price column if it exists (for prediction)
        if 'price' in df.columns:
            df = df.drop('price', axis=1)

        return df

    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def main():
    # Custom header with logo and animation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div class="diamond-logo">
            <div class="diamond-shape"></div>
        </div>
        <h1 class="main-title">Diamond Price Predictor</h1>
        <p style="font-size: 1.1rem; color: #CCCCCC; font-family: 'Orbitron', monospace;">
            ✨ Powered by K-Nearest Neighbors AI ✨
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add diamond animation
    if diamond_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(diamond_animation, height=160, key="diamond")

    st.markdown("---")

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "transparent",
                "margin": "0",
                "width": "100%",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "flex-wrap": "nowrap"
            },
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "margin": "0px 20px",
                "padding": "12px 40px",
                "color": "white",
                "background-color": "transparent",
                "border": "2px solid white",
                "border-radius": "20px",
                "font-family": "Orbitron, monospace",
                "white-space": "nowrap",
                "flex": "0 0 auto"
            },
            "nav-link-selected": {"background-color": "white", "color": "black"},
        }
    )

    if selected == "Home":
        show_home_page()
    elif selected == "Predict":
        show_prediction_page()
    elif selected == "About":
        show_about_page()

def show_home_page():
    st.markdown("""
    <div class="metric-card sparkle">
        <h2 style="text-align: center;">🌟 Welcome to Diamond Price Predictor 🌟</h2>
        <p style="text-align: center; font-size: 1.1rem;">
            Experience the power of AI-driven diamond valuation with our advanced KNN algorithm.
            Upload your diamond data and get instant, accurate price predictions!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights with animations
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card floating" style="height: 140px;">
            <h3 style="text-align: center;">🎯 96.55% Accuracy</h3>
            <p style="text-align: center;">State-of-the-art KNN model trained on thousands of diamond records</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card floating" style="animation-delay: 0.2s; height: 140px;">
            <h3 style="text-align: center;">⚡ Lightning Fast</h3>
            <p style="text-align: center;">Get instant predictions for hundreds of diamonds in seconds</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card floating" style="animation-delay: 0.4s; height: 140px;">
            <h3 style="text-align: center;">📈 Smart Analytics</h3>
            <p style="text-align: center;">Comprehensive statistics and downloadable prediction results</p>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_page():
    # Sidebar
    st.sidebar.markdown("""
    <div class="metric-card">
        <h3>🔧 Model Information</h3>
        <p><strong>Algorithm:</strong> K-Nearest Neighbors</p>
        <p><strong>Test Accuracy:</strong> 96.55%</p>
        <p><strong>Train Accuracy:</strong> 97.76%</p>
        <p><strong>Features:</strong> 8 parameters</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model with animation
    with st.spinner("🔄 Loading KNN model..."):
        if loading_animation:
            animation_placeholder = st.empty()
            animation_placeholder.markdown(
                f'<div style="text-align: center;">{st_lottie(loading_animation, height=100, key="loading")}</div>',
                unsafe_allow_html=True
            )

        model_data = train_knn_model()
        time.sleep(1)  # Brief pause for effect

        if loading_animation:
            animation_placeholder.empty()

    if model_data is None:
        st.error("❌ Failed to load the model. Please check your data file.")
        return

    model, feature_columns = model_data

    # Success animation
    if success_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(success_animation, height=150, key="success")

    st.success("✅ KNN Model loaded successfully!")

    # File upload section
    st.markdown("""
    <div class="metric-card">
        <h2 style="text-align: center;">📁 Upload Diamond Data</h2>
        <p style="text-align: center;">Choose a CSV file containing diamond data for price prediction</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: carat, cut, color, clarity, table, x, y, z"
    )

    if uploaded_file is not None:
        try:
            # Read uploaded file with progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            df = pd.read_csv(uploaded_file)
            progress_bar.empty()

            st.markdown("""
            <div class="metric-card">
                <h3>📊 Uploaded Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(df.head(), use_container_width=True)

            # Animated metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="text-align: center; color: #FFFFFF;">📊 {len(df)}</h3>
                    <p style="text-align: center;">Total Rows</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="text-align: center; color: #FFFFFF;">📋 {len(df.columns)}</h3>
                    <p style="text-align: center;">Total Columns</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="text-align: center; color: #FFFFFF;">💾 {uploaded_file.size}</h3>
                    <p style="text-align: center;">File Size (bytes)</p>
                </div>
                """, unsafe_allow_html=True)

            # Preprocess data with animation
            with st.spinner("🔄 Preprocessing data..."):
                processed_df = preprocess_uploaded_data(df.copy())
                time.sleep(1)

            if processed_df is not None and len(processed_df) > 0:
                st.success(f"✅ Data preprocessed successfully! {len(processed_df)} rows ready for prediction.")

                # Show processed data
                with st.expander("👁️ View Processed Data"):
                    st.dataframe(processed_df, use_container_width=True)

                # Prediction section
                st.markdown("""
                <div class="metric-card sparkle">
                    <h2 style="text-align: center;">🔮 Price Predictions</h2>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🚀 Predict Prices", type="primary"):
                    with st.spinner("🤖 AI is analyzing your diamonds..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)

                        try:
                            predictions = model.predict(processed_df)
                            progress_bar.empty()

                            # Success celebration
                            st.balloons()

                            results_df = processed_df.copy()
                            results_df['Predicted_Price'] = predictions.round(2)

                            st.markdown("""
                            <div class="metric-card">
                                <h3>📈 Prediction Results</h3>
                            </div>
                            """, unsafe_allow_html=True)

                            st.dataframe(results_df, use_container_width=True)

                            # Animated statistics with charts
                            create_prediction_charts(predictions)

                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="diamond_price_predictions.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"❌ Error making predictions: {str(e)}")
            else:
                st.error("❌ No valid data remaining after preprocessing. Please check your data format.")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def create_prediction_charts(predictions):
    """Create animated charts for predictions"""
    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig_hist = px.histogram(
            x=predictions,
            nbins=30,
            title="💰 Price Distribution",
            labels={'x': 'Predicted Price ($)', 'y': 'Count'},
            color_discrete_sequence=['white']
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Price statistics gauge
        avg_price = predictions.mean()
        max_price = predictions.max()

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_price,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "💎 Average Price ($)"},
            delta = {'reference': max_price/2},
            gauge = {
                'axis': {'range': [None, max_price]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, max_price/3], 'color': "#333333"},
                    {'range': [max_price/3, 2*max_price/3], 'color': "#666666"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': max_price * 0.9
                }
            }
        ))

        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card floating">
            <h3 style="text-align: center; color: #FFFFFF;">${predictions.mean():.2f}</h3>
            <p style="text-align: center;">Average Price</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.1s;">
            <h3 style="text-align: center; color: #FFFFFF;">${predictions.min():.2f}</h3>
            <p style="text-align: center;">Min Price</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.2s;">
            <h3 style="text-align: center; color: #FFFFFF;">${predictions.max():.2f}</h3>
            <p style="text-align: center;">Max Price</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.3s;">
            <h3 style="text-align: center; color: #FFFFFF;">${(predictions.max() - predictions.min()):.2f}</h3>
            <p style="text-align: center;">Price Range</p>
        </div>
        """, unsafe_allow_html=True)

def show_about_page():
    st.markdown("""
    <div class="metric-card sparkle">
        <h2 style="text-align: center;">📋 About Diamond Price Predictor</h2>
        <p style="text-align: center; font-size: 1.1rem;">
            This application uses advanced machine learning to predict diamond prices with exceptional accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Expected data format
    st.markdown("""
    <div class="metric-card">
        <h3>📋 Expected Data Format</h3>
        <p>Your CSV file should contain the following columns:</p>
    </div>
    """, unsafe_allow_html=True)

    sample_data = {
        'carat': [0.23, 0.21, 0.32],
        'cut': ['Ideal', 'Premium', 'Good'],
        'color': ['E', 'F', 'G'],
        'clarity': ['SI2', 'SI1', 'VS1'],
        'table': [55.0, 61.0, 58.0],
        'x': [3.95, 3.89, 4.05],
        'y': [3.98, 3.84, 4.07],
        'z': [2.43, 2.31, 2.31]
    }

    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

    # Feature descriptions
    with st.expander("📖 Feature Descriptions"):
        st.markdown("""
        <div class="metric-card">
            <ul style="list-style-type: none;">
                <li><strong>💎 carat:</strong> Weight of the diamond (0.2-5.01)</li>
                <li><strong>✂️ cut:</strong> Quality of the cut (Fair, Good, Very Good, Premium, Ideal)</li>
                <li><strong>🌈 color:</strong> Diamond color, from J (worst) to D (best)</li>
                <li><strong>🔍 clarity:</strong> How clear the diamond is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)</li>
                <li><strong>📏 table:</strong> Width of top of diamond relative to widest point (43-95)</li>
                <li><strong>📐 x:</strong> Length in mm (0-10.74)</li>
                <li><strong>📐 y:</strong> Width in mm (0-58.9)</li>
                <li><strong>📐 z:</strong> Depth in mm (0-31.8)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
