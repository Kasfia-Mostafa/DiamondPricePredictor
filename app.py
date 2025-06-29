import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config for black and white theme
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black and white theme
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
    }
    
    .stFileUploader > div {
        background-color: #1a1a1a;
        border: 2px dashed #FFFFFF;
    }
    
    .stButton > button {
        background-color: #FFFFFF;
        color: #000000;
        border: 2px solid #FFFFFF;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
    }
    
    .stDataFrame {
        background-color: #1a1a1a;
        color: #FFFFFF;
    }
    
    .stMetric {
        background-color: #1a1a1a;
        border: 1px solid #FFFFFF;
        padding: 10px;
        border-radius: 5px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    .stMarkdown {
        color: #FFFFFF;
    }
    
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
    }
    
    .stSidebar {
        background-color: #0a0a0a;
    }
    
    .css-1d391kg {
        background-color: #0a0a0a;
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
    st.title("💎 Diamond Price Predictor")
    st.markdown("### Predict diamond prices using K-Nearest Neighbors (KNN) algorithm")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Model Information")
    st.sidebar.markdown("**Model Type:** K-Nearest Neighbors")
    st.sidebar.markdown("**Accuracy:** 96.55%")
    st.sidebar.markdown("**Features:** 8 parameters")
    st.sidebar.markdown("---")
    
    # Load model
    with st.spinner("Loading KNN model..."):
        model_data = train_knn_model()
    
    if model_data is None:
        st.error("Failed to load the model. Please check your data file.")
        return
    
    model, feature_columns = model_data
    st.success("✅ KNN Model loaded successfully!")
    
    # File upload section
    st.header("📁 Upload Diamond Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing diamond data",
        type=['csv'],
        help="Upload a CSV file with columns: carat, cut, color, clarity, table, x, y, z"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size} bytes")
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                processed_df = preprocess_uploaded_data(df.copy())
            
            if processed_df is not None and len(processed_df) > 0:
                st.success(f"✅ Data preprocessed successfully! {len(processed_df)} rows ready for prediction.")
                
                # Show processed data
                with st.expander("View Processed Data"):
                    st.dataframe(processed_df, use_container_width=True)
                
                # Prediction section
                st.header("🔮 Price Predictions")
                
                if st.button("🚀 Predict Prices", type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            # Make predictions
                            predictions = model.predict(processed_df)
                            
                            # Create results dataframe
                            results_df = processed_df.copy()
                            results_df['Predicted_Price'] = predictions.round(2)
                            
                            st.subheader("📈 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average Price", f"${predictions.mean():.2f}")
                            with col2:
                                st.metric("Min Price", f"${predictions.min():.2f}")
                            with col3:
                                st.metric("Max Price", f"${predictions.max():.2f}")
                            with col4:
                                st.metric("Price Range", f"${(predictions.max() - predictions.min()):.2f}")
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="diamond_price_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
            else:
                st.error("❌ No valid data remaining after preprocessing. Please check your data format.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Sample data format
    st.header("📋 Expected Data Format")
    st.markdown("Your CSV file should contain the following columns:")
    
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
        - **carat**: Weight of the diamond (0.2-5.01)
        - **cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
        - **color**: Diamond color, from J (worst) to D (best)
        - **clarity**: How clear the diamond is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
        - **table**: Width of top of diamond relative to widest point (43-95)
        - **x**: Length in mm (0-10.74)
        - **y**: Width in mm (0-58.9)
        - **z**: Depth in mm (0-31.8)
        """)

if __name__ == "__main__":
    main()
