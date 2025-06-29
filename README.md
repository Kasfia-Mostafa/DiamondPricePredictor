# 💎 Diamond Price Predictor

A sleek black and white themed web application for predicting diamond prices using K-Nearest Neighbors (KNN) machine learning algorithm with professional UI/UX design.

## ✨ Features

- **Pure Black & White Theme** with elegant monochrome design
- **Professional Navigation Bar** with centered buttons and optimal spacing
- **Animated Diamond Logo** with rotating diamond shape and glowing effects
- **Lottie Animations** for loading states and visual feedback
- **CSV Batch Processing** for multiple diamond price predictions
- **Interactive Charts** using Plotly with black & white color scheme
- **Real-time Statistics** with floating animated cards
- **Progress Indicators** with custom animations
- **Responsive Design** optimized for all screen sizes
- **Model Information Sidebar** with accuracy metrics
- **Downloadable Results** in CSV format
- **Data Validation** with automatic preprocessing

## 🎯 Model Performance

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Test Accuracy**: 96.55%
- **Train Accuracy**: 97.76%
- **Features**: 8 diamond characteristics
- **Training Data**: Thousands of diamond records

## 🚀 Quick Start

### Using the Batch File (Recommended)
Simply double-click `run_app.bat` to automatically install dependencies and launch the app.

### Manual Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Requirements

The application requires the following Python packages:
- streamlit==1.28.1
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- streamlit-lottie==0.0.5
- requests==2.31.0
- plotly==5.17.0
- streamlit-option-menu==0.3.6
- streamlit-extras==0.3.5
- seaborn==0.12.2
- matplotlib==3.7.2

## 📊 Usage Guide

1. **Launch the Application**
   - Run `run_app.bat` or use `streamlit run app.py`
   - Navigate to `http://localhost:8501` in your browser

2. **Navigate Through Pages**
   - **Home**: Overview of features and model information
   - **Predict**: Upload CSV files and get predictions
   - **About**: Data format requirements and feature descriptions

3. **Upload Diamond Data**
   - Prepare a CSV file with required columns (see format below)
   - Upload via the file uploader in the Predict section
   - Review data preview and statistics

4. **Get Predictions**
   - Click "🚀 Predict Prices" button
   - View results with interactive charts
   - Download predictions as CSV file
## 📝 Data Format Requirements

Your CSV file must contain the following columns:

| Column  | Description | Valid Values |
|---------|-------------|--------------|
| `carat` | Weight of the diamond (0.2-5.01) | Decimal numbers |
| `cut` | Quality of the cut | Fair, Good, Very Good, Premium, Ideal |
| `color` | Diamond color grade | D, E, F, G, H, I, J (D=best, J=worst) |
| `clarity` | Clarity grade | IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1 |
| `table` | Table percentage (43-95) | Decimal numbers |
| `x` | Length in mm (0-10.74) | Decimal numbers |
| `y` | Width in mm (0-58.9) | Decimal numbers |
| `z` | Depth in mm (0-31.8) | Decimal numbers |

### Sample Data Format
```csv
carat,cut,color,clarity,table,x,y,z
0.23,Ideal,E,SI2,55.0,3.95,3.98,2.43
0.21,Premium,F,SI1,61.0,3.89,3.84,2.31
0.32,Good,G,VS1,58.0,4.05,4.07,2.31
```

## 🔧 Technical Details

### Data Preprocessing
- **Outlier Removal**: Removes diamonds with extreme measurements
- **Zero Value Imputation**: Replaces zero values with median values
- **Categorical Encoding**: Maps categorical variables to numerical values
- **Data Validation**: Ensures all required columns are present

### Model Training
- **Algorithm**: K-Nearest Neighbors with 5 neighbors
- **Features**: 8 diamond characteristics after preprocessing
- **Validation**: 75/25 train-test split with random state 44

### UI/UX Features
- **Orbitron Font**: Futuristic monospace typography
- **Smooth Animations**: CSS keyframes for floating and glowing effects
- **Responsive Layout**: Adapts to different screen sizes
- **Interactive Elements**: Hover effects and click animations

## 📁 Project Structure

```
DiamondPricePredictor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── run_app.bat        # Windows batch file for easy launch
├── Diamonds.ipynb     # Jupyter notebook with analysis
└── Diamond/
    └── diamonds.csv   # Training dataset
```

## 🎨 Design Philosophy

The application follows a strict **black and white** design philosophy:
- **Pure Black Backgrounds** (#000000)
- **White Text and Borders** (#FFFFFF) 
- **Gray Accents** (#1a1a1a, #2d2d2d) for depth
- **No Color Elements** - maintaining monochrome aesthetics
- **High Contrast** for optimal readability

## 🚀 Performance Optimizations

- **Streamlit Caching**: Model training and data loading are cached
- **Efficient Data Processing**: Optimized pandas operations
- **Minimal Dependencies**: Only essential packages included
- **Fast Predictions**: KNN algorithm provides quick inference

## 📈 Future Enhancements

- Additional machine learning models (Random Forest, XGBoost)
- Real-time price tracking and market analysis
- Mobile app version
- API endpoints for external integrations
- Advanced data visualization options

---

**Built with ❤️ using Streamlit, scikit-learn, and modern web technologies**
