# 💎 Diamond Price Predictor

An advanced machine learning-powered web application for predicting diamond prices with comprehensive model analysis and comparison. Features a sleek black and white themed interface built with Streamlit, offering multiple ML algorithms and detailed performance analytics.

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
- **Advanced Data Preprocessing** with outlier detection and feature engineering
- **Comprehensive Model Analysis** with performance comparison

## 🎯 Model Performance Comparison

Our comprehensive analysis evaluated multiple machine learning algorithms:

| Model | Train Score | Test Score | Overfitting | Rank | Status |
|-------|-------------|------------|-------------|------|---------|
| **KNN** | 97.76% | 96.55% | 1.21% | 🥇 #1 | 🏆 Best Performer |
| **Linear Regression** | 90.13% | 90.61% | -0.48% | 🥈 #2 | ✅ Most Balanced |
| **Decision Tree** | 87.99% | 88.87% | -0.88% | 🥉 #3 | ✅ Good Performance |
| **Random Forest** | 81.56% | 82.25% | -0.69% | 4th | ⚠️ Needs Tuning |

### Key Insights:
- **KNN Algorithm** selected as primary model for superior performance
- **Excellent Generalization** with minimal overfitting across all models
- **Robust Feature Engineering** contributing to consistent performance
- **53,940 training samples** after data cleaning and preprocessing

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

### Data Preprocessing Pipeline
- **Dataset**: 53,940 diamond records from comprehensive diamond database
- **Data Cleaning**: 
  - Removed duplicate entries and unnamed columns
  - Zero value imputation using median replacement for x, y, z dimensions
  - Outlier removal using statistical thresholds
- **Feature Engineering**:
  - Categorical encoding for cut, color, and clarity grades
  - Correlation analysis and feature selection (removed 'depth' due to low correlation)
  - Standardization for numerical features
- **Quality Filters Applied**:
  - Carat: < 1.9 carats
  - Table: 53-61% range
  - Dimensions: x,y < 9.2mm, z between 1.2-5.8mm

### Model Architecture & Training
- **Primary Algorithm**: K-Nearest Neighbors (k=5) with StandardScaler pipeline
- **Alternative Models**: Linear Regression, Decision Tree, Random Forest tested
- **Validation Strategy**: 75/25 train-test split (random_state=44)
- **Feature Set**: 9 engineered features after preprocessing
- **Performance Metrics**: MSE, R², accuracy scores with comprehensive visualization

### Advanced Analytics
- **Correlation Heatmaps**: Feature relationship analysis
- **Distribution Plots**: Univariate analysis for all variables
- **Regression Analysis**: Price relationship visualization
- **Box Plot Analysis**: Outlier detection and removal
- **Performance Matrix**: Multi-dimensional model comparison with rankings

### UI/UX Architecture
- **Framework**: Streamlit with custom CSS styling
- **Typography**: Orbitron font for futuristic aesthetics
- **Animations**: CSS keyframes for floating and glowing effects
- **Responsive Design**: Mobile-first approach with flexbox layouts
- **Interactive Elements**: Plotly charts with hover effects and animations

## 📁 Project Structure

```
DiamondPricePredictor/
├── app.py                 # Main Streamlit web application
├── demo.py               # Demo script for quick testing
├── requirements.txt      # Python dependencies
├── README.md            # Comprehensive project documentation
├── run_app.bat          # Windows batch file for easy launch
├── Diamonds.ipynb       # Complete ML analysis notebook with:
│                        #   - Data preprocessing & cleaning
│                        #   - Exploratory data analysis
│                        #   - Model training & comparison
│                        #   - Performance visualization
└── Diamond_Dataset/
    └── diamonds.csv     # Training dataset (53,940+ records)
```

## 📊 Data Analysis Highlights

### Dataset Characteristics
- **Total Records**: 53,940 diamonds after preprocessing
- **Features**: 10 original features reduced to 9 after feature selection
- **Target Variable**: Price (US Dollars)
- **Data Quality**: High-quality dataset with comprehensive cleaning

### Key Findings from EDA
- **Cut Distribution**: Ideal (40%), Premium (25%), Very Good (20%)
- **Color Distribution**: G, H, E, F colors most common
- **Clarity Distribution**: SI1 and VS2 most frequent
- **Price Range**: Wide distribution with right-skewed pattern
- **Feature Correlations**: Strong positive correlation between carat weight and price

### Model Insights
- **Best Performer**: KNN with 96.55% test accuracy
- **Most Balanced**: Linear Regression with minimal overfitting
- **Feature Importance**: Carat weight, dimensions (x,y,z), and cut quality most predictive
- **Generalization**: All models show excellent generalization capabilities

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

### Model Improvements
- **Ensemble Methods**: Implement voting classifiers combining top performers
- **Hyperparameter Tuning**: Grid search optimization for all algorithms
- **Deep Learning**: Neural network implementation for complex patterns
- **Feature Engineering**: Additional derived features and polynomial terms

### Application Features
- **Real-time Market Data**: Integration with live diamond market prices
- **Advanced Visualizations**: 3D scatter plots and interactive dashboards
- **Model Interpretability**: SHAP values and feature importance explanations
- **A/B Testing**: Multiple model deployment with performance tracking

### Technical Enhancements
- **API Development**: RESTful API for external integrations
- **Mobile Application**: Native iOS/Android app development
- **Database Integration**: PostgreSQL/MongoDB for data persistence
- **Docker Containerization**: Easy deployment and scaling
- **CI/CD Pipeline**: Automated testing and deployment workflows

### Analytics & Monitoring
- **Model Drift Detection**: Continuous monitoring of model performance
- **User Analytics**: Usage patterns and prediction accuracy tracking
- **Performance Dashboards**: Real-time metrics and KPI monitoring
- **Automated Retraining**: Scheduled model updates with new data

---

**Built with ❤️ using Python, Scikit-learn, Streamlit, and modern ML practices**

*Last Updated: July 2025 | Model Version: 2.0 | Dataset: 53,940 diamonds*
