# 💎 Diamond Price Predictor

A machine learning web application for predicting diamond prices with 96.55% accuracy using KNN algorithm. Features a sleek black & white interface with interactive visualizations and comprehensive model analysis.

![image alt](https://github.com/Kasfia-Mostafa/DiamondPricePredictor/blob/main/Diamaond.jpg?raw=true)

## 🎯 Key Features

- 🏆 **96.55% Accuracy** with KNN algorithm trained on 53,940+ diamonds
- � **4 ML Models Compared**: KNN, Linear Regression, Decision Tree, Random Forest
- 🎨 **Professional UI** with black & white theme and interactive charts
- 📈 **CSV Batch Processing** for multiple diamond predictions
- 🔧 **Advanced Preprocessing** with outlier detection and feature engineering

## 🚀 Quick Start

```bash
# Option 1: One-click launch (Windows)
run_app.bat

# Option 2: Manual installation
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## � Model Performance

| Model | Test Accuracy | Status |
|-------|---------------|---------|
| **KNN** | 96.55% | 🏆 Selected |
| Linear Regression | 90.61% | ✅ Good |
| Decision Tree | 88.87% | ✅ Good |
| Random Forest | 82.25% | ⚠️ Needs Tuning |

## 📝 Data Format

Your CSV file should contain these columns:
- `carat`: Diamond weight (0.2-5.01)
- `cut`: Fair, Good, Very Good, Premium, Ideal
- `color`: D, E, F, G, H, I, J (D=best)
- `clarity`: IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1
- `table`: Table percentage (43-95)
- `x`, `y`, `z`: Dimensions in mm

**Example:**
```csv
carat,cut,color,clarity,table,x,y,z
0.23,Ideal,E,SI2,55.0,3.95,3.98,2.43
0.21,Premium,F,SI1,61.0,3.89,3.84,2.31
```

## 🔧 Technical Details

### Data Processing
- **Dataset**: 53,940 diamond records after cleaning
- **Features**: 9 engineered features (removed 'depth' due to low correlation)
- **Preprocessing**: Outlier removal, zero imputation, categorical encoding
- **Validation**: 75/25 train-test split

### Model Architecture
- **Primary**: KNN (k=5) with StandardScaler
- **Pipeline**: Data validation → Preprocessing → ML prediction
- **Performance**: 96.55% test accuracy, minimal overfitting (1.21%)

## 📁 Project Structure

```
DiamondPricePredictor/
├── app.py                 # Main Streamlit application
├── demo.py               # Demo script
├── requirements.txt      # Dependencies
├── run_app.bat          # Windows launcher
├── Diamonds.ipynb       # Analysis notebook
└── Diamond_Dataset/
    └── diamonds.csv     # Training data
```

## 📈 Future Enhancements

- Ensemble methods and hyperparameter tuning
- Real-time market data integration
- Mobile app development
- API endpoints for external integrations
- Advanced visualizations and model interpretability

