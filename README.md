# 💎 Diamond Price Predictor

A black and white themed web application for predicting diamond prices using K-Nearest Neighbors (KNN) machine learning algorithm.

## Features

- **Upload CSV files** with diamond data for batch predictions
- **KNN Model** with 96.55% accuracy 
- **Black & White UI** theme for elegant appearance
- **Data preprocessing** including outlier removal and categorical encoding
- **Downloadable results** in CSV format
- **Interactive data visualization** and statistics

## Installation

1. The Python environment has been configured. Install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn
```

## Usage

1. Run the Streamlit application using one of these methods:

   **Method 1 - Using the batch file:**
   ```bash
   run_app.bat
   ```

   **Method 2 - Using command line:**
   ```bash
   "f:/Machine Learning/DiamondPricePredictor/.venv/Scripts/streamlit.exe" run app.py
   ```

   **Method 3 - Using Python module:**
   ```bash
   "f:/Machine Learning/DiamondPricePredictor/.venv/Scripts/python.exe" -m streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Upload a CSV file containing diamond data with the following columns:
   - `carat`: Weight of the diamond
   - `cut`: Quality of cut (Fair, Good, Very Good, Premium, Ideal)
   - `color`: Diamond color (D, E, F, G, H, I, J)
   - `clarity`: Clarity grade (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
   - `table`: Table percentage
   - `x`: Length in mm
   - `y`: Width in mm
   - `z`: Depth in mm

4. Click "Predict Prices" to get price predictions for all diamonds in your dataset

## Data Format

Your CSV file should look like this:

| carat | cut     | color | clarity | table | x    | y    | z    |
|-------|---------|-------|---------|-------|------|------|------|
| 0.23  | Ideal   | E     | SI2     | 55.0  | 3.95 | 3.98 | 2.43 |
| 0.21  | Premium | F     | SI1     | 61.0  | 3.89 | 3.84 | 2.31 |

## Model Information

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Test Accuracy**: 96.55%
- **Train Accuracy**: 97.76%
- **Features**: 8 diamond characteristics
- **Preprocessing**: Outlier removal, categorical encoding, zero value imputation

## File Structure

```
DiamondPricePredictor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── Diamonds.ipynb     # Original analysis notebook
└── Diamond/
    └── diamonds.csv   # Training dataset
```
