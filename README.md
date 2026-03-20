## Hi there 👋

3D Printer Error Detection using ML
This project implements a machine learning-based system to detect potential positional errors in a 3D printer using accelerometer data. It includes a complete pipeline from data preprocessing and model training to a live web application for predictions.

🏗️ Project Structure
ADXL345_SensorData.csv: The dataset containing accelerometer readings (X, Y, Z directions) and error labels.

LP3 - 3D Printer.ipynb: Jupyter notebook detailing the exploratory data analysis (EDA), data cleaning, and model training process.

app.py: A Flask web application to serve the model and provide a user interface for real-time error detection.

model.pkl: The trained Support Vector Classifier (SVC) model.

scaler.pkl: The MinMaxScaler used to normalize input features.

printergridsvc.pkl: A saved version of the model optimized via Grid Search.

📊 Dataset Overview
The model is trained on sensor data from an ADXL345 accelerometer mounted on a 3D printer.

Features: X-direction, Y-direction, Z-direction (float values).

Target: Error_found (yes or no, converted to binary 1 and 0).

🛠️ Data Pipeline
Preprocessing: Categorical target labels are converted to binary format.

Outlier Handling: Outliers are managed using the percentile method (5th and 95th percentiles) to ensure model robustness.

Feature Scaling: Features are normalized using MinMaxScaler to bring them into a uniform range.

Modeling: A Support Vector Machine (SVM) was utilized, with hyperparameter tuning performed via GridSearchCV.

🚀 Deployment (Flask App)
The project includes a web interface that allows users to input current X, Y, and Z coordinates to check for errors.

How to Run:

Install dependencies:

Bash
pip install flask numpy pandas scikit-learn
Run the application:

Bash
python app.py
Access the app at http://127.0.0.1:5000/ in your browser.

📈 Model Performance
The model was evaluated using accuracy scores and confusion matrices within the Jupyter notebook to ensure reliable error detection.

Author: Yash Bavale
Niche: Artificial Intelligence & Machine Learning<!--
**Yash-Bavale/yash-bavale** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.


