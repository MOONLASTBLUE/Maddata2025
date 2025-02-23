# 🏥 Lifestyle-Based Health Prediction App

## 🔗 Streamlit app link
https://maddata2025-college-life-essentials.streamlit.app/

## 📌 Project Overview  
This project is an **AI-powered simulator that predicts weight and health status based on user lifestyle data**.  
We built a web interface using **Streamlit** and applied **machine learning models to predict BMI changes and sleep quality**.

## 📂 Dataset  
- **Data Used**: `merged_dataset2.csv`, `Sleep_health_and_lifestyle_dataset.csv`  
- **Source**: Processed survey data on health and lifestyle  
- **Key Features**:
  - **For BMI Prediction**
    - `Weight`, `Height` (Body weight, height)
    - `FAF` (Frequency of physical activity, 0-5 days)
    - `FCVC` (Frequency of vegetable consumption, 1-3)
    - `TUE` (Technology usage time, 0-10 hours)
  - **For Sleep Quality Prediction**
    - `Sleep Duration` (Hours of sleep)
    - `Physical Activity Level` (Days per week)
    - `Stress Level` (Scale of 1-10)
    - `Heart Rate`, `Systolic_BP`, `Diastolic_BP`, `Daily Steps` (Biometric data)

## 🧠 Machine Learning Models  

### 1️⃣ **Obesity Prediction (BMI Prediction)**  
- **Model Used**: `MLPRegressor`
- **Input Variables**: Weight, height, physical activity frequency, vegetable consumption, technology usage  
- **Output**: Predicted BMI score and change  
- **Model Structure**:
  - Hidden Layers: (200, 150, 100, 50) with ReLU activation
  - Optimizer: Adam
  - Learning Rate: 0.0065
  - Regularization (alpha): 0.0014  
- **Performance Metrics**: Mean Squared Error (MSE), R² Score  

### 2️⃣ **Sleep Quality Prediction**  
- **Model Used**: `Optimized Random Forest Regressor`
- **Input Variables**: Sleep duration, stress levels, physical activity, heart rate, blood pressure, daily steps  
- **Output**: Predicted sleep quality score (0-10 scale)  
- **Model Structure**:
  - `n_estimators=100`, `max_depth=10`, `max_features='sqrt'`
  - `min_samples_split=2`, `min_samples_leaf=1`  
- **Performance Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score  

## 🎨 Web Application (Streamlit)
### **Features**
1. **📊 BMI Change Prediction**  
   - Users input height, weight, and lifestyle habits to predict BMI changes  
   - `MLPRegressor` model provides results  

2. **🛌 Sleep Quality Prediction**  
   - Predicts sleep quality score based on lifestyle habits  
   - `Optimized Random Forest` model estimates the score (0-10)  

3. **🏋️‍♂️ UW Madison Fitness Center Map**  
   - Displays nearby gyms and fitness centers on an interactive map  
   - Uses `folium` for visualization  

4. **📚 Health-Related Course Recommendations**  
   - Provides **DANCE, KINESIOLOGY, and FOOD SCIENCE** courses at UW-Madison  
   - Helps users find courses aligned with their health goals  
