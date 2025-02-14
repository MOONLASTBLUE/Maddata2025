import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page titles and navigation
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Navigate", 
    ["How to Use the App", "About Obesity", "Make Prediction", "Learn More about Obesity", "About Project"]
)

# Page 1: Home & How to Use the App
if page == "How to Use the App":
    st.title("Obesity Level Prediction: Eating Habits & Fitness")
    st.image("20240810_IRD001.webp", use_column_width=True)
    st.write("""
        Welcome to the Obesity Level Prediction App! This app helps you better understand and manage your health by analyzing your eating habits and physical activity.
        Use the menu on the left to explore the app's features.
        
        **What you can do here**:
        1. **Learn about obesity**: Discover factors contributing to obesity and its health impact.
        2. **Predict your obesity level**: Input your lifestyle habits for a personalized BMI prediction.
        3. **Explore project insights**: Understand the background and methods of this project.
    """)

# Page 2: About Obesity
elif page == "About Obesity":
    st.title("About Obesity")

    # Section 1: Obesity Overview
    st.markdown("""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #4CAF50; text-align: center;">Obesity Overview</h4>
            <p style="font-size: 16px;">
                <strong>Obesity</strong> is a complex health condition characterized by excessive body fat accumulation. 
                It is typically measured using the <strong>Body Mass Index (BMI)</strong>, calculated as weight in kilograms divided by height in meters squared.
            </p>
        </div>
    """, unsafe_allow_html=True)

     # Section 2: Global and Regional Obesity Trends
    with st.expander("📊 **Global and Regional Obesity Trends**", expanded=False):
        st.write("""
        1. **Lifestyle Choices**: High-calorie diets, lack of physical activity, and irregular eating patterns can lead to weight gain.
        2. **Genetics**: A family history of overweight or obesity increases the likelihood of developing obesity.
        3. **Environmental Influences**: Access to healthy food, sedentary occupations, and urban lifestyles often contribute to higher obesity rates.
        </div>
        """, unsafe_allow_html=True)
        st.image("1aa0d26994667aa8697314d0e077c9c7b53f49ca-1240x874.webp", caption="Factors Influencing Obesity", use_column_width=True)
        

    # Section 3: Key Factors Influencing Obesity
    with st.expander("🔑 **Key Factors Influencing Obesity**", expanded=False):
        st.markdown("""
        1. **Lifestyle Choices**: High-calorie diets, lack of physical activity, and irregular eating patterns can lead to weight gain.
        2. **Genetics**: A family history of overweight or obesity increases the likelihood of developing obesity.
        3. **Environmental Influences**: Access to healthy food, sedentary occupations, and urban lifestyles often contribute to higher obesity rates.
        </div>
        """, unsafe_allow_html=True)
        st.image("Factors-contributing-to-pediatric-obesity-BED-binge-eating-disorder.png", caption="Factors Influencing Obesity", use_column_width=True)

    # Section 4: Why BMI Differs Between Men and Women
    with st.expander("⚖️ **Why BMI Differs Between Men and Women**", expanded=False):
        st.markdown("""
        Men and women may show slightly different BMI levels due to variations in fat cell distribution and hormonal influences. For example:
        - **Women**:
            - Generally have higher essential body fat percentages due to reproductive functions.
            - Estrogen promotes fat storage around the hips, thighs, and buttocks, providing reserves for pregnancy and childbearing.
            - This fat distribution poses lower cardiovascular risks compared to visceral fat.
        - **Men**:
            - Tend to accumulate visceral fat, primarily around the abdomen.
            - Visceral fat is strongly associated with metabolic syndrome and cardiovascular diseases.
            - Testosterone suppresses fat accumulation in non-abdominal areas, leading to these differences.
        </div>
        """, unsafe_allow_html=True)
        st.image("bmi-for-men-and-women.webp", caption="BMI Difference between Men and Women", use_column_width=True)

    # Section 5: Health Risks Associated with Obesity
    with st.expander("⚠️ **Health Risks Associated with Obesity**", expanded=False):
        st.markdown("""
        Obesity significantly increases the risk of several chronic diseases and health conditions, including:
        
        1. **Cardiovascular Diseases**:
           - Excess body fat leads to increased blood pressure, cholesterol levels, and inflammation, which are major risk factors for heart attacks and strokes.
           - Studies show that obese individuals have a 50% higher risk of developing coronary artery disease.
        
        2. **Type 2 Diabetes**:
           - Obesity is the leading cause of insulin resistance, which can lead to type 2 diabetes.
           - Approximately 90% of individuals with type 2 diabetes are overweight or obese.
        
        3. **Hypertension (High Blood Pressure)**:
           - Excess weight puts additional strain on the heart, increasing blood pressure.
           - Obesity is estimated to contribute to 65-75% of cases of primary hypertension.
        
        4. **Certain Types of Cancer**:
           - Obesity is linked to cancers such as breast, colon, endometrial, and kidney cancer.
           - Adipose tissue produces excess estrogen and inflammation, which are associated with tumor growth.
        
        5. **Other Conditions**:
           - **Sleep Apnea**: Excess fat around the neck can block airways, causing breathing difficulties during sleep.
           - **Joint Problems**: Increased weight puts stress on joints, leading to osteoarthritis.
           - **Mental Health**: Obesity can contribute to depression and anxiety due to societal stigma and reduced quality of life.

        </div>
        """, unsafe_allow_html=True)
        st.image("369185864_614149610892096_6036807610600070169_n.jpg", caption="Health Risks of Obesity", use_column_width=True)

# Page 3: Make Prediction
elif page == "Make Prediction":
    st.title("Predict BMI Change")
    st.write("Input your current details and adjust lifestyle variables to predict BMI change.")
    
    # Age range calculation
    def age_group(age):
        if age < 10:
            return "Under 10"
        elif 10 <= age < 20:
            return "10s"
        elif 20 <= age < 30:
            return "20s"
        elif 30 <= age < 40:
            return "30s"
        elif 40 <= age < 50:
            return "40s"
        elif 50 <= age < 60:
            return "50s"
        elif 60 <= age < 70:
            return "60s"
        else:
            return "70+"


    # BMI level criteria definition
    def bmi_level(bmi, gender):
        if gender == "Male":
            if bmi < 18:
                return "Underweight", "#1f3f73"
            elif 18 <= bmi < 24:
                return "Normal", "#1a7323"
            elif 24 <= bmi < 29:
                return "Overweight", "#F0E68C"
            else:
                return "Obesity", "#9c2418"
        elif gender == "Female":
            if bmi < 17.5:
                return "Underweight", "#1f3f73"
            elif 17.5 <= bmi < 24.5:
                return "Normal", "#1a7323"
            elif 24.5 <= bmi < 30:
                return "Overweight", "#F0E68C"
            else:
                return "Obesity", "#9c2418"


    # Create personalized advice
    def generate_recommendations(weight, height, family_history, faf, fcvc, tue, bmi_level):
        recommendations = []
         # Underweight Recommendations
        if bmi_level in "Underweight":
            recommendations.append("🍽️  Consider increasing your calorie intake with nutrient-dense foods such as nuts, avocados, and lean protein sources.")
            recommendations.append("🏋️  Engage in light strength training to build muscle mass in a healthy way.")
        
        
        elif bmi_level == "Normal":
            recommendations.append( "👏 Your BMI is in the normal range. Maintain your current lifestyle for continued good health.")
            recommendations.append("💧 Ensure you stay hydrated and get regular check-ups to monitor your overall health.")

        if bmi_level == "Overweight":
            recommendations.append("🏃‍♂️ Increase your physical activity to at least 3-4 days per week. Cardio and strength exercises are highly effective.")
            if faf < 2:
                recommendations.append( "⏱️ Gradually increase your activity level starting with short walks and progressing to more intense activities.")
            recommendations.append("🥗 Incorporate more vegetables and whole grains into your meals while reducing processed food intake.")
            recommendations.append("📉 Monitor your weight regularly and set realistic goals for gradual improvement.")

        if bmi_level == "Obesity":
            recommendations.append("🩺 Consult a healthcare professional for personalized advice tailored to your specific health needs.")
            if family_history == 1:
                recommendations.append("🧬 Since you have a family history of overweight, consider genetic or metabolic screenings for additional insights.")
            recommendations.append("📉 Aim to reduce your calorie intake by focusing on portion control and balanced meals.")
            recommendations.append("⏱️ Try to gradually increase your physical activity to more than 3 days a week, focusing on low-impact exercises to start.")
            recommendations.append("🧘‍♂️ Incorporate mindfulness practices like yoga or meditation to manage stress, which can impact weight management.")

            if tue > 5:
                recommendations.append("📵 Reduce technology usage to less than 5 hours daily and spend more time engaging in physical or outdoor activities.")
                recommendations.append("📖 Consider taking regular breaks and using technology-free hours to enhance overall well-being.")

            if fcvc < 2:
                recommendations.append("🥦 Try to consume at least 2 servings of vegetables daily. Opt for colorful vegetables rich in nutrients.")
                recommendations.append("🌽 Add a variety of vegetables to your meals to ensure a balanced nutrient intake.")

        if not recommendations:
            recommendations.append("🌟 Your lifestyle habits are balanced. Keep up the good work!")

        return recommendations   
        
        
        

    @st.cache_resource
    def train_model():
        df = pd.read_csv("merged_dataset2.csv")
        
#         # 'Weight', 'Height','FAF', 'FCVC', 'TUE', 'FAVC', 'NCP', 'SMOKE', 'MTRANS'
#         df["Age_Group"] = df["Age"].apply(age_group)
#         df["Calculated_BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
        
#         X = df[['Weight', 'Height','FAF', 'FCVC', 'TUE', 'FAVC', 'NCP', 'SMOKE', 'MTRANS']]
#         y = df['Calculated_BMI']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

        # Age_Group column and Calculated_BMI column added
        df["Age_Group"] = df["Age"].apply(age_group)
        df["Calculated_BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

        X = df[['Weight', 'Height', 'FAF', 'FCVC', 'TUE']]
        y = df['Calculated_BMI']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=(200, 150, 100, 50), 
            activation='relu',
            solver='adam',
            alpha=0.0014053639037901015,
            learning_rate_init=0.006537959433035273,
            max_iter=1900
        )
        
        #{'alpha': 0.0014053639037901015, 'hidden_layer_sizes': 3, 'learning_rate_init': 0.006537959433035273, 'max_iter': 1900.0}

        model.fit(X_train, y_train)
        return scaler, model, df


    scaler, model, dataset = train_model()

    # 1: Enter height, weight, age, gender
    col1, col2 = st.columns(2)
    with col1:
        height = st.slider("Height (cm):", min_value=120.0, max_value=220.0, value=170.0, key="height_slider")
    with col2:
        weight = st.slider("Weight (kg):", min_value=30.0, max_value=150.0, value=70.0, key="weight_slider")

    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender:", ["Male", "Female"], key="gender_select")
    with col4:
        age = st.slider("Age:", min_value=0, max_value=100, value=30, key="age_slider")

    # Calculate current BMI and print with color reflected
    current_bmi = weight / ((height / 100) ** 2)
    current_level, current_color = bmi_level(current_bmi, gender)
    st.markdown(f"<h3 style='color:{current_color};'>Your Current BMI: {current_bmi:.2f} ({current_level})</h3>", unsafe_allow_html=True)

    # BMI distribution graph for the input person’s age
    age_group_label = age_group(age)
    age_group_data = dataset[dataset["Age_Group"] == age_group_label]["Calculated_BMI"]

    if not age_group_data.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(age_group_data, bins=15, color="lightblue", edgecolor="black", alpha=0.7)
        plt.axvline(current_bmi, color=current_color, linestyle="dashed", linewidth=2, label=f"Your BMI: {current_bmi:.2f}")
        plt.title(f"BMI Distribution for Your Age Group: {age_group_label}")
        plt.xlabel("BMI")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write(f"No BMI data available for the age group: {age_group_label}")
    
    
    # 2: Rest of the input variables
    col5, col6 = st.columns(2)
    with col5:
        physical_activity = st.slider("Physical Activity Frequency (days/week):", min_value=0, max_value=10, value=3, key="physical_activity_slider")
    with col6:
        vegetables = st.slider("Vegetable Consumption Frequency:", min_value=0, max_value=5, value=2, key="vegetables_slider")

    col7, col8 = st.columns(2)
    with col7:
        family_history = st.selectbox("Family History of Overweight:", ["Yes", "No"], key="family_history_select")
    with col8:
        high_calorie_food = st.selectbox("Do you consume high caloric food frequently?", ["Yes", "No"], key="calorie_food_select")
    
    col9, col10 = st.columns(2)
    with col9:
        technology_usage = st.slider("Technology Usage (hours/day):", min_value=0.0, max_value=10.0, value=3.0, key="technology_usage_slider")
    with col10:
        food_between_meals = st.selectbox("How often do you consume food between meals?", ["Never", "Sometimes", "Frequently"], key="food_between_meals_select")

    col11, col12 = st.columns(2)
    with col11:
        smoking = st.selectbox("Do you smoke?", ["Yes", "No"], key="smoking_select")
    with col12:
        alcohol = st.selectbox("Do you consume alcohol?", ["No", "Yes"], key="alcohol_select")

    col13, col14 = st.columns(2)
    with col13:
        main_meals = st.slider("Main Meals Frequency (meals/day):", min_value=0, max_value=10, value=3, key="main_meals_slider")
    with col14:
        calorie_tracking = st.selectbox("Do you track your daily calorie intake?", ["Yes", "No"], key="calorie_tracking_select")

    col15, col16 = st.columns(2)
    with col15:
        water_consumption = st.slider("Water Consumption (liters/day):", min_value=0.0, max_value=20.0, value=2.0, key="water_consumption_slider")
    with col16:
        transportation = st.selectbox("Transportation Method:", ["Walking", "Public_Transportation", "Car", "Bike"], key="transportation_select")

    # Predict with variables closely related to BMI
    if st.button("Predict BMI Change"):
        # Convert category data to numbers
        high_calorie_food_encoded = 1 if high_calorie_food == "Yes" else 0
        food_between_meals_encoded = {"Never": 0, "Sometimes": 1, "Frequently": 2}[food_between_meals]
        smoking_encoded = 1 if smoking == "Yes" else 0
        transportation_encoded = {"Walking": 0, "Public_Transportation": 1, "Car": 2, "Bike": 3}[transportation]
        
        input_data = pd.DataFrame({
            "Weight": [weight],
            "Height": [height],
            "FAF": [physical_activity],
            "FCVC": [vegetables],
            "TUE": [technology_usage]
        })

        scaled_input = scaler.transform(input_data)
        predicted_bmi = model.predict(scaled_input)[0]

        st.write(f"**Predicted BMI after Adjustments:** {predicted_bmi:.2f}")
        bmi_change = predicted_bmi - current_bmi
        st.write(f"**Change in BMI:** {bmi_change:+.2f}")

        # Print recommended comments
        level, color = bmi_level(predicted_bmi, gender)
        st.markdown(f"<h3 style='color:{color};'>Predicted BMI Level: {predicted_bmi:.2f} ({level})</h3>", unsafe_allow_html=True)

        recommendations = generate_recommendations(weight, height, 1 if family_history == "Yes" else 0,
                                                   physical_activity, vegetables, technology_usage, level)
        st.write("❕ Recommendations:")
        for rec in recommendations:
            st.markdown(f"- {rec}")

# Page 4: Learn More about Obesity
elif page == "Learn More about Obesity":
    st.title("Learn More About Obesity")

    st.write("""
        Obesity has become a global health concern, with over 650 million adults classified as obese according to the World Health Organization (WHO).
        It significantly increases the risk of chronic diseases such as type 2 diabetes, cardiovascular diseases, and certain cancers.
        Here are practical steps for overcoming obesity and leading a healthier lifestyle:
    """)

    # Section: Importance of Physical Activity
    st.subheader("🚶 Physical Activity")
    st.write("""
        Physical activity is essential for maintaining a healthy weight and reducing obesity-related risks:
        
        - **Low FAF (Frequency of Physical Activity)** (less than 2 days/week): A sedentary lifestyle increases obesity risks.
        - **Moderate FAF** (3-5 days/week): Helps balance energy intake and expenditure, maintaining body weight.
        - **High FAF** (6+ days/week): Supports significant fat burning, builds muscle mass, and improves overall health.
        
        **Tips for Increasing Physical Activity**:
        - Start small by incorporating a 10-minute daily walk and gradually increase the duration.
        - Include strength training exercises twice a week to build muscle mass.
        - Find activities you enjoy, like dancing, cycling, or swimming, to stay motivated.
    """)

    # Section: Dietary Habits
    st.subheader("🥗 Healthy Dietary Habits")
    st.write("""
        Eating a balanced diet is a key strategy for preventing and managing obesity:
        
        - **Increase Vegetable and Fruit Intake**: Aim for at least 5 servings daily to boost fiber and nutrient intake.
        - **Choose Whole Grains**: Replace refined grains with whole grains for better satiety and energy control.
        - **Limit Sugary Drinks and Snacks**: Reduce consumption of high-calorie, low-nutrient items like sodas and chips.
        - **Portion Control**: Be mindful of portion sizes to prevent overeating.
        - **Mindful Eating**: Avoid distractions while eating and listen to your body's hunger and fullness cues.
        
        **Practical Tips**:
        - Meal prep healthy meals for the week to avoid fast food temptations.
        - Use smaller plates and bowls to help with portion control.
        - Drink water before meals to reduce calorie intake.
    """)

    # Section: Stress Management
    st.subheader("🧘 Stress Management")
    st.write("""
        Stress is a common trigger for overeating and weight gain:
        
        - Practice relaxation techniques like deep breathing, yoga, or meditation.
        - Engage in hobbies or activities that bring you joy and reduce stress.
        - Prioritize sleep, as poor sleep quality can lead to weight gain.
    """)

    # Section: Technology and Screen Time
    st.subheader("📱 Reducing Screen Time")
    st.write("""
        Excessive screen time is associated with sedentary behavior and weight gain:
        
        - Limit technology usage to less than 5 hours per day.
        - Take regular breaks during long work sessions to stretch and move around.
        - Avoid eating meals in front of screens to focus on mindful eating.
    """)

    # Section: Professional Support
    st.subheader("🩺 Seek Professional Support")
    st.write("""
        For individuals struggling with obesity, professional guidance can be invaluable:
        
        - Consult a registered dietitian or nutritionist for personalized dietary advice.
        - Join a supervised exercise program designed for weight management.
        - Speak to a healthcare provider about medical treatments or interventions, such as weight-loss medications or bariatric surgery if needed.
    """)

    # Section: Setting Realistic Goals
    st.subheader("🎯 Setting Realistic Goals")
    st.write("""
        Weight loss is a gradual process that requires consistency and realistic expectations:
        
        - Aim for a sustainable weight loss of 0.5 to 1 kg (1 to 2 pounds) per week.
        - Focus on long-term lifestyle changes rather than quick fixes.
        - Celebrate small achievements, like increased energy or better sleep, along the way.
    """)

    # Section: Did You Know?
    st.subheader("💡 Did You Know?")
    st.write("""
        - Losing just 5-10% of your body weight can significantly improve your health.
        - Regular physical activity not only helps with weight management but also reduces symptoms of depression and anxiety.
        - Hydration is critical: drinking enough water supports metabolism and reduces hunger.
    """)


# Page 5: About Project
elif page == "About Project":
    st.title("About Project")
    st.write("""
        
        ### 👭 Team Members
        - **Hari Kang**
        - **Yunji Lee**
        
        ### 🏃‍♂️ Background and Purpose
        - **Importance of Obesity and Health Management**:
          - Globally, more than 1 billion people were classified as obese in 2022.
          - In South Korea, adult obesity rates rose from 35.1% in 2011 to 46.3% in 2021, highlighting the increasing need for preventive health management. (Source: WHO)
        - **Need for Personalized Health Management Technology**:
          - 89% of South Koreans recognize the importance of preventive health management, with 59% preferring personalized solutions. 
          - This project offers tools to calculate BMI and predict health improvements through lifestyle changes, allowing users to gain actionable insights for better health outcomes. 
          - (Source: [Philips Survey](https://www.philips.co.kr/a-w/about/news/archive/standard/about/news/press/2022/20220720-philips-announces-results-of-personal-health-management-survey-in-asia.html))

        ### 🔍 Project Overview
        - **Goal**: To support healthy living by providing users with BMI calculations and predictive results based on lifestyle improvements.
        - **Key Features**:
          1. **BMI Calculation**: Calculates BMI using user input data.
          2. **Lifestyle Improvement Prediction**: Predicts and visualizes BMI changes based on user-provided lifestyle information, such as exercise frequency and vegetable consumption.
          3. **Streamlit Interface**: A simple and intuitive web application for real-time results.

        ### 💻 Technical Stack and Implementation
        - **Programming Language**: Python
        - **Analysis and Modeling**:
          - EDA: Identified key factors influencing BMI through correlation analysis.
          - Modeling: Used Multilayer Perceptron (MLP) for BMI prediction.
        - **Web Implementation Tool**: Streamlit
        - **Data Analysis Tools**: Pandas, Numpy, Scikit-learn
        - **Visualization Tools**: Matplotlib

        ### ⛳️ Expected Benefits
        - **Personalized Health Management**: Allows users to view their BMI and predict outcomes of lifestyle changes, empowering them to set health goals.
        - **Data-Driven Approach**: Provides reliable results through correlation analysis and machine learning models.
        - **User-Friendly Platform**: Enhances accessibility and usability through an intuitive Streamlit-based UI.

        ### 👩‍🔧 Model and Algorithm
        - **Model Explanation and Selection**:
          - MLP (Multilayer Perceptron) processes data through multiple layers of neurons for complex relationship learning.
          - MLP was chosen for its suitability for learning non-linear relationships and its compatibility with HyperOpt for parameter optimization.
        - **Model Structure**:
          - Key Components:
            1. Input Layer: Receives independent variables (features).
            2. Hidden Layer: Learns patterns and complex relationships through non-linear transformations.
            3. Output Layer: Predicts target variables.
          - Activation Function: ReLU for faster learning and gradient vanishing prevention.
          - Loss Function: Mean Squared Error (MSE) minimizes prediction errors.
          - Optimizer: Adam Optimizer for stable and efficient learning.
          - Configuration: hidden_layer_sizes=(200, 150, 100, 50), learning_rate_init=0.0098
        - **Hyperparameter Optimization**:
          - Tuned parameters using HyperOpt:
            - Search Space: Hidden layer structure, learning rate, regularization parameter (alpha), and max iterations.
            - Best Parameters: hidden_layer_sizes=(200, 150, 100, 50), alpha=0.000464, learning_rate_init=0.0098, max_iter=900
        - **Training and Validation**:
          - Split the dataset into training and testing sets (80:20 ratio).
          - Used 5-fold cross-validation to evaluate model stability.

        ### 🕹️ Optimization and Evaluation Metrics
        - **Metrics**:
          1. Mean Squared Error (MSE): Average of squared differences between actual and predicted values.
          2. Mean Absolute Error (MAE): Average of absolute differences between actual and predicted values.
          3. Mean Absolute Percentage Error (MAPE): Percentage difference between actual and predicted values.
          4. R² Score: Proportion of variance explained by the model.
          5. Adjusted R²: R² adjusted for the number of variables.
        - **Results**:
          - Train MSE: 0.01624
          - Test MSE: 0.01952
          - Train MAE: 0.1021
          - Test MAE: 0.1096
          - MAPE: 0.38%
          - R² Score: 0.9997
          - Adjusted R² Score: 0.9997
        - **Learning Curve**:
          - Demonstrates that training and validation errors are close, indicating low risk of overfitting.

        ### 🔮 Future Expansions
          1. Data Expansion: Acquire more data for underrepresented age groups (e.g., individuals over 50).
          2. Algorithm Experiments: Compare with other non-linear models (e.g., XGBoost, LightGBM).
          3. Streamlit Features: Add a gym locator based on BMI predictions.
    """)