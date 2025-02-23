import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import re
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest

# Set page titles and navigation
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Navigate", 
    ["How to Use the App", " Young Adults’ Health Challenges (10s-30s)", "Obesity Prediction", "Sleep Quality Prediction" ,"About Project"]
)

# Make Prediction Submenu
sub_page = None
if page == "Obesity Prediction":
    with st.sidebar.expander("Choose a Feature"):
        sub_page = st.radio("Select an option:", 
                            ["Make an Obesity Prediction", "Fitness Centers Map", "Health & Wellness Courses" ,"Learn More about Obesity"], index=0)

sleep_sub_page = None
if page == "Sleep Quality Prediction":
    with st.sidebar.expander("Select an Analysis Type"):
        sleep_sub_page = st.radio("Choose an option:", 
                            ["Make a Sleep Quality Prediction", "Learn More about Sleep Quality"], index=0)
        
# Page 1: Home & How to Use the App
if page == "How to Use the App":
    st.title("College Life essentials")
    st.subheader("Transform Your Lifestyle with Smart Insights")
    st.image("image/First-Day-Students-Bascom-2023-09-06BR-0062-1600x1066.jpg", use_container_width=True)

    st.write("""
        Welcome to **College Life essentials**, your AI-powered health companion! 🦡✨  
        This app helps you gain **personalized health insights** by analyzing your **eating habits, physical activity**, and **sleep quality**.  

        **🚀 Key Features:**  
        - 🏃 **Obesity Prediction**: Analyze your BMI and predict obesity risk based on lifestyle habits.  
        - 🌙 **Sleep Quality Prediction**: Understand how your lifestyle impacts your sleep quality.  
        - 📍 **Find Fitness Centers**: Locate nearby fitness centers to support your health journey.  
        - 📚 **Health Insights**: Learn about common health issues and ways to improve well-being.  

        **🔍 How to Use College Life essentials?**  
        1️⃣ Select a feature from the left menu.  
        2️⃣ Enter your details & lifestyle habits.  
        3️⃣ Get **personalized predictions & insights** for a healthier lifestyle!  
    """)
    
# Page 2:  Young Adults’ Health Challenges (10s-30s)
elif page == " Young Adults’ Health Challenges (10s-30s)":
    st.title("Common Health Challenges for 10s-30s")

    # Section 1: Raising issues (college student health issues)
    st.markdown("""
        <div style="border: 2px solid #FF2400; border-radius: 10px; background-color: #f9f9f9; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #FF2400; text-align: center;">Health Issues Among College Students</h4>
            <p style="font-size: 16px;">
                Many young adults in their 10s to 30s face increasing health challenges due to sedentary lifestyles, poor dietary habits, and high stress levels.
                Studies show that over 60% of young adults report irregular sleep patterns and 35% experience obesity-related risks before their 30s.
                These issues contribute to long-term problems like metabolic disorders, chronic stress, and sleep deprivation.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Section 2: Major Health Concerns
    with st.expander("📊 **Key Health Issues in 10s-30s**", expanded=False):
        st.write("""
        ### 🔎 **Top Health Challenges & Their Impact**
        
        1️⃣ **Unhealthy Eating Habits & High Fast Food Consumption** 🍔  
           - Skipping meals or relying on processed food leads to **weight gain, metabolic disorders, and chronic fatigue**.
        
        2️⃣ **Lack of Physical Activity & Sedentary Lifestyle** 🏋️  
           - Over 70% of young adults **don’t meet WHO's recommended exercise levels**, increasing obesity risks.
        
        3️⃣ **Sleep Deprivation & Poor Sleep Quality** 😴  
           - Irregular sleep patterns **affect memory, focus, and emotional health**, leading to long-term issues.
        
        4️⃣ **Mental Health Challenges** 🧠  
           - Academic and work-related stress contributes to **anxiety, depression, and lower productivity**.
        
        📌 **Did you know?**  
        Studies show that even **a small increase in daily steps and regular sleep schedules can significantly improve mental and physical health.**
        """)

    # Section 3: How Our App Helps
    with st.expander("🚀 **How This App Helps You**", expanded=False):
        st.write("""
        Our app is designed to **empower young adults** with **AI-driven insights** to **improve their health habits** effectively.
        
        ### 🔥 **Why Choose Our App?**
        ✅ **Personalized AI Health Insights** 🏥  
        - Get a **customized BMI analysis** and lifestyle-based recommendations.  
        - Predict how **small habit changes** (e.g., more sleep, fewer processed foods) affect your health.

        ✅ **Sleep Quality & Habit Tracking** 💤  
        - Understand **which lifestyle factors** impact your sleep and get personalized tips.  
        - Use our **habit tracker** to measure improvements over time.

        ✅ **Interactive Data Visualization** 📊  
        - View **real-time progress reports** on sleep, physical activity, and health risk factors.  
        - Compare **before vs. after** health predictions based on your daily habits.

        ✅ **Community & Expert Insights** 🌍  
        - Access expert-backed **health resources** and connect with others on similar health journeys.  
        - Share your progress and get **motivational support**.

        👉 **Start your health journey today! Small changes lead to long-term success. 🚀**
        """)
    st.image("image/LE8_circle.png", caption="Life's Essential 8 for health", use_container_width=True)

# Page 3: Obesity Prediction
elif page == "Obesity Prediction":
    if sub_page == "Make an Obesity Prediction":
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
            df["Age_Group"] = df["Age"].apply(age_group)

            # update to 10s~30s
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
            weight = st.slider("Weight (kg):", min_value=30.0, max_value=150.0, value=65.0, key="weight_slider")
    
        col3, col4 = st.columns(2)
        with col3:
            gender = st.selectbox("Gender:", ["Male", "Female"], key="gender_select")
        with col4:
            age = st.slider("Age:", min_value=10, max_value=39, value=20, key="age_slider")
    
        # Calculate current BMI and print with color reflected
        current_bmi = weight / ((height / 100) ** 2)
        current_level, current_color = bmi_level(current_bmi, gender)
        st.markdown(f"<h3 style='color:{current_color};'>Your Current BMI: {current_bmi:.2f} ({current_level})</h3>", unsafe_allow_html=True)
    
        # BMI distribution graph for the input person’s age
        age_group_label = age_group(age)
        age_group_data = dataset[dataset["Age_Group"] == age_group_label]["Calculated_BMI"]
        print("Available Age Groups:", dataset["Age_Group"].unique())
        print("20s Count:", len(dataset[dataset["Age_Group"] == "20s"]))
    
        if not age_group_data.empty:
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            colors = sns.color_palette("coolwarm", as_cmap=True)
            n, bins, patches = plt.hist(age_group_data, bins=15, color="lightblue", edgecolor="black", alpha=0.7)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = (bin_centers - min(bin_centers)) / (max(bin_centers) - min(bin_centers))
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", colors(c))
            bmi_categories = {
                "Underweight": 18.5,
                "Normal": 24.9,
                "Overweight": 29.9,
                "Obesity": max(age_group_data)  # 최대값으로 설정
            }
        
            for category, value in bmi_categories.items():
                plt.axvline(value, color="gray", linestyle="dashed", linewidth=1)
                plt.text(value, max(n) * 0.9, category, color="black", fontsize=12, ha="center", fontweight="bold")
            plt.axvline(current_bmi, color=current_color, linestyle="solid", linewidth=2, label=f"Your BMI: {current_bmi:.2f}")
            plt.title(f"BMI Distribution for Your Age Group: {age_group_label}", fontsize=14, fontweight="bold")
            plt.xlabel("BMI", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend()
            st.pyplot(plt)
        else:
            st.write(f"No BMI data available for the age group: {age_group_label}")
        
    
        # 2: Rest of the input variables
        col5, col6 = st.columns(2)
        with col5:
            physical_activity = st.slider(
                "Physical Activity Frequency (days/week):", 
                min_value=0, max_value=10, value=3, key="physical_activity_slider",
                help="How often do you engage in physical activity per week? (0: None, 1-2: Light, 3-4: Moderate, 5+: Active)"
            )
        with col6:
            vegetables = st.slider(
                "Vegetable Consumption Frequency:", 
                min_value=0, max_value=5, value=2, key="vegetables_slider",
                help="How frequently do you eat vegetables? (0: Never, 1: Rarely, 2: Sometimes, 3+: Always)"
            )
        
        col7, col8 = st.columns(2)
        with col7:
            family_history = st.selectbox(
                "Family History of Overweight:", ["Yes", "No"], key="family_history_select",
                help="Do you have a family history of overweight or obesity? (Yes/No)"
            )
        with col8:
            high_calorie_food = st.selectbox(
                "Do you consume high caloric food frequently?", ["Yes", "No"], key="calorie_food_select",
                help="Do you prefer high-calorie foods? (Yes: Frequently eat fast food or processed foods)"
            )
        
        col9, col10 = st.columns(2)
        with col9:
            technology_usage = st.slider(
                "Technology Usage (hours/day):", 
                min_value=0.0, max_value=10.0, value=3.0, key="technology_usage_slider",
                help="How many hours per day do you spend on screens? (0-2: Low, 3-5: Moderate, 5+: High)"
            )
        with col10:
            food_between_meals = st.selectbox(
                "How often do you consume food between meals?", ["Never", "Sometimes", "Frequently"], key="food_between_meals_select",
                help="How often do you snack between meals? (Never, Sometimes, Frequently)"
            )
        
        col11, col12 = st.columns(2)
        with col11:
            smoking = st.selectbox(
                "Do you smoke?", ["Yes", "No"], key="smoking_select",
                help="Do you currently smoke? (Yes/No)"
            )
        with col12:
            alcohol = st.selectbox(
                "Do you consume alcohol?", ["No", "Yes"], key="alcohol_select",
                help="Do you consume alcohol regularly? (No/Yes)"
            )
        
        col13, col14 = st.columns(2)
        with col13:
            main_meals = st.slider(
                "Main Meals Frequency (meals/day):", 
                min_value=0, max_value=10, value=3, key="main_meals_slider",
                help="How many main meals do you eat per day? (1-2: Low, 3: Normal, 4+: High)"
            )
        with col14:
            calorie_tracking = st.selectbox(
                "Do you track your daily calorie intake?", ["Yes", "No"], key="calorie_tracking_select",
                help="Do you monitor your calorie intake? (Yes/No)"
            )
        
        col15, col16 = st.columns(2)
        with col15:
            water_consumption = st.slider(
                "Water Consumption (liters/day):", 
                min_value=0.0, max_value=20.0, value=2.0, key="water_consumption_slider",
                help="How much water do you drink daily? (<1L: Low, 1-2L: Normal, 2+L: High)"
            )
        with col16:
            transportation = st.selectbox(
                "Transportation Method:", ["Walking", "Public_Transportation", "Car", "Bike"], key="transportation_select",
                help="What is your main mode of transportation? (Walking, Public Transport, Car, Bike)"
            )

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


    # SubPage2: Fitness Centers Map
    elif sub_page == "Fitness Centers Map":
        st.title("🏋️ UW Madison Fitness Center Map")
        st.write("Check the locations of fitness centers on the map below!")
    
        file_path = "fitness_center.csv"  
        df = pd.read_csv(file_path, encoding="utf-8")
    
        # 🛠 Clean column names to remove hidden characters & spaces
        df.columns = df.columns.str.encode("utf-8").str.decode("utf-8")
        df.columns = df.columns.str.replace(r"[^\x20-\x7E]", "", regex=True).str.strip()
    
        # 🗺️ Create a map centered around UW Madison
        m = folium.Map(location=[43.0731, -89.4012], zoom_start=14)
    
        # 📍 Add fitness centers to the map
        for _, row in df.iterrows():
            if pd.notnull(row["Latitude"]) and pd.notnull(row["Longitude"]):
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"<b>{row['Name']}</b><br>{row['Address']}<br><a href='{row['Site']}' target='_blank'>Website</a>",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)
    
        # 🗺️ Display the map
        folium_static(m)
    
        # 🔽 Divider line
        st.markdown("---")
    
        # 📋 Display fitness centers as cards (3 per row)
        st.write("## 🏢 Fitness Centers")
    
        # Group facilities in rows of 3
        cols_per_row = 3
        rows = [df.iloc[i:i + cols_per_row] for i in range(0, len(df), cols_per_row)]
    
        for row in rows:
            cols = st.columns(len(row)) 
            for col, (_, facility) in zip(cols, row.iterrows()):
                with col:
                    st.markdown(f"""
                    <div style="border-radius: 10px; padding: 15px; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); padding: 10px;">
                        <h4>{facility["Name"]}</h4>
                        <p>📍 <b>Address:</b> {facility["Address"]}</p>
                        <p>📝 <b>Description:</b> {str(facility["Description"])[:100]}...</p>
                        <p>🕒 <b>Hours:</b><br>
                           <b>Mon-Thu:</b> {facility["Monday - Thursday"]}<br>
                           <b>Fri:</b> {facility["Friday"]}<br>
                           <b>Sat:</b> {facility["Saturday"]}<br>
                           <b>Sun:</b> {facility["Sunday"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
                    # 🔗 Corrected Website Link
                    if pd.notnull(facility["Site"]):  
                        st.markdown(f"[🌐 Visit Website]({facility['Site']})", unsafe_allow_html=True)


    # SubPage2: Health-related classes (DANCE, FOOD SCI, KINES divided into tabs)
    elif sub_page == "Health & Wellness Courses":
        st.title("📚 Health & Wellness Courses")
        st.write("Explore courses related to fitness, nutrition, and dance.")

        health_course = "filtered_courses.csv"  
        df_courses = pd.read_csv(health_course, encoding="utf-8")
        
        # create tap (DANCE, KINES, FOOD SCI)
        tab_dance, tab_kines, tab_food_sci = st.tabs(["💃 Dance", "🏋️ Kinesiology", "🥗 Food Science"])
    
        df_dance = df_courses[df_courses["Course Code"].str.startswith("DANCE")]
        df_kines = df_courses[df_courses["Course Code"].str.startswith("KINES")]
        df_food_sci = df_courses[df_courses["Course Code"].str.startswith("FOOD SCI")]
    
        items_per_page = 6
    
        def display_courses(df, tab_name):
            total_pages = -(-len(df) // items_per_page)  
            current_page = st.number_input(f"Page ({tab_name})", min_value=1, max_value=max(total_pages, 1), value=1, step=1) - 1
    
            start_idx = current_page * items_per_page
            end_idx = start_idx + items_per_page
            paginated_df = df.iloc[start_idx:end_idx]
    
            # course list
            for _, course in paginated_df.iterrows():
                with st.expander(f"📌 **{course['Course Name']}** ({course['Course Code']})"):
                    st.markdown(f"""
                    **Course Number:** {course["Course Number"]}  
                    **Description:**  
                    {course["Description"]}
                    """)
    
                    # course link
                    if pd.notnull(course["Link"]):
                        st.markdown(f"[📖 Course Guide]({course['Link']})", unsafe_allow_html=True)
    
            # button for page
            st.write(f"📄 Page {current_page + 1} of {total_pages}")
    
        # DANCE
        with tab_dance:
            display_courses(df_dance, "Dance")
    
        # KINES
        with tab_kines:
            display_courses(df_kines, "Kinesiology")
    
        # FOOD SCI
        with tab_food_sci:
            display_courses(df_food_sci, "Food Science")

    # SubPage4: Learn More about Obesity
    elif sub_page == "Learn More about Obesity":
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



# Page 4: Sleep Quality Prediction
elif page == "Sleep Quality Prediction":
    if sleep_sub_page == "Make a Sleep Quality Prediction":
        st.title("🛋 Predict Your Sleep Quality")
        st.write("Adjust your lifestyle inputs to see how they affect your sleep quality.")
    
        @st.cache_data
        def load_sleep_data():
            df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
        
            # Systolic/Diastolic Blood Pressure separation
            df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
            df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
            df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')
            df.drop(columns=['Blood Pressure'], inplace=True)
        
            # Scale numeric columns
            numeric_cols = ['Sleep Duration', 'Physical Activity Level',
                            'Stress Level', 'Systolic_BP', 'Diastolic_BP', 'Heart Rate', 'Daily Steps']
            
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
            # Standard Scaling
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            return df, scaler
        
        df, sleep_scaler = load_sleep_data()
    
        # Train Optimized Random Forest model
        @st.cache_resource
        def train_sleep_model():
            X = df[['Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Systolic_BP', 'Diastolic_BP', 'Heart Rate', 'Daily Steps']]
            y = df["Quality of Sleep"]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=10, max_features='sqrt', 
                min_samples_leaf=1, min_samples_split=2, random_state=42
            )
            rf.fit(X_train, y_train)
        
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
        
            return rf, mse
    
        sleep_model, sleep_mse = train_sleep_model()
        
        # Get user input
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.slider("Sleep Duration (hours):", min_value=3.0, max_value=12.0, value=7.0)
        with col2:
            physical_activity = st.slider("Physical Activity Level (days/week):", min_value=0, max_value=7, value=3)
    
        col3, col4 = st.columns(2)
        with col3:
            stress_level = st.slider("Stress Level (1-10):", min_value=1, max_value=10, value=5)
        with col4:
            heart_rate = st.slider("Resting Heart Rate (bpm):", min_value=40, max_value=120, value=70)
    
        col5, col6 = st.columns(2)
        with col5:
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)- Optional:", min_value=90, max_value=180, value=120)
        with col6:
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg) - Optional:", min_value=50, max_value=110, value=80)
    
        daily_steps = st.slider("Daily Steps:", min_value=1000, max_value=20000, value=8000, step=500)
    
        input_data = pd.DataFrame({
            "Sleep Duration": [sleep_duration],
            "Physical Activity Level": [physical_activity],
            "Stress Level": [stress_level],
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
            "Heart Rate": [heart_rate],
            "Daily Steps": [daily_steps]
        })
    
        input_data_scaled = sleep_scaler.transform(input_data)
    
        # Predict sleep quality using Random Forest
        predicted_quality = sleep_model.predict(input_data_scaled)[0]
        predicted_quality = max(0, min(10, predicted_quality))
        
        st.markdown(f"<h3 style='color:#FF5733;'>Predicted Sleep Quality: {predicted_quality:.2f} / 10</h3>", unsafe_allow_html=True)

        # Provide feedback based on forecast values
        if predicted_quality >= 8.0:
            st.success("🌙 Your predicted sleep quality is excellent! Keep up the good habits!")
            st.write("💡 You're doing great! Maintain your current sleep routine and lifestyle to continue enjoying high-quality rest.")
        
        elif predicted_quality >= 7.0:
            st.info("💤 Your sleep quality is good, but there's room for improvement!")
        
        elif predicted_quality >= 6.1:
            st.warning("😴 Your sleep quality is moderate. Try reducing stress and increasing activity!")
        
        else:
            st.error("⚠️ Your predicted sleep quality is low. Consider adjusting your sleep habits and lifestyle.")
        
        # Show personalized sleep improvement tips only if sleep quality is below 8.0
        if predicted_quality < 8.0:
            st.subheader("🛌 Personalized Sleep Improvement Tips")
            st.write("💡 Based on your inputs, here are some evidence-based strategies to improve your sleep quality!")
        
            # 1️⃣ Increase Sleep Duration (if < 6 hours)
            if sleep_duration < 6:
                st.write("### 🕰️ Increase Sleep Duration")
                st.write("""
                - Aim for **7-9 hours** of sleep per night, as recommended by the National Sleep Foundation.
                - Establish a **consistent bedtime routine** (e.g., reading, meditation) to improve sleep onset.
                - Avoid **blue light exposure** from screens at least **1 hour before bed** to support melatonin production.
                - Consider **short naps (20-30 minutes)** if you experience daytime fatigue, but avoid long naps that disrupt nighttime sleep.
                """)
        
            # 2️⃣ Reduce Stress (if Stress Level > 7)
            if stress_level > 7:
                st.write("### 😆 Reduce Stress")
                st.write("""
                - Engage in **mindfulness practices** such as meditation, deep breathing, or progressive muscle relaxation.
                - Try **journaling or expressive writing** to process thoughts and reduce mental clutter before bedtime.
                - Physical activity can be a great stress reliever—**aim for 30 minutes of moderate exercise** during the day.
                - Explore **Aromatherapy**: Essential oils like **lavender** and **chamomile** are known to promote relaxation.
                """)
        
            # 3️⃣ Improve Physical Activity (if < 3 days/week)
            if physical_activity < 3:
                st.write("### 🏋🏻 Improve Physical Activity")
                st.write("""
                - Incorporate at least **150 minutes of moderate exercise per week** (e.g., walking, yoga, swimming).
                - Morning or afternoon workouts are ideal—**avoid intense exercise within 2 hours of bedtime**, as it can increase alertness.
                - If you're short on time, **even 10-minute stretching or yoga before bed** can improve sleep quality.
                """)
        
            # 4️⃣ Optimize Blood Pressure (if Systolic BP > 130 or Diastolic BP > 85)
            if systolic_bp > 130 or diastolic_bp > 85:
                st.write("### 🩸 Optimize Blood Pressure")
                st.write("""
                - Follow a **DASH (Dietary Approaches to Stop Hypertension) diet**, which includes **more fruits, vegetables, whole grains, and lean proteins**.
                - Reduce **sodium intake**, which can contribute to high blood pressure and affect sleep.
                - Engage in **relaxing activities before bed**, such as listening to calming music or taking a warm bath.
                """)
        
            # 5️⃣ Lower Resting Heart Rate (if > 90 bpm)
            if heart_rate > 90:
                st.write("### 🩺 Lower Resting Heart Rate")
                st.write("""
                - **Hydration is key**—dehydration can raise heart rate, so ensure you **drink enough water** throughout the day.
                - Try **slow, deep breathing techniques** (e.g., inhale for 4 seconds, hold for 4 seconds, exhale for 8 seconds).
                - Engage in **low-impact activities like Tai Chi or light yoga**, which promote relaxation.
                """)
        
            # 6️⃣ Increase Daily Steps (if < 5000 steps/day)
            if daily_steps < 5000:
                st.write("### 👣 Increase Daily Steps")
                st.write("""
                - **Take breaks to walk** every hour, even if it's just for **5 minutes**.
                - Consider a **standing desk** or **walking meetings** to incorporate movement into your daily routine.
                - Walking after meals can **help regulate blood sugar** and improve overall sleep quality.
                """)
        
            # 7️⃣ Create a Sleep-Friendly Environment
            st.write("### 🛌 Create a Sleep-Friendly Environment")
            st.write("""
            - **Ideal bedroom temperature**: Keep your room between **60-67°F (15-19°C)** for optimal sleep.
            - **Upgrade your mattress and pillow**: A comfortable sleeping surface **reduces body aches and promotes deep sleep**.
            - **Minimize noise**: Use **earplugs or white noise machines** if external sounds disrupt your sleep.
            - **Use blackout curtains** to **reduce light pollution** and create a darker sleep environment.
            """)
        
            st.write("💡 *Even small changes in your routine can lead to significant improvements in sleep quality!* 😴✨")

     
    elif sleep_sub_page == "Learn More about Sleep Quality":
        st.title("Sleep Quality Insights")
        st.write("Explore the Sleep Quality Insights.")

        # Academic & Focus Improvement Tips
        st.subheader("📚 Academic & Focus Enhancement Tips")
        st.write("💡 Proper sleep is crucial for cognitive function and academic performance. Consider the following:")
    
        with st.expander("🕰️ Stick to a Consistent Sleep Schedule for Better Memory"):
            st.write("""
            - Regular sleep patterns **enhance memory consolidation and cognitive performance**.  
            - **Irregular sleep disrupts learning**, making it harder to retain new information.  
            - The **most critical sleep period** for memory retention occurs right after learning.  
            - **Skipping sleep or pulling all-nighters** weakens both factual and procedural memory.  
            - **Study Link:** [Click here](https://sleep.hms.harvard.edu/education-training/public-education/sleep-and-health-education-program/sleep-health-education-88?utm_source=chatgpt.com)
            """)
    
    
        with st.expander("🤧 Get Enough Sleep to Strengthen Immunity"):
            st.write("""
            - **Short sleep duration increases the risk of catching a cold.**  
            - People who sleep **less than 6 hours per night** are **over 4 times more likely** to develop a cold after viral exposure.  
            - This effect is **independent of age, BMI, stress levels, and health habits**.  
            - **Consistent, high-quality sleep supports a stronger immune response** and lowers susceptibility to infections.  
            - **Study Link:** [Click here](https://jcsm.aasm.org/doi/10.5664/jcsm.5020)
            """)
    
    
        # Research Papers & References
        st.subheader("📖 Research Papers & References")
    
        with st.expander("📖 The Link Between Sleep Quality and Learning Engagement"):
           st.write("""
            - **Better sleep quality significantly improves learning engagement** in students.  
            - Poor sleep is linked to **lower attention, memory retention, and cognitive function**, leading to reduced academic performance.  
            - **Mental health moderates this effect**—students with good mental health can better cope with sleep deprivation.  
            - Schools and families should promote **healthy sleep habits** to enhance students’ academic success.  
            - **Read Here:** [Click here](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1476840/full)
            """)
    
    
        with st.expander("📝 The Link Between Sleep and Academic Performance"):
            st.write("""
            - **High-achieving students tend to sleep less and go to bed later** than lower-performing students.
            - **Later bedtimes are linked to increased online activity at night**, often due to educational or entertainment activities.
            - **Sleep deprivation negatively impacts cognitive function, concentration, and stress levels**, affecting overall well-being.
            - The study suggests **academic success may come at the cost of individual health**.
            - **Read Here:** [Click here](https://arxiv.org/abs/2005.07806)
            """)
    
    
        with st.expander("🛁 Warm Baths and Sleep Improvement"):
            st.write("""
            - **Taking a warm shower or bath (40–42.5°C) 1–2 hours before bed** can improve sleep quality.
            - **Shortens sleep onset latency (SOL)**, helping you fall asleep faster.
            - Works by enhancing **core body temperature decline**, promoting relaxation and sleep efficiency.
            - Findings suggest that as little as **10 minutes of passive body heating (PBHWB)** can be effective.
            - **Read Here:** [Click here](https://doi.org/10.1016/j.smrv.2019.04.008)
            """)



  

# Page 5: About Project
elif page == "About Project":
    st.title("About Project")
    st.write("""
        
        ### 👭 Team Members
        - **Iahn Shim**
        - **Hari Kang**
        - **Yunji Lee**
        
        ### 🏃‍♂️ Background and Purpose
        - **Health Challenges in the Modern Age**:
          - Obesity and poor sleep quality are increasingly affecting young adults due to sedentary lifestyles, unhealthy eating habits, and high stress levels.
          - According to WHO, global obesity rates have risen significantly, with sleep deprivation contributing to cognitive decline and metabolic disorders.
        - **Need for Personalized Health Monitoring**:
          - Many individuals lack personalized insights into how their daily habits impact their health.
          - This project aims to bridge this gap by providing real-time predictions for BMI and sleep quality based on lifestyle inputs.
          
        ### 🔍 Project Overview
        - **Goal**: To help individuals make data-driven decisions for improving their health by predicting obesity levels and sleep quality based on lifestyle factors.
        - **Key Features**:
          1. **Obesity Prediction**: Calculates BMI and predicts weight trends based on user habits.
          2. **Sleep Quality Prediction**: Estimates sleep quality based on factors such as stress levels, physical activity, and heart rate.
          3. **Data-Driven Insights**: Provides personalized health recommendations.
          4. **Interactive Streamlit Interface**: Enables real-time predictions and visualizations.

        ### 💻 Technical Stack and Implementation
        - **Programming Language**: Python
        - **Data Science Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
        - **Machine Learning Models**:
          - **Obesity Prediction**: MLPRegressor for predicting BMI changes based on user inputs.
          - **Sleep Quality Prediction**: Optimized Random Forest Regressor for estimating sleep quality scores.  
        - **Web Implementation Tool**: Streamlit for user interaction and visualization.

        ### 🤖 Model and Algorithm Details
        #### 1️⃣ Obesity Prediction Model
        - **Model Used**: MLPRegressor (Multilayer Perceptron)
        - **Key Features**:
          - Input: Weight, height, activity levels, vegetable consumption, and technology usage.
          - Output: Predicted BMI level.
          - `Hidden Layers`: (200, 150, 100, 50) with ReLU activation.
          - Optimization: `Adam optimizer` with fine-tuned learning rate and regularization.
          
        #### 2️⃣ Sleep Quality Prediction Model
        - **Model Used**: Optimized Random Forest Regressor
        - **Key Features**:
          - Input Features: Sleep duration, stress levels, physical activity, heart rate, blood pressure, and daily steps.
          - Output: Predicted sleep quality score (continuous scale)
          - Optimized Hyperparameters:
            - `n_estimators`: 100
            - `max_depth`: 10
            - `max_features`: 'sqrt'
            - `min_samples_split`: 2
            - `min_samples_leaf`: 1
          
        ### 📊 Evaluation Metrics
        - **Obesity Prediction Model**:
          - MSE: 0.01952
          - MAE: 0.1096
          - R² Score: 0.9997
        - **Sleep Quality Prediction Model**:
          - MSE: 0.02043
          - MAE: 0.04901
          - R² Score: 0.9841
        
        ### 🔮 Future Expansions
        1. **Additional Features**: Incorporate meal tracking and hydration levels for more accurate predictions.
        2. **Improved Sleep Analysis**: Implement time-series modeling to analyze sleep patterns over time.
        3. **User Engagement**: Develop a mobile app version for real-time health tracking.
        4. **Integration with Wearables**: Use smartwatch data for continuous health monitoring.
    """)
