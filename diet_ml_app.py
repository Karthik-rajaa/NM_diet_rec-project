import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained ML model
model = joblib.load("calorie_predictor_model.joblib")

# App title
st.title("🍽️ AI-Powered Diet Recommendation System")

# Input form
st.header("Enter Your Details")
age = st.slider("Age", 10, 80, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.slider("Height (cm)", 100, 220, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)
activity = st.selectbox("Activity Level", ["Low", "Medium", "High"])
goal = st.selectbox("Fitness Goal", ["Fat Loss", "Maintain", "Muscle Gain"])

if st.button("Get My Diet Plan"):
    # Encode categorical inputs
    gender_map = {"Male": 1, "Female": 0}
    activity_map = {"Low": 0, "Medium": 1, "High": 2}
    goal_map = {"Fat Loss": 0, "Maintain": 1, "Muscle Gain": 2}

    input_data = np.array([[
        age,
        gender_map[gender],
        height,
        weight,
        activity_map[activity],
        goal_map[goal]
    ]])

    # Prediction
    predicted_calories = int(model.predict(input_data)[0])
    st.success(f"🔥 Your recommended daily calorie intake: **{predicted_calories} kcal**")

    # 1️⃣ Food Suggestions
    st.header("🍱 Suggested Meals")

    food_db = {
        "Fat Loss": [("Grilled Chicken Salad", 250), ("Boiled Eggs", 150), ("Green Smoothie", 180)],
        "Maintain": [("Brown Rice + Chicken", 500), ("Chapati + Veggies", 400), ("Fruit Bowl", 200)],
        "Muscle Gain": [("Peanut Butter Toast", 400), ("Banana Shake", 300), ("Oats + Nuts", 450)]
    }

    selected_foods = food_db[goal]
    for food, kcal in selected_foods:
        st.write(f"🍴 {food} – {kcal} kcal")

    
    # 2️⃣ BMI & Visualization
    st.header("📊 Your BMI Report")

    bmi = round(weight / ((height / 100) ** 2), 2)
    st.write(f"Your BMI: **{bmi}**")

    if bmi < 18.5:
        category = "Underweight"
        color = "blue"
    elif 18.5 <= bmi < 25:
        category = "Normal"
        color = "green"
    elif 25 <= bmi < 30:
        category = "Overweight"
        color = "orange"
    else:
        category = "Obese"
        color = "red"

    st.markdown(f"Category: **{category}**")

    # BMI chart
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(["BMI"], [bmi], color=color, height=0.3)
    ax.axvline(18.5, color="gray", linestyle="--", label="18.5")
    ax.axvline(25, color="gray", linestyle="--", label="25")
    ax.axvline(30, color="gray", linestyle="--", label="30")
    ax.set_xlim(10, 40)
    ax.set_xlabel("BMI Value")
    ax.set_title("BMI Range")
    st.pyplot(fig)
