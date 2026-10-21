import pandas as pd

# 1️⃣ User Profiles
users = pd.DataFrame([
    [1, "Ammu K", "female", 31, "Bangalore", "Karnataka", "ammuk@example.com"],
    [2, "Riya S", "female", 29, "Mumbai", "Maharashtra", "riyas@example.com"],
    [3, "Sneha R", "female", 34, "Hyderabad", "Telangana", "snehar@example.com"],
    [4, "Anita M", "female", 28, "Chennai", "Tamil Nadu", "anitam@example.com"],
    [5, "Divya T", "female", 32, "Delhi", "Delhi", "divyat@example.com"],
    [6, "Neha J", "female", 30, "Pune", "Maharashtra", "nehaj@example.com"],
    [7, "Kavya P", "female", 33, "Kochi", "Kerala", "kavyap@example.com"]
], columns=["user_id", "name", "gender", "age", "city", "state", "email"])

# 2️⃣ Clinics Dataset
clinics = pd.DataFrame([
    ["Fertility Plus", "Mumbai", "Maharashtra", "Dr. Anjali Mehta", 4.6, "9876543210"],
    ["Bloom IVF", "Bangalore", "Karnataka", "Dr. Vinutha Rao", 4.8, "9880011223"],
    ["Nova Fertility", "Hyderabad", "Telangana", "Dr. Rajesh Reddy", 4.5, "9778899001"],
    ["Cloudnine IVF", "Chennai", "Tamil Nadu", "Dr. Preeti Nair", 4.7, "9765432109"],
    ["Indira IVF", "Delhi", "Delhi", "Dr. Aarti Sharma", 4.9, "9556677889"],
    ["Oasis Fertility", "Pune", "Maharashtra", "Dr. Kavita Joshi", 4.4, "9900112233"],
    ["Morpheus IVF", "Kochi", "Kerala", "Dr. Sneha Pillai", 4.3, "9898989898"]
], columns=["clinic_name", "city", "state", "doctor_name", "rating", "contact"])

# 3️⃣ Wellness Logs
wellness = pd.DataFrame([
    [1, "2025-11-01", 4, "Happy", "Low", 7, "Good", "Mild fatigue", 3, 20, 15],
    [2, "2025-11-01", 3, "Neutral", "Medium", 6, "Average", "No symptoms", 4, 10, 5],
    [3, "2025-11-01", 2, "Sad", "High", 5, "Poor", "Cramps", 2, 5, 10],
    [4, "2025-11-02", 5, "Joyful", "Low", 8, "Excellent", "None", 5, 30, 20],
    [5, "2025-11-02", 3, "Tired", "Medium", 6, "Okay", "Headache", 3, 15, 10],
    [6, "2025-11-03", 4, "Calm", "Low", 7.5, "Good", "No issues", 4, 25, 15],
    [7, "2025-11-03", 2, "Anxious", "High", 5, "Poor", "Nausea", 2, 10, 10]
], columns=["user_id", "date", "mood_rating", "mood_desc", "stress", "sleep_hours", "sleep_quality",
            "symptoms", "energy_level", "exercise_min", "meditation_min"])

# 4️⃣ Nutrition & Hydration
nutrition = pd.DataFrame([
    [1, "2025-11-01", 8, "Good", "Yes", "Oats + Fruits", "Grilled Chicken Salad", "Soup", "Dry fruits"],
    [2, "2025-11-01", 6, "Average", "No", "Upma", "Rice + Dal", "Chapati", "Banana"],
    [3, "2025-11-01", 5, "Poor", "Yes", "Bread", "Pizza", "Fried Rice", "Chips"],
    [4, "2025-11-02", 9, "Excellent", "Yes", "Smoothie", "Quinoa Bowl", "Soup", "Apple"],
    [5, "2025-11-02", 7, "Good", "Yes", "Idli", "Curd Rice", "Roti + Curry", "Nuts"],
    [6, "2025-11-03", 8, "Good", "No", "Cornflakes", "Veg Rice", "Soup", "Orange"],
    [7, "2025-11-03", 6, "Average", "Yes", "Paratha", "Biryani", "Dal", "Samosa"]
], columns=["user_id", "date", "water_intake", "nutrition_score", "supplements_taken",
            "breakfast", "lunch", "dinner", "snacks"])

# 5️⃣ Prediction Results (AI Embryo / Treatment Outcome)
predictions = pd.DataFrame([
    [1, "2025-10-30", "High", 87],
    [2, "2025-10-31", "Medium", 73],
    [3, "2025-11-01", "Low", 59],
    [4, "2025-11-02", "High", 91],
    [5, "2025-11-03", "Medium", 76],
    [6, "2025-11-04", "Low", 60],
    [7, "2025-11-05", "High", 89]
], columns=["user_id", "prediction_date", "success_category", "confidence_score"])

# 6️⃣ Uploaded Documents
documents = pd.DataFrame([
    [1, "lab_report.pdf", "2025-10-25", "Report"],
    [2, "scan_result.png", "2025-10-26", "Ultrasound"],
    [3, "blood_test.pdf", "2025-10-27", "Test"],
    [4, "embryo_image.jpg", "2025-10-28", "Embryo"],
    [5, "prescription.pdf", "2025-10-29", "Prescription"],
    [6, "report_summary.pdf", "2025-10-30", "Report"],
    [7, "analysis.txt", "2025-10-31", "Notes"]
], columns=["user_id", "file_name", "upload_date", "file_type"])

# 7️⃣ AI Chat Logs
chat_logs = pd.DataFrame([
    [1, "How to improve IVF success?", "Maintain balanced diet", "2025-11-01"],
    [2, "Feeling anxious after embryo transfer", "Try meditation and rest", "2025-11-01"],
    [3, "What foods help fertility?", "Leafy greens, nuts, omega-3", "2025-11-02"],
    [4, "Can stress affect IVF?", "Yes, practice relaxation techniques", "2025-11-02"],
    [5, "How to track ovulation?", "Use BBT and ovulation kits", "2025-11-03"],
    [6, "Sleep impact on fertility?", "Yes, 7-8 hrs is ideal", "2025-11-03"],
    [7, "Can I exercise daily?", "Light exercise is beneficial", "2025-11-04"]
], columns=["user_id", "user_message", "ai_response", "date"])

# ✅ Save all 7 datasets to Excel
with pd.ExcelWriter("IVF_Journey_Tracker_Datasets.xlsx") as writer:
    users.to_excel(writer, sheet_name="Users", index=False)
    clinics.to_excel(writer, sheet_name="Clinics", index=False)
    wellness.to_excel(writer, sheet_name="Wellness", index=False)
    nutrition.to_excel(writer, sheet_name="Nutrition", index=False)
    predictions.to_excel(writer, sheet_name="Predictions", index=False)
    documents.to_excel(writer, sheet_name="Documents", index=False)
    chat_logs.to_excel(writer, sheet_name="ChatLogs", index=False)

print("✅ IVF_Journey_Tracker_Datasets.xlsx created successfully!")
