import os
import json
import random
import base64
import requests
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "fallback-key")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY != "fallback-key" else None

# Google Gemini API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
GEMINI_TTS_MODEL = "gemini-2.0-flash-lite"

def get_chatbot_response(message, user, patient_data=None):
    """Get AI chatbot response for IVF-related questions"""
    
    # Check if OpenAI client is available
    if client is None:
        # Return a helpful fallback response about IVF
        return get_fallback_ivf_response(message, user, patient_data)
    
    try:
        # Build context based on user data
        context = f"You are an AI assistant specialized in IVF (In Vitro Fertilization) support. "
        context += f"You're speaking with {user.first_name}, a {user.user_type}. "
        
        if patient_data:
            context += f"Patient details: Age {patient_data.age}, "
            if patient_data.diagnosis:
                context += f"Diagnosis: {patient_data.diagnosis}, "
            if patient_data.previous_ivf_cycles:
                context += f"Previous IVF cycles: {patient_data.previous_ivf_cycles}, "
        
        context += """
        Provide helpful, accurate, and empathetic responses about:
        - IVF procedures and timelines
        - Medication guidance and side effects
        - Emotional support and encouragement
        - Appointment preparation
        - Lifestyle recommendations
        - Nutritional advice for fertility
        
        Always be supportive and remind users to consult their healthcare provider for medical decisions.
        Keep responses concise but informative.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Return fallback response on error
        return get_fallback_ivf_response(message, user, patient_data)


def get_fallback_ivf_response(message, user, patient_data=None):
    """Provide helpful IVF-related responses when AI is not available"""
    message_lower = message.lower()
    
    # IVF Basics responses
    if any(keyword in message_lower for keyword in ['what is ivf', 'how does ivf work', 'ivf process']):
        return """IVF (In Vitro Fertilization) is an assisted reproductive technology where an egg and sperm are combined outside the body in a laboratory setting. The process typically involves:

1. **Ovarian Stimulation**: Fertility medications are used to stimulate the ovaries to produce multiple eggs
2. **Egg Retrieval**: A minor surgical procedure to collect eggs from the ovaries
3. **Fertilization**: Eggs are combined with sperm in a lab to create embryos
4. **Embryo Transfer**: One or more embryos are transferred into the uterus

The entire cycle takes about 2-3 weeks. Success rates vary based on factors like age, reproductive history, and clinic expertise.

Always consult with your fertility specialist for personalized guidance."""
    
    # Success rates responses
    elif any(keyword in message_lower for keyword in ['success rate', 'chance', 'probability', 'likelihood']):
        return """IVF success rates depend on several factors:

- **Age**: Women under 35 have the highest success rates (around 40-50%), declining with age
- **Previous pregnancies**: History of previous pregnancies can improve success rates
- **Ovarian reserve**: AMH and FSH levels affect response to treatment
- **Lifestyle factors**: Maintaining a healthy weight, avoiding smoking, and reducing stress can help
- **Clinic expertise**: Success rates vary between clinics

On average, the live birth rate per IVF cycle is approximately:
- Under 35: 40-50%
- 35-37: 35-40%
- 38-40: 25-30%
- Over 40: 10-15%

Your doctor can provide personalized estimates based on your specific situation."""
    
    # Cost responses
    elif any(keyword in message_lower for keyword in ['cost', 'price', 'expensive', 'afford']):
        return """IVF costs vary significantly depending on location and clinic. Here's a general breakdown:

**Average IVF Costs (per cycle):**
- United States: $12,000 - $15,000
- United Kingdom: £5,000 - £8,000
- India: ₹1,5,000 - ₹3,00,000

**Additional costs may include:**
- Fertility medications: $3,000 - $7,000
- ICSI (if needed): $1,500 - $2,500
- Preimplantation genetic testing: $3,000 - $6,000
- Embryo freezing: $500 - $1,000

Many clinics offer payment plans, and some insurance plans cover part of the costs. Research your options and discuss financing with your clinic."""
    
    # Medication responses
    elif any(keyword in message_lower for keyword in ['medication', 'medicine', 'drugs', 'injection', 'gonal-f', 'menopur', 'clomid']):
        return """Common IVF medications include:

**For Ovarian Stimulation:**
- **Gonadotropins** (Gonal-F, Follistim, Menopur): Stimulate egg production
- **Clomiphene Citrate** (Clomid): Oral medication to induce ovulation
- **Letrozole**: Often used as an alternative to Clomid

**Other Medications:**
- **Ganirelix/Cetrotide**: Prevent premature ovulation
- **hCG trigger** (Ovidrel, Pregnyl): Triggers egg maturation
- **Progesterone**: Supports the uterine lining after transfer

**Important Notes:**
- Medications are typically self-administered as injections
- Side effects may include bloating, mood swings, and injection site reactions
- Always follow your doctor's instructions precisely
- Report any severe symptoms to your clinic immediately"""
    
    # Side effects / risks responses
    elif any(keyword in message_lower for keyword in ['side effect', 'risk', 'danger', 'safe', 'complication']):
        return """IVF has several potential side effects and risks:

**Common Side Effects:**
- Bloating and abdominal discomfort
- Mood swings from hormonal medications
- Breast tenderness
- Injection site bruising
- Fatigue

**Medical Risks:**
- **OHSS** (Ovarian Hyperstimulation Syndrome): Rare but serious; symptoms include severe abdominal pain, vomiting, and difficulty breathing
- **Multiple pregnancies**: Twins or more can occur with embryo transfers
- **Ectopic pregnancy**: Small risk of embryo implanting outside the uterus
- **Miscarriage**: Risk is similar to natural conception (~15-25%)
- **Egg retrieval complications**: Rare; includes infection or bleeding

Discuss all risks with your fertility specialist before starting treatment."""
    
    # Emotional support responses
    elif any(keyword in message_lower for keyword in ['stress', 'anxiety', 'emotional', 'depression', 'feel', 'support', 'mental']):
        return """It's completely normal to experience emotional challenges during IVF. Here's some guidance:

**Common Emotional Responses:**
- Anxiety about success rates
- Grief over fertility struggles
- Stress from treatments and costs
- Strain on relationships
- Hope and disappointment cycles

**Coping Strategies:**
1. **Connect with others**: Join support groups (online or in-person)
2. **Communicate openly**: Talk with your partner about feelings
3. **Practice self-care**: Gentle exercise, meditation, hobbies
4. **Set boundaries**: It's okay to skip baby showers or limit social media
5. **Seek professional help**: Consider a therapist specializing in infertility
6. **Limit research**: Too much googling can increase anxiety

Remember: Your worth as a person isn't determined by your ability to conceive. Be kind to yourself throughout this journey."""
    
    # Diet / nutrition responses
    elif any(keyword in message_lower for keyword in ['diet', 'food', 'eat', 'nutrition', 'weight', 'exercise', 'yoga']):
        return """Lifestyle factors can influence IVF success. Here are evidence-based recommendations:

**Diet Recommendations:**
- Eat protein-rich foods (lean meats, fish, eggs, legumes)
- Include plenty of fruits and vegetables
- Choose whole grains over refined carbs
- Limit caffeine (200mg/day max - about 2 cups coffee)
- Avoid alcohol and smoking completely
- Stay hydrated (8-10 glasses water daily)

**Exercise Guidelines:**
- Moderate exercise is beneficial (walking, swimming, yoga)
- Avoid intense workouts during stimulation
- After transfer, stick to gentle activities

**Supplements to Consider (consult your doctor):**
- Folic acid (400-800mcg daily)
- Vitamin D
- CoQ10 (may improve egg quality)
- Omega-3 fatty acids

Always discuss any supplements or major diet changes with your fertility doctor."""
    
    # Timeline / process duration responses
    elif any(keyword in message_lower for keyword in ['how long', 'timeline', 'duration', 'time', 'when', 'cycle']):
        return """IVF timeline varies, but here's a general overview:

**Single IVF Cycle:**
- **Pre-treatment**: 2-4 weeks for testing and preparation
- **Stimulation**: 10-14 days of fertility injections
- **Egg retrieval**: 1 day (procedure takes 20-30 minutes)
- **Fertilization**: 3-5 days in lab
- **Embryo transfer**: 1 day (5-10 minutes)
- **Pregnancy test**: 10-14 days after transfer

**Total Time**: Approximately 4-6 weeks per cycle

**Important Notes:**
- Some couples need multiple cycles (average 2-3)
- Between cycles, you may need 1-2 months break
- Frozen embryo transfers can be done in subsequent months

Your clinic will provide a more specific timeline based on your protocol."""
    
    # Age-related responses
    elif any(keyword in message_lower for keyword in ['age', 'older', 'younger', 'late', 'early']):
        return """Age is one of the most important factors affecting IVF success:

**Female Age and IVF Success:**
- **Under 35**: Highest success rates (40-50% per cycle)
- **35-37**: Good success rates (35-40%)
- **38-40**: Decreased success (25-30%)
- **40+**: Lower success rates (10-15%)
- **45+**: Very low success with own eggs; donor eggs often recommended

**Why Age Matters:**
- Egg quantity and quality decline with age
- Genetic abnormalities become more common
- Uterine receptivity may decrease

**Options for Older Patients:**
- Donor eggs (significantly higher success rates)
- Embryo donation
- Surrogacy
- Multiple cycles for better odds

Discuss your specific situation with your fertility specialist."""
    
    # PCOS / Endometriosis / specific conditions
    elif 'pcos' in message_lower:
        return """PCOS (Polycystic Ovary Syndrome) is a common cause of infertility. Here's what you should know:

**How PCOS Affects Fertility:**
- Irregular or absent ovulation
- High androgens (male hormones) can interfere with egg development
- Insulin resistance is common

**IVF Considerations for PCOS:**
- Higher risk of Ovarian Hyperstimulation Syndrome (OHSS)
- May need lower medication doses
- Often good response to stimulation (many eggs)
- Success rates are generally good with proper protocol

**Tips for PCOS Patients:**
- Maintain healthy weight (even 5-10% loss helps)
- Consider inositols (myo-inositol) supplements
- Choose a clinic experienced with PCOS patients
- Discuss PGT-A (genetic testing) with your doctor

Your doctor may recommend specific protocols like minimal stimulation or antagonist protocols."""
    
    elif 'endometriosis' in message_lower:
        return """Endometriosis can significantly impact fertility. Here's what to know:

**How Endometriosis Affects Fertility:**
- Can damage ovarian tissue and reduce egg quality
- May cause uterine inflammation
- Can affect embryo implantation
- May block fallopian tubes

**IVF with Endometriosis:**
- Success rates are slightly lower but still significant
- Often recommended before severe progression
- May benefit from longer suppression before IVF
- Surgery to remove endometriomas may help

**Tips:**
- Choose a specialist experienced with endometriosis
- Consider immune supplementation (ask your doctor)
- Embryo transfer timing may need special consideration

Discuss your specific case with both your RE and endometriosis specialist."""
    
    # General greeting
    elif any(keyword in message_lower for keyword in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
        return f"""Hello {user.first_name}! 👋

I'm your IVF Assistant, here to help you navigate your fertility journey. I can provide information about:

- IVF procedures and processes
- Medications and protocols
- Success rates and factors
- Lifestyle and nutrition
- Emotional support and coping strategies
- Costs and planning

Please note: I'm an AI assistant and can provide general information, but always consult with your healthcare provider for personalized medical advice.

How can I help you today?"""
    
    # Thank you
    elif any(keyword in message_lower for keyword in ['thank', 'thanks']):
        return """You're welcome! 😊

I'm here to help throughout your IVF journey. Remember:
- You're not alone in this process
- It's okay to ask questions
- Take it one step at a time
- Be kind to yourself

Is there anything else you'd like to know?"""
    
    # Default response
    else:
        return f"""Thank you for your question! Here's some general information about IVF:

**IVF (In Vitro Fertilization)** is one of the most effective assisted reproductive technologies, helping millions of couples conceive.

**Key Points:**
- IVF involves combining eggs and sperm in a laboratory
- The process typically takes 4-6 weeks per cycle
- Success rates depend on age, health, and clinic expertise
- Multiple cycles may be needed

**Next Steps:**
1. Consult with a fertility specialist
2. Undergo necessary testing
3. Discuss your treatment options
4. Prepare emotionally and physically

Would you like more specific information about any aspect of IVF? I'm here to help!
"""

def generate_medical_image(prompt):
    """Generate medical illustration using DALL-E"""
    try:
        if client is None:
            # Generate placeholder images for yoga poses
            if "child" in prompt.lower() and "pose" in prompt.lower():
                return "/static/images/child_pose.jpg"
            elif "cat" in prompt.lower() and "cow" in prompt.lower():
                return "/static/images/cat_cow.jpg"
            elif "seated" in prompt.lower() and "twist" in prompt.lower():
                return "/static/images/seated_twist.jpg"
            else:
                return "/static/images/placeholder-medical.svg"

        # Enhance prompt for medical context
        enhanced_prompt = f"Medical illustration: {prompt}, professional medical style, clean, educational, accurate anatomy"

        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024",
            style="natural"
        )

        return response.data[0].url

    except Exception as e:
        # Fallback to placeholder images
        if "child" in prompt.lower() and "pose" in prompt.lower():
            return "/static/images/child_pose.jpg"
        elif "cat" in prompt.lower() and "cow" in prompt.lower():
            return "/static/images/cat_cow.jpg"
        elif "seated" in prompt.lower() and "twist" in prompt.lower():
            return "/static/images/seated_twist.jpg"
        else:
            return "/static/images/placeholder-medical.svg"


def generate_diagram(prompt):
    """Generate a technical/medical diagram using the generative image model.

    This is a light wrapper around the same DALL-E interface but
    emphasises diagrams and schematics in the prompt.  The front end
    can call /generate_diagram to obtain a URL for the generated image.
    """
    try:
        if client is None:
            return "/static/images/placeholder-diagram.svg"

        enhanced = (
            f"Technical schematic/diagram: {prompt}, clean line art style, "
            "labeled where appropriate, suitable for medical education"
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced,
            n=1,
            size="1024x1024",
            style="line_art"
        )
        return response.data[0].url
    except Exception as e:
        # fallback placeholder
        return "/static/images/placeholder-diagram.svg"

def get_nutrition_plan(patient_data):
    """Generate personalized nutrition plan using AI"""
    try:
        # Add randomness to ensure different results each time
        seed = random.randint(1, 10000)
        context = f"Generate a personalized nutrition plan for IVF patients. Seed: {seed}. "
        if patient_data:
            context += f"Patient details: Age {patient_data.age}, BMI {patient_data.bmi}, "
            if patient_data.diagnosis:
                context += f"Diagnosis: {patient_data.diagnosis}"

        context += """
        Provide a JSON response with:
        - daily_calories: recommended daily calories (vary between 1800-2500)
        - key_nutrients: list of 4-6 important nutrients with benefits (vary the nutrients each time)
        - meal_suggestions: breakfast, lunch, dinner, snacks (provide completely different meal ideas each time)
        - foods_to_avoid: list of foods to limit or avoid (vary the list each time)
        - supplements: recommended supplements (vary the supplements each time)
        - hydration: water intake recommendations
        Make sure all content is different each time this is called.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fertility nutrition specialist. Provide evidence-based nutrition advice for IVF patients. Always generate different content each time."},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        # Fallback nutrition plan with randomization
        calorie_options = ["1800-2200", "1900-2300", "2000-2400", "2100-2500"]
        daily_calories = random.choice(calorie_options)

        nutrient_options = [
            {"name": "Folic Acid", "benefit": "Supports embryo development"},
            {"name": "Iron", "benefit": "Prevents anemia during treatment"},
            {"name": "Calcium", "benefit": "Supports bone health"},
            {"name": "Omega-3", "benefit": "Reduces inflammation"},
            {"name": "Vitamin D", "benefit": "Enhances egg quality"},
            {"name": "Zinc", "benefit": "Supports hormone production"},
            {"name": "Iodine", "benefit": "Thyroid function support"},
            {"name": "Vitamin B12", "benefit": "Energy and red blood cell production"}
        ]
        key_nutrients = random.sample(nutrient_options, random.randint(4, 6))

        breakfast_options = [
            {"description": "Greek yogurt with berries, chia seeds, and walnuts.", "benefit": "High in antioxidants and omega-3s for egg quality."},
            {"description": "Oatmeal with almond milk, banana, and flaxseeds.", "benefit": "Provides sustained energy and fiber."},
            {"description": "Smoothie with spinach, banana, protein powder, and almond butter.", "benefit": "Nutrient-dense start to the day."},
            {"description": "Whole grain toast with avocado and poached eggs.", "benefit": "Healthy fats and protein for hormonal balance."},
            {"description": "Cottage cheese with pineapple and pumpkin seeds.", "benefit": "Calcium and zinc for reproductive health."}
        ]

        lunch_options = [
            {"description": "Grilled salmon salad with spinach, avocado, and quinoa.", "benefit": "Rich in protein and healthy fats for hormonal balance."},
            {"description": "Turkey wrap with whole grain tortilla, lettuce, tomato, and hummus.", "benefit": "Lean protein and vegetables for sustained energy."},
            {"description": "Lentil soup with mixed vegetables and whole grain bread.", "benefit": "Plant-based protein and fiber-rich meal."},
            {"description": "Chicken stir-fry with broccoli, carrots, and brown rice.", "benefit": "Anti-inflammatory ingredients and complete protein."},
            {"description": "Tuna salad with mixed greens, cucumber, and olive oil dressing.", "benefit": "Omega-3s and antioxidants for fertility support."}
        ]

        dinner_options = [
            {"description": "Stir-fried vegetables with tofu and brown rice.", "benefit": "Anti-inflammatory foods to support reproductive health."},
            {"description": "Baked chicken with sweet potato and steamed broccoli.", "benefit": "Complete protein and vitamin-rich vegetables."},
            {"description": "Grilled fish with quinoa and asparagus.", "benefit": "Omega-3s and folate for egg quality."},
            {"description": "Vegetable curry with chickpeas and whole grain naan.", "benefit": "Spice benefits and plant-based protein."},
            {"description": "Turkey meatballs with zucchini noodles and tomato sauce.", "benefit": "Lean protein and low-carb vegetables."}
        ]

        snack_options = [
            {"description": "Apple slices with almond butter and dark chocolate.", "benefit": "Provides sustained energy and fertility-boosting nutrients."},
            {"description": "Handful of mixed nuts and dried fruit.", "benefit": "Healthy fats and natural sweetness."},
            {"description": "Greek yogurt with honey and cinnamon.", "benefit": "Probiotics and anti-inflammatory spices."},
            {"description": "Carrot sticks with hummus.", "benefit": "Vitamin A and plant-based protein."},
            {"description": "Protein bar made with natural ingredients.", "benefit": "Convenient protein boost between meals."}
        ]

        meal_suggestions = [
            {"meal": "Breakfast", **random.choice(breakfast_options)},
            {"meal": "Lunch", **random.choice(lunch_options)},
            {"meal": "Dinner", **random.choice(dinner_options)},
            {"meal": "Snacks", **random.choice(snack_options)}
        ]

        avoid_options = [
            ["High mercury fish", "Excessive caffeine", "Processed foods", "Trans fats"],
            ["Artificial sweeteners", "Excess sugar", "Alcohol", "Unpasteurized dairy"],
            ["Soy products", "Raw eggs", "Undercooked meat", "Excess salt"],
            ["Fried foods", "Refined grains", "Artificial colors", "High fructose corn syrup"]
        ]
        foods_to_avoid = random.choice(avoid_options)

        supplement_options = [
            ["Prenatal vitamins", "Folic acid", "Vitamin D", "Omega-3"],
            ["CoQ10", "Myo-inositol", "DHEA", "Melatonin"],
            ["Probiotics", "Magnesium", "Vitamin C", "Selenium"],
            ["Acetyl-L-carnitine", "N-acetyl cysteine", "L-arginine", "Vitamin E"]
        ]
        supplements = random.choice(supplement_options)

        return {
            "daily_calories": daily_calories,
            "key_nutrients": key_nutrients,
            "meal_suggestions": meal_suggestions,
            "foods_to_avoid": foods_to_avoid,
            "supplements": supplements,
            "hydration": "8-10 glasses of water daily"
        }

def get_yoga_routine(patient_data):
    """Generate personalized yoga routine for IVF patients"""
    try:
        context = "Create a gentle yoga routine specifically designed for IVF patients. "
        if patient_data and patient_data.age:
            context += f"Patient age: {patient_data.age}. "
        
        context += """
        Provide a JSON response with:
        - routine_name: name of the routine
        - duration: total duration in minutes
        - poses: list of yoga poses with descriptions and benefits
        - breathing_exercises: breathing techniques
        - meditation: short meditation guidance
        - precautions: important safety notes for IVF patients
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fertility yoga specialist. Create safe, gentle yoga routines for IVF patients."},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        # Fallback yoga routine
        return {
            "routine_name": "Gentle IVF Support Routine",
            "duration": "25-30 minutes",
            "poses": [
                {"name": "Sukhasana (Easy Pose)", "description": "Sit comfortably cross-legged with a straight spine.", "benefit": "Calms the mind, reduces stress, and opens the hips."},
                {"name": "Baddha Konasana (Bound Angle Pose)", "description": "Sit with the soles of your feet together and let your knees fall to the sides.", "benefit": "Stimulates ovaries and improves blood flow to the pelvic region."},
                {"name": "Supta Baddha Konasana (Reclined Bound Angle)", "description": "Lie on your back with the soles of your feet together and knees out.", "benefit": "Promotes deep relaxation and opens the pelvic area."},
                {"name": "Cat-Cow Pose (Marjaryasana-Bitilasana)", "description": "On hands and knees, alternate between arching and rounding your back.", "benefit": "Improves spinal flexibility and relieves tension."},
                {"name": "Viparita Karani (Legs-Up-the-Wall Pose)", "description": "Lie on your back with your legs extended up against a wall.", "benefit": "Enhances blood circulation to the pelvis and calms the nervous system."},
                {"name": "Setu Bandhasana (Bridge Pose)", "description": "Lie on your back, bend your knees, and lift your hips off the floor.", "benefit": "Stretches the pelvic region and improves circulation to the uterus."},
                {"name": "Paschimottanasana (Seated Forward Bend)", "description": "Sit with legs extended and fold forward from the hips.", "benefit": "Stretches the hamstrings and lower back, stimulating the uterus and ovaries."},
                {"name": "Balasana (Child's Pose)", "description": "Kneel on the floor, sit back on your heels, and fold forward.", "benefit": "Deeply relaxing pose that helps to reduce stress and fatigue."},
                {"name": "Tadasana (Mountain Pose)", "description": "Stand tall with feet together, grounding through your feet and lengthening your spine.", "benefit": "Improves posture and creates a sense of stability and centeredness."},
                {"name": "Savasana (Corpse Pose)", "description": "Lie flat on your back with arms and legs relaxed.", "benefit": "Promotes deep relaxation, allowing the body to rest and repair."}
            ],
            "breathing_exercises": [
                {"name": "Nadi Shodhana (Alternate Nostril Breathing)", "description": "Balances energy and calms the mind."},
                {"name": "Bhramari Pranayama (Bee Breath)", "description": "Instantly relieves tension and anxiety."}
            ],
            "meditation": {"name": "Yoga Nidra (Yogic Sleep)", "description": "A 10-minute guided practice for deep physical and mental relaxation."},
            "precautions": [
                "Avoid deep twists and backbends",
                "No hot yoga during treatment",
                "Listen to your body and rest when needed",
                "Consult your doctor before starting any exercise program"
            ]
        }

def get_nutrition_analysis(meal_descriptions):
    """Analyze nutritional content of meals using AI"""
    try:
        context = f"""
        Analyze the nutritional content of the following meals for an IVF patient.
        Meals: {json.dumps(meal_descriptions)}

        Provide a JSON response with estimated values for:
        - total_calories: integer
        - protein_g: integer
        - folic_acid_mcg: integer
        - iron_mg: integer
        - omega_3_mg: integer
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition analysis expert. Estimate nutrient values from meal descriptions."},
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        # Fallback analysis
        return {
            "total_calories": 1950,
            "protein_g": 75,
            "folic_acid_mcg": 350,
            "iron_mg": 16,
            "omega_3_mg": 1200
        }


# ==================== Google Gemini API Functions ====================

def get_chatbot_response_gemini(message, user, patient_data=None):
    """Get AI chatbot response using Google Gemini API"""
    
    if not GOOGLE_API_KEY:
        # Return fallback if no API key
        return get_fallback_ivf_response(message, user, patient_data)
    
    try:
        # Build context based on user data
        context = f"You are an AI assistant specialized in IVF (In Vitro Fertilization) support. "
        context += f"You're speaking with {user.first_name}, a {user.user_type}. "
        
        if patient_data:
            context += f"Patient details: Age {patient_data.age}, "
            if patient_data.diagnosis:
                context += f"Diagnosis: {patient_data.diagnosis}, "
            if patient_data.previous_ivf_cycles:
                context += f"Previous IVF cycles: {patient_data.previous_ivf_cycles}, "
        
        context += """
        Provide helpful, accurate, and empathetic responses about:
        - IVF procedures and timelines
        - Medication guidance and side effects
        - Emotional support and encouragement
        - Appointment preparation
        - Lifestyle recommendations
        - Nutritional advice for fertility
        
        Always be supportive and remind users to consult their healthcare provider for medical decisions.
        Keep responses concise but informative.
        """
        
        # Prepare the prompt
        full_prompt = f"{context}\n\nUser question: {message}"
        
        # Call Google Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_CHAT_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("candidates") and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        
        # If API call fails, return fallback
        return get_fallback_ivf_response(message, user, patient_data)
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_fallback_ivf_response(message, user, patient_data)


def generate_tts_audio(text):
    """Generate TTS audio using Google Gemini API
    
    Returns:
        dict: Contains 'audio_data' (base64 encoded audio), 'mime_type', and 'sample_rate'
              or 'error' if failed
    """
    
    if not GOOGLE_API_KEY:
        return {"error": "No API key configured"}
    
    try:
        # Call Google Gemini TTS API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Iapetus"
                        }
                    }
                }
            }
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            part = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
            inline_data = part.get("inlineData", {})
            
            if inline_data:
                return {
                    "audio_data": inline_data.get("data"),
                    "mime_type": inline_data.get("mimeType", "audio/pcm")
                }
        
        return {"error": "Failed to generate audio"}
        
    except Exception as e:
        print(f"TTS API error: {e}")
        return {"error": str(e)}