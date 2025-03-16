from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/plan": {
        "origins": ["https://jetzy-nutrition-plan.netlify.app"],  # Replace with your frontend URL
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load ML model and encoder
try:
    model = joblib.load('nutrition_model.pk1')
    encoder = joblib.load('feature_encoder.pk1')
except Exception as e:
    print(f"Error loading ML artifacts: {str(e)}")
    model = None

# Meal Plans Database (from meal_plans.py)
MEAL_PLANS = {
    0: {'name': 'Vegetarian Weight Loss', 'meals': {'breakfast': 'Oatmeal with berries', 'lunch': 'Quinoa salad', 'dinner': 'Lentil curry'}, 'calories': 1500, 'macros': {'protein': 50, 'carbs': 200, 'fats': 40}},
    1: {'name': 'High Protein Muscle Gain', 'meals': {'breakfast': 'Egg white omelette', 'lunch': 'Grilled chicken', 'dinner': 'Salmon with asparagus'}, 'calories': 2500, 'macros': {'protein': 150, 'carbs': 180, 'fats': 70}},
    2: {'name': 'Low Carb Maintenance', 'meals': {'breakfast': 'Avocado egg bake', 'lunch': 'Zucchini noodles', 'dinner': 'Cauliflower crust pizza'}, 'calories': 1800, 'macros': {'protein': 90, 'carbs': 100, 'fats': 80}},
    3: {'name': 'Vegan Weight Loss', 'meals': {'breakfast': 'Chia pudding', 'lunch': 'Kale salad', 'dinner': 'Tofu stir-fry'}, 'calories': 1400, 'macros': {'protein': 40, 'carbs': 150, 'fats': 30}},
    4: {'name': 'Athlete Performance', 'meals': {'breakfast': 'Protein pancakes', 'lunch': 'Turkey burger', 'dinner': 'Grilled steak'}, 'calories': 3000, 'macros': {'protein': 200, 'carbs': 250, 'fats': 100}},
    5: {'name': 'Keto Weight Loss', 'meals': {'breakfast': 'Bulletproof coffee', 'lunch': 'Cauliflower rice bowl', 'dinner': 'Zucchini noodles'}, 'calories': 1600, 'macros': {'protein': 80, 'carbs': 50, 'fats': 120}},
    6: {'name': 'Mediterranean Heart Health', 'meals': {'breakfast': 'Greek yogurt', 'lunch': 'Grilled fish', 'dinner': 'Chickpea stew'}, 'calories': 1800, 'macros': {'protein': 90, 'carbs': 150, 'fats': 60}},
    7: {'name': 'Pescatarian Muscle Gain', 'meals': {'breakfast': 'Salmon omelette', 'lunch': 'Tuna steak', 'dinner': 'Shrimp stir-fry'}, 'calories': 2400, 'macros': {'protein': 160, 'carbs': 180, 'fats': 80}},
    8: {'name': 'Gluten-Free Maintenance', 'meals': {'breakfast': 'Buckwheat pancakes', 'lunch': 'Stuffed bell peppers', 'dinner': 'Baked chicken'}, 'calories': 2000, 'macros': {'protein': 100, 'carbs': 150, 'fats': 70}},
    9: {'name': 'Dairy-Free Energy Boost', 'meals': {'breakfast': 'Chia pudding', 'lunch': 'Chicken salad', 'dinner': 'Beef curry'}, 'calories': 1900, 'macros': {'protein': 90, 'carbs': 140, 'fats': 60}},
    10: {'name': 'Paleo Athletic Performance', 'meals': {'breakfast': 'Sweet potato hash', 'lunch': 'Steak salad', 'dinner': 'Bison burgers'}, 'calories': 2800, 'macros': {'protein': 180, 'carbs': 200, 'fats': 120}},
    11: {'name': 'Low-Fat General Health', 'meals': {'breakfast': 'Oat bran', 'lunch': 'Turkey chili', 'dinner': 'Baked cod'}, 'calories': 1700, 'macros': {'protein': 80, 'carbs': 150, 'fats': 40}},
    12: {'name': 'High-Fiber Digestive Health', 'meals': {'breakfast': 'Bran cereal', 'lunch': 'Lentil soup', 'dinner': 'Roasted vegetable bowl'}, 'calories': 1850, 'macros': {'protein': 70, 'carbs': 200, 'fats': 50}},
    13: {'name': 'Plant-Based Endurance', 'meals': {'breakfast': 'Tofu scramble', 'lunch': 'Black bean bowl', 'dinner': 'Tempeh stir-fry'}, 'calories': 2300, 'macros': {'protein': 90, 'carbs': 250, 'fats': 70}},
    14: {'name': 'Balanced Family Meals', 'meals': {'breakfast': 'Whole grain waffles', 'lunch': 'Chicken fajita bowl', 'dinner': 'Salmon pasta'}, 'calories': 2100, 'macros': {'protein': 100, 'carbs': 200, 'fats': 80}},
    15: {'name': 'Senior Health Plan', 'meals': {'breakfast': 'Oatmeal with almonds', 'lunch': 'Grilled fish with veggies', 'dinner': 'Turkey meatloaf'}, 'calories': 1800, 'macros': {'protein': 80, 'carbs': 150, 'fats': 60}},
    16: {'name': 'Pregnancy Nutrition Plan', 'meals': {'breakfast': 'Greek yogurt parfait', 'lunch': 'Spinach and cheese quesadilla', 'dinner': 'Grilled salmon with quinoa'}, 'calories': 2200, 'macros': {'protein': 90, 'carbs': 200, 'fats': 70}},
    17: {'name': 'Diabetes-Friendly Plan', 'meals': {'breakfast': 'Scrambled eggs with whole wheat toast', 'lunch': 'Grilled chicken with veggies', 'dinner': 'Baked fish with roasted Brussels sprouts'}, 'calories': 1800, 'macros': {'protein': 80, 'carbs': 150, 'fats': 60}},
    18: {'name': 'High-Calorie Mass Gain', 'meals': {'breakfast': 'Peanut butter smoothie', 'lunch': 'Beef burrito bowl', 'dinner': 'Steak with mashed potatoes'}, 'calories': 3500, 'macros': {'protein': 200, 'carbs': 300, 'fats': 150}},
    19: {'name': 'Budget-Friendly Nutrition', 'meals': {'breakfast': 'Banana oatmeal', 'lunch': 'Rice and beans', 'dinner': 'Vegetable stir-fry with tofu'}, 'calories': 1800, 'macros': {'protein': 70, 'carbs': 200, 'fats': 50}}
}

# API Endpoint
@app.route('/plan', methods=['POST'])
def get_plan():
    try:
        # Log incoming request
        print("Received request:", request.json)
        
        # Validate input
        data = request.json
        if not data.get('diet') or not data.get('goal'):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Generate plan
        if model:
            # Prepare input
            input_df = pd.DataFrame([{
                'diet': data['diet'],
                'goal': data['goal']
            }])
            
            # Encode features
            encoded = encoder.transform(input_df)
            
            # Predict plan
            plan_id = model.predict(encoded)[0]
            plan = MEAL_PLANS[plan_id]
        else:
            plan = get_rule_based_plan(data)
        
        # Log generated plan
        print("Generated plan:", plan)
        
        return jsonify({
            "meals": plan['meals'],
            "calories": plan['calories'],
            "macros": plan.get('macros', {})
        })
    
    except Exception as e:
        # Log error
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

# Fallback rule-based plan
def get_rule_based_plan(data):
    diet = data['diet'][0] if data['diet'] else 'vegetarian'
    goal = data['goal']
    
    if diet == 'vegetarian' and goal == 'weight-loss':
        return MEAL_PLANS[0]
    return MEAL_PLANS[1]

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
