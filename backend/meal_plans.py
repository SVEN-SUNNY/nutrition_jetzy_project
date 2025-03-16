import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Meal Plans Database
MEAL_PLANS = {
    0: {
        'name': 'Vegetarian Weight Loss',
        'meals': {
            'breakfast': 'Oatmeal with berries and chia seeds',
            'lunch': 'Quinoa salad with roasted vegetables',
            'dinner': 'Lentil curry with brown rice'
        },
        'calories': 1500,
        'macros': {'protein': 55, 'carbs': 180, 'fats': 40}
    },
    1: {
        'name': 'High Protein Muscle Gain',
        'meals': {
            'breakfast': 'Egg white omelette with spinach',
            'lunch': 'Grilled chicken with sweet potato',
            'dinner': 'Salmon with asparagus'
        },
        'calories': 2500,
        'macros': {'protein': 150, 'carbs': 200, 'fats': 70}
    },
    2: {
        'name': 'Low Carb Maintenance',
        'meals': {
            'breakfast': 'Avocado egg bake',
            'lunch': 'Zucchini noodle chicken stir-fry',
            'dinner': 'Cauliflower crust pizza'
        },
        'calories': 1800,
        'macros': {'protein': 90, 'carbs': 50, 'fats': 120}
    },
    3: {
        'name': 'Vegan Weight Loss',
        'meals': {
            'breakfast': 'Chia seed pudding',
            'lunch': 'Kale and roasted chickpea salad',
            'dinner': 'Tofu stir-fry with quinoa'
        },
        'calories': 1400,
        'macros': {'protein': 45, 'carbs': 160, 'fats': 35}
    },
    4: {
        'name': 'Athlete Performance',
        'meals': {
            'breakfast': 'Protein pancakes',
            'lunch': 'Turkey burger with sweet potato fries',
            'dinner': 'Grilled steak with wild rice'
        },
        'calories': 3000,
        'macros': {'protein': 180, 'carbs': 300, 'fats': 80}
    },
    5: {
        'name': 'Keto Weight Loss',
        'meals': {
            'breakfast': 'Bulletproof coffee with avocado',
            'lunch': 'Cauliflower rice chicken bowl',
            'dinner': 'Zucchini noodles with pesto and shrimp'
        },
        'calories': 1600,
        'macros': {'protein': 90, 'carbs': 30, 'fats': 120}
    },
    6: {
        'name': 'Mediterranean Heart Health',
        'meals': {
            'breakfast': 'Greek yogurt with walnuts and honey',
            'lunch': 'Grilled fish with olive tapenade',
            'dinner': 'Chickpea stew with whole grain pita'
        },
        'calories': 1800,
        'macros': {'protein': 80, 'carbs': 150, 'fats': 70}
    },
    7: {
        'name': 'Pescatarian Muscle Gain',
        'meals': {
            'breakfast': 'Smoked salmon omelette',
            'lunch': 'Tuna steak with quinoa',
            'dinner': 'Shrimp stir-fry with brown rice'
        },
        'calories': 2400,
        'macros': {'protein': 140, 'carbs': 200, 'fats': 80}
    },
    8: {
        'name': 'Gluten-Free Maintenance',
        'meals': {
            'breakfast': 'Buckwheat pancakes',
            'lunch': 'Stuffed bell peppers with ground turkey',
            'dinner': 'Baked chicken with mashed cauliflower'
        },
        'calories': 2000,
        'macros': {'protein': 100, 'carbs': 120, 'fats': 90}
    },
    9: {
        'name': 'Dairy-Free Energy Boost',
        'meals': {
            'breakfast': 'Chia pudding with almond milk',
            'lunch': 'Chicken salad with tahini dressing',
            'dinner': 'Beef curry with coconut milk'
        },
        'calories': 1900,
        'macros': {'protein': 110, 'carbs': 100, 'fats': 100}
    },
    10: {
        'name': 'Paleo Athletic Performance',
        'meals': {
            'breakfast': 'Sweet potato hash with eggs',
            'lunch': 'Grilled steak salad',
            'dinner': 'Bison burgers with sweet potato fries'
        },
        'calories': 2800,
        'macros': {'protein': 160, 'carbs': 150, 'fats': 110}
    },
    11: {
        'name': 'Low-Fat General Health',
        'meals': {
            'breakfast': 'Oat bran with fruit',
            'lunch': 'Turkey chili',
            'dinner': 'Baked cod with steamed vegetables'
        },
        'calories': 1700,
        'macros': {'protein': 95, 'carbs': 180, 'fats': 35}
    },
    12: {
        'name': 'High-Fiber Digestive Health',
        'meals': {
            'breakfast': 'Bran cereal with berries',
            'lunch': 'Lentil soup with whole grain bread',
            'dinner': 'Roasted vegetable quinoa bowl'
        },
        'calories': 1850,
        'macros': {'protein': 75, 'carbs': 220, 'fats': 50}
    },
    13: {
        'name': 'Plant-Based Endurance',
        'meals': {
            'breakfast': 'Tofu scramble',
            'lunch': 'Black bean Buddha bowl',
            'dinner': 'Tempeh stir-fry with brown rice'
        },
        'calories': 2300,
        'macros': {'protein': 85, 'carbs': 300, 'fats': 60}
    },
    14: {
        'name': 'Balanced Family Meals',
        'meals': {
            'breakfast': 'Whole grain waffles with fruit',
            'lunch': 'Chicken fajita bowl',
            'dinner': 'Salmon pasta primavera'
        },
        'calories': 2100,
        'macros': {'protein': 100, 'carbs': 200, 'fats': 70}
    }
}

def create_dataset() -> pd.DataFrame:
    """Create synthetic dataset for training"""
    np.random.seed(42)
    return pd.DataFrame({
        'diet': np.random.choice(
            ['vegetarian', 'high-protein', 'low-carb', 'vegan'], 
            500
        ),
        'goal': np.random.choice(
            ['weight-loss', 'muscle-gain', 'general-health'], 
            500
        ),
        'plan_id': np.random.choice(list(MEAL_PLANS.keys()), 500)
    })

def train_model() -> None:
    """Train and save the ML model"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Create and prepare dataset
        df = create_dataset()
        encoder = OneHotEncoder(handle_unknown='ignore')
        
        # Transform features
        features = encoder.fit_transform(df[['diet', 'goal']])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(features, df['plan_id'])
        
        # Save artifacts
        joblib.dump(model, 'model/nutrition_model.pkl')
        joblib.dump(encoder, 'model/feature_encoder.pkl')
        
        print("✅ Model trained and saved successfully")
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        raise

def load_model_artifacts():
    """Load model and encoder from disk"""
    try:
        model_path = os.path.join('model', 'nutrition_model.pkl')
        encoder_path = os.path.join('model', 'feature_encoder.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("Model files not found")
            
        return (
            joblib.load(model_path),
            joblib.load(encoder_path)
        )
    except Exception as e:
        print(f"❌ Error loading model artifacts: {str(e)}")
        return None, None

def generate_ml_plan(user_data, model, encoder):
    """Generate a meal plan using the ML model"""
    try:
        # Prepare input
        input_df = pd.DataFrame([{
            'diet': user_data['diet'],
            'goal': user_data['goal']
        }])
        
        # Encode features
        encoded = encoder.transform(input_df)
        
        # Predict plan
        plan_id = model.predict(encoded)[0]
        return MEAL_PLANS[plan_id]
    except Exception as e:
        raise ValueError(f"ML prediction failed: {str(e)}")

def get_rule_based_plan(user_data):
    """Fallback rule-based plan generation"""
    try:
        diet = user_data['diet'][0] if user_data['diet'] else 'vegetarian'
        goal = user_data['goal']
        
        if diet == 'vegetarian' and goal == 'weight-loss':
            return MEAL_PLANS[0]
        elif diet == 'high-protein' and goal == 'muscle-gain':
            return MEAL_PLANS[1]
        else:
            return MEAL_PLANS[2]  # Default plan
    except Exception as e:
        raise ValueError(f"Rule-based plan generation failed: {str(e)}")

if __name__ == '__main__':
    # Train and save model when run directly
    try:
        train_model()
        print("Model training completed successfully")
    except Exception as e:
        print(f"Model training failed: {str(e)}")
