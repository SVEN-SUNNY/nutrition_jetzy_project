# backend/meal_plans.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Meal Plan Database
MEAL_PLANS = {
    0: {
        'name': 'Vegetarian Weight Loss',
        'meals': {
            'breakfast': 'Oatmeal with berries and almond butter',
            'lunch': 'Quinoa salad with chickpeas and vegetables',
            'dinner': 'Grilled vegetable wrap with hummus'
        },
        'calories': 1500,
        'macros': {
            'protein': 55,
            'carbs': 180,
            'fats': 45
        }
    },
    1: {
        'name': 'High Protein Muscle Gain',
        'meals': {
            'breakfast': 'Egg white omelette with spinach and whole grain toast',
            'lunch': 'Grilled chicken breast with brown rice and broccoli',
            'dinner': 'Salmon with sweet potato and asparagus'
        },
        'calories': 2500,
        'macros': {
            'protein': 150,
            'carbs': 250,
            'fats': 70
        }
    },
    2: {
        'name': 'Balanced Maintenance',
        'meals': {
            'breakfast': 'Greek yogurt with granola and fruit',
            'lunch': 'Turkey and avocado sandwich',
            'dinner': 'Lean beef stir-fry with mixed vegetables'
        },
        'calories': 2000,
        'macros': {
            'protein': 100,
            'carbs': 200,
            'fats': 60
        }
    }
}

def create_dataset():
    """Generate synthetic training data"""
    np.random.seed(42)
    data = {
        'diet': np.random.choice(['vegetarian', 'high-protein', 'vegan', 'low-carb'], 500),
        'goal': np.random.choice(['weight-loss', 'muscle-gain', 'maintenance'], 500),
        'age': np.random.randint(18, 65, 500),
        'activity_level': np.random.choice(['sedentary', 'light', 'moderate', 'active'], 500),
        'plan_id': np.random.choice(list(MEAL_PLANS.keys()), 500)
    }
    return pd.DataFrame(data)

def train_model():
    """Train and save ML model"""
    try:
        # Create synthetic dataset
        df = create_dataset()
        
        # Prepare features
        encoder = OneHotEncoder()
        features = encoder.fit_transform(df[['diet', 'goal', 'activity_level']])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, df['plan_id'])
        
        # Save artifacts
        joblib.dump(model, 'model/nutrition_model.pkl')
        joblib.dump(encoder, 'model/feature_encoder.pkl')
        
        print("Model trained and saved successfully")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def generate_ml_plan(user_data, model, encoder):
    """Generate plan using ML model"""
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'diet': user_data.get('diet', ['vegetarian'])[0],
            'goal': user_data.get('goal', 'weight-loss'),
            'activity_level': user_data.get('activity_level', 'moderate')
        }])
        
        # Transform features
        encoded = encoder.transform(input_df)
        
        # Predict
        plan_id = model.predict(encoded)[0]
        return MEAL_PLANS[plan_id]
        
    except Exception as e:
        raise ValueError(f"ML prediction failed: {str(e)}")

def get_rule_based_plan(user_data):
    """Fallback rule-based recommendation"""
    try:
        diet = user_data.get('diet', ['vegetarian'])[0]
        goal = user_data.get('goal', 'weight-loss')
        
        # Simple rule-based matching
        if 'vegetarian' in diet and goal == 'weight-loss':
            return MEAL_PLANS[0]
        elif 'high-protein' in diet and goal == 'muscle-gain':
            return MEAL_PLANS[1]
        return MEAL_PLANS[2]
        
    except Exception as e:
        return MEAL_PLANS[2]  # Default plan

if __name__ == '__main__':
    # Train model when file is run directly
    train_model()
