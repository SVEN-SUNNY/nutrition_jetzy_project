// script.js
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('nutritionForm');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');
    
    // Replace with your Render backend URL
    const API_URL = 'https://nutrition-jetzy-backend.onrender.com';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get form data
        const formData = {
            diet: Array.from(document.querySelectorAll('#diet option:checked'))
                      .map(option => option.value),
            goal: document.getElementById('goal').value
        };

        // Basic validation
        if (formData.diet.length === 0 || !formData.goal) {
            showError('Please fill in all required fields');
            return;
        }

        try {
            // Show loading state
            toggleLoading(true);
            resetResults();

            // API call
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const plan = await response.json();
            displayPlan(plan);

        } catch (error) {
            console.error('API Error:', error);
            showError('Failed to generate plan. Please try again later.');
        } finally {
            toggleLoading(false);
        }
    });

    function toggleLoading(isLoading) {
        if (isLoading) {
            loader.classList.remove('hidden');
            resultSection.classList.add('hidden');
        } else {
            loader.classList.add('hidden');
        }
    }

    function displayPlan(plan) {
        // Update meals
        document.getElementById('breakfastMeal').textContent = plan.meals.breakfast;
        document.getElementById('lunchMeal').textContent = plan.meals.lunch;
        document.getElementById('dinnerMeal').textContent = plan.meals.dinner;

        // Update calories
        document.getElementById('totalCalories').textContent = plan.calories;

        // Update macros
        document.getElementById('proteinValue').textContent = plan.macros.protein;
        document.getElementById('carbsValue').textContent = plan.macros.carbs;
        document.getElementById('fatsValue').textContent = plan.macros.fats;

        // Show results
        resultSection.classList.remove('hidden');
    }

    function showError(message) {
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            ${message}
        `;
        document.body.appendChild(errorEl);
        setTimeout(() => errorEl.remove(), 5000);
    }

    function resetResults() {
        // Reset all result fields
        const resetFields = [
            'breakfastMeal', 'lunchMeal', 'dinnerMeal',
            'totalCalories', 'proteinValue', 'carbsValue', 'fatsValue'
        ];
        
        resetFields.forEach(id => {
            document.getElementById(id).textContent = '...';
        });
    }
});
