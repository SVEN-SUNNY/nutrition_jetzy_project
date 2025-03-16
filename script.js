// script.js - Root Directory Version
document.addEventListener('DOMContentLoaded', () => {
    const API_ENDPOINT = 'https://nutrition-jetzy-backend.onrender.com/plan'; // REPLACE WITH YOUR RENDER URL
    const form = document.getElementById('nutritionForm');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');

    // Initialize default state
    let isSubmitting = false;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isSubmitting) return;
        
        try {
            isSubmitting = true;
            toggleLoading(true);
            clearErrors();
            resetResults();

            const formData = getFormData();
            validateFormData(formData);

            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            handleResponseErrors(response);
            const plan = await response.json();
            
            updateMealPlanUI(plan);
            resultSection.classList.remove('hidden');

        } catch (error) {
            handleError(error);
        } finally {
            isSubmitting = false;
            toggleLoading(false);
        }
    });

    // Core Functions
    function getFormData() {
        return {
            diet: Array.from(document.querySelectorAll('#diet option:checked'))
                     .map(option => option.value),
            goal: document.getElementById('goal').value
        };
    }

    function validateFormData(data) {
        if (data.diet.length === 0) throw new Error('Please select at least one dietary preference');
        if (!data.goal) throw new Error('Please select a health goal');
    }

    function handleResponseErrors(response) {
        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status} ${response.statusText}`);
        }
    }

    // UI Functions
    function toggleLoading(show) {
        loader.classList.toggle('hidden', !show);
        form.style.opacity = show ? 0.5 : 1;
        form.style.pointerEvents = show ? 'none' : 'all';
    }

    function updateMealPlanUI(plan) {
        // Update meals
        updateMeal('breakfast', plan.meals.breakfast, plan.macros.breakfast);
        updateMeal('lunch', plan.meals.lunch, plan.macros.lunch);
        updateMeal('dinner', plan.meals.dinner, plan.macros.dinner);

        // Update totals
        document.getElementById('totalCalories').textContent = plan.calories || 'N/A';
        document.getElementById('totalProtein').textContent = plan.macros.daily.protein || '0';
        document.getElementById('totalCarbs').textContent = plan.macros.daily.carbs || '0';
        document.getElementById('totalFats').textContent = plan.macros.daily.fats || '0';
    }

    function updateMeal(mealType, mealName, macros) {
        document.getElementById(`${mealType}Meal`).textContent = mealName || 'Recommendation pending';
        document.getElementById(`${mealType}Calories`).textContent = macros?.calories || '0';
        document.getElementById(`${mealType}Protein`).textContent = macros?.protein || '0';
    }

    function handleError(error) {
        console.error('Error:', error);
        const errorMessage = error.message.includes('HTTP Error') 
            ? 'Server error. Please try again later.' 
            : error.message;
        showError(errorMessage);
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            ${message}
        `;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    function clearErrors() {
        document.querySelectorAll('.error-message').forEach(el => el.remove());
    }

    function resetResults() {
        ['breakfast', 'lunch', 'dinner'].forEach(meal => {
            document.getElementById(`${meal}Meal`).textContent = 'Loading...';
            document.getElementById(`${meal}Calories`).textContent = '0';
            document.getElementById(`${meal}Protein`).textContent = '0';
        });
    }
});
