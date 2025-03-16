// API Configuration - MUST REPLACE WITH YOUR RENDER URL
const API_URL = 'https://nutrition-jetzy-backend.onrender.com/plan';

// DOM Elements
const form = document.getElementById('nutritionForm');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('resultSection');

// Meal Elements
const mealElements = {
  breakfast: {
    meal: document.getElementById('breakfastMeal'),
    calories: document.getElementById('breakfastCalories'),
    protein: document.getElementById('breakfastProtein')
  },
  lunch: {
    meal: document.getElementById('lunchMeal'),
    calories: document.getElementById('lunchCalories'),
    protein: document.getElementById('lunchProtein')
  },
  dinner: {
    meal: document.getElementById('dinnerMeal'),
    calories: document.getElementById('dinnerCalories'),
    protein: document.getElementById('dinnerProtein')
  }
};

// Macro Elements
const macroElements = {
  protein: document.getElementById('totalProtein'),
  carbs: document.getElementById('totalCarbs'),
  fats: document.getElementById('totalFats')
};

// Global Abort Controller
let abortController = null;

// Event Listeners
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  try {
    // Abort previous request if exists
    if (abortController) abortController.abort();
    
    // Initialize new abort controller
    abortController = new AbortController();
    const signal = abortController.signal;
    
    // Show loading state
    toggleLoading(true);
    clearErrors();
    
    // Get and validate form data
    const formData = getFormData();
    validateFormData(formData);
    
    // Fetch plan with timeout
    const plan = await fetchWithTimeout(API_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(formData),
      signal
    }, 10000); // 10 second timeout

    // Update UI with received plan
    updatePlanUI(plan);
    
  } catch (error) {
    handleError(error);
  } finally {
    toggleLoading(false);
  }
});

// Helper Functions
function getFormData() {
  return {
    diet: Array.from(document.querySelectorAll('#diet option:checked'))
              .map(option => option.value),
    goal: document.getElementById('goal').value
  };
}

function validateFormData(formData) {
  if (!formData.diet.length) throw new Error('Please select at least one dietary preference');
  if (!formData.goal) throw new Error('Please select a health goal');
}

async function fetchWithTimeout(url, options, timeout) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('API Error Response:', errorData);
      throw new Error(errorData.error || 'Server returned an error');
    }
    
    return await response.json();
    
  } catch (error) {
    console.error('Fetch Error:', error);
    throw error;
  }
}

function updatePlanUI(plan) {
  // Update meals
  Object.entries(mealElements).forEach(([mealType, elements]) => {
    elements.meal.textContent = plan.meals[mealType];
    elements.calories.textContent = plan.macros[mealType].calories;
    elements.protein.textContent = plan.macros[mealType].protein;
  });

  // Update macros
  Object.entries(macroElements).forEach(([macroType, element]) => {
    element.textContent = `${plan.macros.daily[macroType]}g`;
  });

  // Show results
  resultSection.classList.remove('hidden');
}

function toggleLoading(show) {
  loader.classList.toggle('hidden', !show);
  form.querySelector('button').disabled = show;
}

function handleError(error) {
  console.error('Application Error:', error);
  
  const errorMessage = error.name === 'AbortError' 
    ? 'Request timed out. Please try again.'
    : error.message || 'Failed to generate plan. Please try again later.';

  showError(errorMessage);
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

function clearErrors() {
  const existingErrors = document.querySelectorAll('.error-message');
  existingErrors.forEach(error => error.remove());
}

// Initialize default state
function init() {
  resultSection.classList.add('hidden');
  loader.classList.add('hidden');
  clearErrors();
}

// Start application
init();
