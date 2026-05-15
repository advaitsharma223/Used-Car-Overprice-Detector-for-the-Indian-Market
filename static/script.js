/* ============================================================================
   USED CAR PRICE EVALUATOR — Frontend JavaScript
   ============================================================================
   
   WHAT THIS DOES:
   1. Handles form submission
   2. Sends data to the Flask backend via fetch() API
   3. Displays the result in a nice card
   
   KEY CONCEPTS:
   - Event Listeners: Respond to user actions (like form submit)
   - fetch(): Modern way to make HTTP requests in JavaScript
   - async/await: Cleaner way to handle asynchronous operations
   - DOM Manipulation: Updating HTML elements dynamically
   ============================================================================ */

// Wait for page to fully load before running our code
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('car-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultCard = document.getElementById('result-card');

    // Handle form submission
    form.addEventListener('submit', async function(event) {
        // Prevent the browser's default form submission (page reload)
        event.preventDefault();

        // Show loading state
        btnText.textContent = 'Evaluating...';
        btnLoader.style.display = 'inline-block';
        submitBtn.disabled = true;
        resultCard.classList.add('hidden');

        try {
            // Collect form data
            const formData = new FormData(form);
            
            // Convert to JSON object
            const data = {
                brand: formData.get('brand'),
                location: formData.get('location'),
                year: formData.get('year'),
                kilometers_driven: formData.get('kilometers_driven'),
                fuel_type: formData.get('fuel_type'),
                transmission: formData.get('transmission'),
                owner_type: formData.get('owner_type'),
                mileage: formData.get('mileage'),
                engine: formData.get('engine'),
                power: formData.get('power'),
                seats: formData.get('seats'),
                listed_price: formData.get('listed_price')
            };

            // Send POST request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                displayResult(result);
            } else {
                alert('Error: ' + result.error);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong. Please try again.');
        } finally {
            // Reset button state
            btnText.textContent = 'Evaluate Price';
            btnLoader.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    /**
     * Display the prediction result in the result card
     */
    
    function displayResult(result) {
        //console.log(result);
        const resultHeader = resultCard.querySelector('.result-header');
        const resultIcon = document.getElementById('result-icon');
        const resultVerdict = document.getElementById('result-verdict');
        const listedPriceEl = document.getElementById('listed-price-result');
        const predictedPriceEl = document.getElementById('predicted-price-result');
        const deviationFill = document.getElementById('deviation-fill');
        const deviationText = document.getElementById('deviation-text');
        const resultMessage = document.getElementById('result-message');
        const confidenceBadge = document.getElementById('confidence-badge');
        const aiAnalysis = document.getElementById('ai-analysis');
        aiAnalysis.textContent = result.ai_analysis;
        
        console.log(aiAnalysis);

        // Set verdict styling based on result
        resultHeader.className = 'result-header';
        deviationFill.className = 'deviation-fill';

        if (result.verdict === 'Overpriced') {
            resultHeader.classList.add('overpriced');
            resultIcon.textContent = '⚠️';
            deviationFill.classList.add('overpriced');
        } else if (result.verdict === 'Underpriced') {
            resultHeader.classList.add('underpriced');
            resultIcon.textContent = '✅';
            deviationFill.classList.add('underpriced');
        } else {
            resultHeader.classList.add('fair');
            resultIcon.textContent = '👍';
            deviationFill.classList.add('fair');
        }

        // Set text content
        resultVerdict.textContent = result.verdict;
        listedPriceEl.textContent = '₹' + result.listed_price.toFixed(2) + 'L';
        predictedPriceEl.textContent = '₹' + result.predicted_price.toFixed(2) + 'L';

        // Calculate deviation bar width (max 100%)
        const deviationPct = Math.min(Math.abs(result.deviation_percent), 50);
        deviationFill.style.width = (deviationPct * 2) + '%';

        // Set deviation text
        const direction = result.deviation_percent > 0 ? 'above' : 'below';
        deviationText.textContent = Math.abs(result.deviation_percent).toFixed(1) + '% ' + direction + ' fair value';

        // Set message
        resultMessage.textContent = result.message;

        // Set confidence badge
        confidenceBadge.textContent = result.confidence.replace('-', ' ') + ' confidence';
        confidenceBadge.className = 'confidence-badge ' + result.confidence.replace(' ', '-');

        // Show the result card
        resultCard.classList.remove('hidden');

        // Scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
