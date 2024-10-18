document.getElementById('predictionForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const district = document.getElementById('district').value;
    const variety = document.getElementById('variety').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ district: district, variety: variety })
        });
        const result = await response.json();
        
        if (response.ok) {
            document.getElementById('result').innerText = `Predicted Price: ${result.predicted_price}`;
        } else {
            document.getElementById('result').innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById('result').innerText = `Request failed: ${error}`;
    }
});
