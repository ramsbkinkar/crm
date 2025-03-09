// Fetch customer segmentation data
fetch('/customer_segment')
    .then(response => response.json())
    .then(data => {
        document.getElementById('segments').innerHTML = JSON.stringify(data);
    });

// Predict churn using customer input
function predictChurn() {
    let customerID = document.getElementById('customerID').value;
    fetch('/predict_churn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "CustomerID": customerID })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('churn').innerHTML = 'Churn Prediction: ' + (data.churn_prediction ? 'Yes' : 'No');
    });
}

// Fetch sales forecast data
fetch('/sales_forecast')
    .then(response => response.text())
    .then(data => {
        document.getElementById('forecast').innerHTML = data;
    });
