import numpy as np
import pickle
from flask import Flask, request, render_template

# Initialize the flask app
app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the root route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form and convert to float
        input_features = [float(x) for x in request.form.values()]
        
        # Arrange features into a numpy array
        final_features = np.array([input_features])
        
        # Scale the input features using the loaded scaler
        scaled_features = scaler.transform(final_features)
        
        # Make a prediction using the loaded model
        prediction = model.predict(scaled_features)
        
        # Determine the output message
        if prediction[0] == 1:
            output = "Potential Error Detected in the 3D Printer."
        else:
            output = "No Potential Error Detected in the 3D Printer."
            
    except Exception as e:
        output = f"An error occurred: {e}"

    # Render the HTML page again with the prediction result
    return render_template('index.html', prediction_text=output)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)