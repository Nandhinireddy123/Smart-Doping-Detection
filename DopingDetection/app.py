import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from doping_model import load_models
detection_model, type_model = load_models()
# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Define the feature columns expected by the models
feature_columns = [
    'Age', 'Biological_Passport_Anomaly', 'Travel_Alerts', 'Medical_Record_Flag',
    'Social_Media_Alerts', 'Investigation_Status', 'Whistleblower_Report',
    'Case_Risk_Score', 'Sanction_Severity', 'Test_Frequency', 'Abnormal_Hemoglobin',
    'Steroid_Profile_Deviation', 'Athlete_Performance_Change', 'Training_Location_Flags',
    'Medical_Exemptions', 'Doping_Association_Score'
]

doping_types = ['Steroids', 'EPO', 'Stimulants', 'Diuretics', 'HGH']

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/templates/<path:path>')
def serve_template(path):
    """Serve template files"""
    return send_from_directory('templates', path)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data and make predictions"""
    try:
        # Get form data and convert to appropriate types
        form_data = {}
        for feature in feature_columns:
            value = request.form.get(feature)
            if value is None:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            
            # Convert to appropriate type
            if feature in ['Abnormal_Hemoglobin', 'Athlete_Performance_Change']:
                form_data[feature] = float(value)
            else:
                form_data[feature] = int(value)
        
        # Create DataFrame with expected columns
        input_df = pd.DataFrame([form_data], columns=feature_columns)
        
        # Make doping detection prediction
        doping_probability = detection_model.predict_proba(input_df)[0][1] * 100
        doping_detected = doping_probability > 50
        
        # Initialize doping type variables
        doping_type = None
        doping_type_probabilities = {}

        # Only predict doping type if doping is detected or probability is high
        if doping_detected or doping_probability > 30:
            try:
                doping_type = type_model.predict(input_df)[0]
                type_probs = type_model.predict_proba(input_df)[0]
                for i, dtype in enumerate(type_model.classes_):
                    doping_type_probabilities[dtype] = float(type_probs[i] * 100)
            except Exception as e:
                print(f"Error predicting doping type: {e}")
                for dtype in doping_types:
                    doping_type_probabilities[dtype] = 20.0
                doping_type = np.random.choice(doping_types)
        else:
            for dtype in doping_types:
                doping_type_probabilities[dtype] = 0.0
        
        # Prepare response
        response = {
            'doping_detected': bool(doping_detected),
            'doping_probability': float(doping_probability),
            'doping_type': doping_type,
            #'doping_type_probabilities': doping_type_probabilities
        }

        print("Prediction response:", response)
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)

    if not os.path.exists('templates/index.html'):
        print("Creating index.html in templates directory...")
        with open('templates/index.html', 'w') as f:
            try:
                with open('index.html', 'r') as source:
                    html_content = source.read()
                f.write(html_content)
            except FileNotFoundError:
                f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Smart Doping Detection System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <h2>Smart Doping Detection System</h2>
        <form id="prediction-form">
            <!-- Form fields will go here -->
            <button type="submit" class="btn btn-primary">Submit</button>
        </form>
        <div id="prediction-result" class="mt-3" style="display: none;">
            <h3>Prediction Results</h3>
            <p><strong>Result:</strong> <span id="result"></span></p>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#prediction-form').submit(function(event) {
                event.preventDefault();
                // Form submission handling will go here
            });
        });
    </script>
</body>
</html>
                """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
