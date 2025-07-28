# Smart-Doping-Detection
Smart Doping Detection identifies whether an athlete has used performance-enhancing substances. If doping is detected, it further classifies the specific type of doping involved.

## 🚀 Features

- Predicts whether an athlete is doped or not.
- Classifies the type of doping (if detected).
- Flask-based web interface for easy input and results.
- Utilizes trained ML models saved as `.pkl` files.

## 🧠 Input Features

- Age  
- Test Results  
- Biological Passport Anomaly  
- Travel Alerts  
- Medical Record Flag  
- Social Media Alerts  
- Investigation Status  
- Whistleblower Report  
- Case Risk Score  
- Sanction Severity  
- Test Frequency  
- Abnormal Hemoglobin  
- Steroid Profile Deviation  
- Athlete Performance Change  
- Training Location Flags  
- Medical Exemptions  
- Doping Association Score  

## 💻 How to Run Locally

1. Clone the repository or download the ZIP file.
2. Make sure you have Python installed (preferably 3.8+).
3. Navigate to the project directory and install dependencies:
   ```bash
   pip install flask pandas scikit-learn
   ````
4. Run the Flask app:

   ```bash
   python app.py
   ```
5. Open your browser and visit: [http://localhost:5000](http://localhost:5000)
