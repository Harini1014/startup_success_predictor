from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('startup_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    funding = int(request.form['funding'])
    team_size = int(request.form['team_size'])
    market_size = int(request.form['market_size'])
    
    # Prepare input data for prediction (list of features)
    features = [[funding, team_size, market_size]]
    
    # Predict the result
    prediction = model.predict(features)
    
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
