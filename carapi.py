from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('carDTC.pkl')
MODEL_LABELS = ['Unacceptable', 'Acceptable', 'Very Good', 'Good']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    Buying = request.args.get('Buying')
    Maint = request.args.get('Maint')
    doors = request.args.get('doors')
    persons = request.args.get('persons')
    lug_boot = request.args.get('lug_boot')
    safety = request.args.get('safety')

    # Our model expects a list of records
    features = [[Buying, Maint, doors, persons, lug_boot, safety]]

    # predict the class and probability of the class
    label_index = MODEL.predict(features)
    
    # get the probabilities list for the prediction
    label_conf = MODEL.predict_proba(features)

    # list down each class with the probabilty value
    probs = ' Unacceptable = {}, Acceptable = {}, Very Good = {}, Good = {}'.format(label_conf[0][0], label_conf[0][1], label_conf[0][2], label_conf[0][3])

    # Retrieve the name of the predicted class
    label = MODEL_LABELS[label_index[0]]

    # Create and send a response to the API caller
    return jsonify(status='complete', Prediction=label, confidence = ''.join(str(label_conf)), probabilities = probs)



if __name__ == '__main__':
    app.run(debug=True)