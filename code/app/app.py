from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_price', methods=['POST', 'GET'])
def predict_price():
    # get the parameters
    shot_power = float(request.form['shot_power'])
    heading_accuracy = float(request.form['heading_accuracy'])
    finishing = float(request.form['finishing'])
    ball_control = float(request.form['ball_control'])
    overall = float(request.form['overall'])

    # load the model and predict
    model = joblib.load('model/linear_regression.pkl')
    prediction = model.predict([[shot_power, heading_accuracy, finishing, ball_control, overall]])
    predicted_price = prediction.round(1)[0]

    return render_template('results.html',
                           shot_power=int(shot_power),
                           heading_accuracy=int(heading_accuracy),
                           finishing=int(finishing),
                           ball_control=int(ball_control),
                           overall=int(overall),
                           predicted_price="{:,}".format(predicted_price)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
