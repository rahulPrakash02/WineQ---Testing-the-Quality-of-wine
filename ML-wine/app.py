from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('wine_mod.pkl', 'rb'))

app = Flask(__name__)



@app.route('/', methods = ['GET', 'POST'])
def man():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['alcohol']
    data2 = request.form['residual_sugar']
    data3 = request.form['fixed_acidity']
    data4 = request.form['pH']
    data5 = request.form['free_sulfur_dioxide']
    testArray = [data1, data2, data3, data4, data5]
    testArray = [float(x) for x in testArray]
    x = np.array(testArray)
    print(x)

    #standardization and fitting has a problem - check tomorrow morning
    scaler = StandardScaler()
    scaler.fit(x.reshape(1,-1))
    x_std = scaler.transform(x.reshape(1,-1))
    x = x_std
    print(x)

    pred = model.predict(x.reshape(1,-1))
    print(pred[0])
    return render_template('after.html', data=pred[0])


if __name__ == "__main__":
    app.run(debug=True)