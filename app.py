from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def helloworld():
    return render_template("heart_pred.html")
    
@app.route('/predict',methods=['POST','GET'])
def predict():  
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final=[np.array(int_features)]
    print(final)
    prediction=model.predict(final)
    output=round(prediction[0],2)
    print(output)
    if output==1:
        return render_template('heart_pred.html',pred='You have heart disease',result="heart disease")
    else:
        return render_template('heart_pred.html',pred='You dont have heart disease',result="not heart disease")

if __name__ == '__main__':
    app.run(debug=True)