from flask import Flask, render_template, request
from model import predict_price
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

df = pd.read_csv("laptop_data.csv")

@app.route('/')
def home():
    brands = sorted(df['Brand'].unique())
    processors = sorted(df['Processor'].unique())
    os_list = sorted(df['Operating_System'].unique())

    return render_template("index.html",
                           brands=brands,
                           processors=processors,
                           os_list=os_list)

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['brand']
    processor = request.form['processor']
    ram = int(request.form['ram'])
    screen_size = float(request.form['screen_size'])
    ssd = int(request.form['ssd'])
    os = request.form['os']

    prediction = predict_price(brand, processor, ram, screen_size, ssd, os)

    return render_template("index.html",
                           prediction=prediction,
                           brands=sorted(df['Brand'].unique()),
                           processors=sorted(df['Processor'].unique()),
                           os_list=sorted(df['Operating_System'].unique()))

if __name__ == "__main__":
    app.run(debug=True)
