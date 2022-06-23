from flask import Flask, url_for, render_template, request, redirect
import matplotlib.pyplot as plt
import mpld3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

csv = pd.DataFrame(pd.read_csv("advertising.csv"))

def linear_reg(X, test):
    test = int(test)

    #linear regression
    X_train, X_test, y_train, y_test = train_test_split(csv[X], csv["Sales"], train_size=1-(test/100), test_size=(test/100), random_state=100)
    lr = LinearRegression()
    lr.fit(np.array(X_train).reshape(-1, 1), y_train)
    c = lr.intercept_
    m = lr.coef_
    
    pred = lr.predict(np.array(X_test).reshape(-1, 1))

    accuracy = lr.score(np.array(X_test).reshape(-1, 1), y_test)
    r2 = r2_score(y_test, pred)

    #plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    print(axes, axes.flat)
    axes[0].scatter(X_train, y_train)
    axes[0].plot(X_train, m*X_train + c, 'r')
    axes[0].title.set_text("Training Data")
    axes[1].scatter(X_test, y_test)
    axes[1].plot(X_test, m*X_test + c, 'r')
    axes[1].title.set_text("Test Data")
    html_str = mpld3.fig_to_html(fig)
    return html_str, accuracy, r2
    

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        getX = request.form['product']
        test = request.form['test_data']
        lr = linear_reg(getX, test)
        return render_template("index.html", fig=lr[0], accuracy=lr[1], r2=lr[2])
    return render_template("index.html")

@app.route("/data")
def data():
    arr = np.array(csv)
    return render_template("data.html", data=arr)

if __name__ == "__main__":
    app.run(debug=True)