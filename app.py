from flask import Flask, url_for, render_template, request, redirect
import load_data
import matplotlib.pyplot as plt
import mpld3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def linear_reg(X, test):
    test = int(test)
    X_train, X_test, y_train, y_test = train_test_split(csv[X], csv["Sales"], train_size=1-(test/100), test_size=(test/100), random_state=100)
    lr = LinearRegression()
    lr.fit(np.array(X_train).reshape(-1, 1), y_train)
    c = lr.intercept_
    m = lr.coef_
    # pred = lr.predict(np.array(X_test).reshape(-1, 1))
    fig, axes = plt.subplots()
    axes.scatter(X_train, y_train)
    axes.plot(X_train, m*X_train + c, 'r')
    html_str = mpld3.fig_to_html(fig)
    return html_str
    

app = Flask(__name__)

csv = load_data.data

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        getX = request.form['product']
        test = request.form['test_data']
        return render_template("new.html", fig=linear_reg(getX, test))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)