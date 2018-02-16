from flask import Flask,render_template,redirect,url_for,request,session,flash
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
import math


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']= 0
app.secret_key = "Virinchi"

@app.route('/', methods = ['GET','POST'])
def test():
    return render_template("index.html")


@app.route('/form_submit', methods = ['POST'])
def form_submit():
    my_list =[]
    my_list.append(float(request.form['A1']))
    my_list.append(float(request.form['A2']))
    my_list.append(float(request.form['A3']))
    my_list.append(float(request.form['A4']))
    my_list.append(float(request.form['A5']))
    my_list.append(float(request.form['A6']))
    my_list.append(float(request.form['A7']))
    my_list.append(float(request.form['A8']))
    my_list.append(float(request.form['A9']))
    my_list.append(float(request.form['A10']))
    my_list1 = np.asarray(my_list)
    print(my_list1)

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    data = pd.read_csv(url, names=names)

    attributes = [ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

    #Changing target variable values to Binary
    d = {'B': 0, 'M': 1}
    data['diagnosis']=data['diagnosis'].map(d)

    #Dropping the ID column
    data = data.drop(['id'],axis = 1)

    #Splitting the data into train and test
    train, test = train_test_split(data, test_size = 0.3, random_state = 5, stratify = data['diagnosis'])
    X_train = train.loc[:, data.columns.difference(['diagnosis'])]
    y_train = train.loc[:, 'diagnosis']
    X_test = test.loc[:, data.columns.difference(['diagnosis'])]
    y_test = test.loc[:, ['diagnosis']]

    #Feature Variables
    feature_names = ["perimeter_mean","area_mean","concavity_mean","concave points_mean","radius_worst","texture_worst", "perimeter_worst", "area_worst","concave points_worst","symmetry_worst"]

    X_train = X_train[feature_names]
    X_test = X_test[feature_names]

    #Min max normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X_train)
    X_train = pd.DataFrame(np_scaled, columns = X_train.columns)
    X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=X_test.columns)

    #Modelling
    log_reg = LogisticRegression(C = 100)
    log_reg.fit(X_train, y_train)
    y_test_pred =log_reg.predict(X_test)
    print(y_test_pred)
    # from sklearn.neural_network import MLPClassifier
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                  hidden_layer_sizes=(15, 10), random_state=1)
    # clf.fit(X_train, y_train)
    # y_test_pred = clf.predict(X_test)
    # print(y_test_pred)

    #Reshaping the input to a row vector
    my_list1 = my_list1.reshape(1, -1)
    my_list1 = min_max_scaler.transform(my_list1)
    y_test_pred1 = log_reg.predict(my_list1)
    if y_test_pred1 == 1:
        y_test_pred1 = 'Malignant'
    if y_test_pred1 == 0:
        y_test_pred1 = 'Benign'
    print(y_test_pred1)

    #Calculating the metrics
    recall1 = recall_score(y_test,y_test_pred)*100
    recall = math.ceil(recall1*100)/100
    precision1 = precision_score(y_test,y_test_pred)*100
    precision = math.ceil(precision1*100)/100
    flash("Sensitivity is: ")
    Sensitivity = recall
    flash("Precision is:")
    Precision = precision
    flash("The Tumor is:")
    y_test_pred2 = y_test_pred1

    return render_template("index.html",result = Sensitivity, result1 = Precision, result2 = y_test_pred2)

if __name__ == '__main__':
    app.run(debug = True, port = 4995)
