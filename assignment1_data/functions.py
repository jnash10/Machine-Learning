import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB

def plot():
    training_instances = pd.read_csv('training_data.csv')
    training_labels = pd.read_csv('training_data_class_labels.csv')

    data = pd.DataFrame()
    data['x']=training_instances.iloc[:,0]
    data['y']=training_instances.iloc[:,1]
    data['labels']=training_labels

    colors = []
    for label in data['labels']:
        if label==1:
            colors.append('red')
        else:
            colors.append('blue')


    plt.scatter(data['x'], data['y'], color = colors, alpha=0.5)
    plt.show()

    data.to_csv('clubbed.csv',  index=False)



def nb():
    x, y, x_train, x_test, y_train, y_test = get_data()
    nb = GaussianNB() 
    parameters = {}
    grid_search = GridSearchCV(nb, parameters, scoring='f1_micro')
    grid_search.fit(x_train, y_train)
    print("best parameters: ")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    
    y_test_hat = grid_search.predict(x_test)

    print(classification_report(y_test, y_test_hat))

    return grid_search            


def lr():
    x, y, x_train, x_test, y_train, y_test = get_data()
    lr = LogisticRegression() 
    parameters = {'random_state':(0,10)}
    grid_search = GridSearchCV(lr, parameters, scoring='f1_micro')
    grid_search.fit(x_train, y_train)
    print("best parameters: ")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    
    y_test_hat = grid_search.predict(x_test)

    print(classification_report(y_test, y_test_hat))

    return grid_search




def svm():
    x, y, x_train, x_test, y_train, y_test = get_data()
    svc = SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    grid_search = GridSearchCV(svc, parameters, scoring='f1_micro')
    grid_search.fit(x_train, y_train)
    print("best parameters: ")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    
    y_test_hat = grid_search.predict(x_test)

    print(classification_report(y_test, y_test_hat))

    return grid_search

def get_data():
    data = pd.read_csv('clubbed.csv')
    x=data.drop('labels', axis=1)
    y=data['labels']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=100)
    return x, y, x_train, x_test, y_train, y_test

def knn():
    x,y, x_train, x_test, y_train, y_test = get_data()
    knn = KNeighborsClassifier()
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='f1_micro')
    grid_search = grid.fit(x_train, y_train)
    print("best parameters for nearest neigbours are: ")
    print(grid_search.best_params_)
    neighbors = grid_search.best_params_['n_neighbors']
    #print(neighbors)
    
    f1 = grid_search.best_score_
    print("F1 score(training data): ", f1)

    new_knn = KNeighborsClassifier(n_neighbors=neighbors)

    new_knn.fit(x,y)
    y_test_hat = new_knn.predict(x_test)

    # precision = precision_score(y_test, y_test_hat)
    # recall = recall_score(y_test, y_test_hat)

    # f1 = f1_score(y_test, y_test_hat)
    # print("on test data: ")
    # print("precision: ", precision, "\nrecall: ",recall, "\nf1 micro", f1)
    print(classification_report(y_test, y_test_hat))

    return new_knn

def result(clf, name):
    temp_data = pd.read_csv('test_data.csv')
    x = temp_data.iloc[:,0]
    y=temp_data.iloc[:,1]

    data = pd.DataFrame()
    data['x']=x
    data['y']=y
    

    predictions = clf.predict(data)
    str_predictions = []
    for prediction in predictions:
        str_predictions.append(str(prediction)+'\n')

    
    file = open(str(name+".txt"),'w')
    file.writelines(str_predictions)

    file.close()













