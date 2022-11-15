

# You can choose the following classifiers:
# Naive bayes: "nb"
# Logistic regression: "lr"
# k nearest neighbours: "knn"
# support vector machine: "svm"

#please update the name variable below with your choice:

from functions import  plot, knn, result, svm, lr, nb

plot()


name = "knn"
#for training data, knn has highest accuracy


print("Classifier: ", name)
if name == "lr":
    result(lr(),"lr")
elif name == "svm":
    result(svm(),"svm")
elif name == "knn":
    result(knn(),"knn")
elif name== "nb":
    result(nb(),name="nb")
