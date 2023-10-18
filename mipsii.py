import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def get_data():
    X, y = load_wine(return_X_y=True, as_frame=True)
    return train_test_split(X, y, stratify=y, test_size=.2, random_state=21)

def initialize(xtr=None, ytr=None, model_path='model.pkl'):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    elif not (xtr is None or ytr is None):
        grid = {'n_estimators': [10, 50, 100],
                'learning_rate': [1e-2, 1e-3, 1e-4]}
        gs = GridSearchCV(GradientBoostingClassifier(random_state=21), grid, scoring='f1_weighted', cv=5, verbose=3)
        gs.fit(xtr, ytr)
        model = gs.best_estimator_
        with open(model_path, 'wb') as out:
            pickle.dump(model, out)
    else:
        raise Exception('There is neither train data nor a pretrained model!')
    return model

def predict(model, X):
    preds = pd.DataFrame(model.predict(X), index=X.index, columns=['prediction'])
    preds.to_csv('predictions.txt')
    return preds

def confmatr(ytrue, ypred):
    # print(ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred)))#.plot()
    from sklearn.metrics import confusion_matrix

    labels = [0, 1, 2]
    cm = confusion_matrix(ytrue, ypred)
    # print(cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    for i in range(len(cm)):
        for j in range(len(cm)):
            _ = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    xtr, xte, ytr, yte = get_data()
    model = initialize(xtr, ytr)
    preds = predict(model, xte)
    confmatr(yte, preds)

