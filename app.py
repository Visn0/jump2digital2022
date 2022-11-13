import pandas as pd
import numpy as np

from os.path import join
import os
import sys

DATA_PATH = 'data/'
TRAIN_PATH= os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')

'''
    Change column names of the datasets in order to have shorter names
'''
def change_column_names(df, df_test):
    old_cols  = list(df.columns)
    feat_cols = ['f'+str(x) for x in range(len(old_cols) - 1)]
    new_cols  = feat_cols + old_cols[-1:]
    target    = 'target'

    df.columns = new_cols
    df_test.columns = feat_cols

    return df, df_test, feat_cols, target

'''
    Load datasets for training and prediction
'''
def load_data():
    df      = pd.read_csv(TRAIN_PATH, sep=';')
    df_test = pd.read_csv(TEST_PATH, sep=';')
    print(df.shape, "raw shape df train")
    print(df_test.shape, "raw shape df test")
    return df, df_test

'''
    Trains and returns a model
'''
def train_model(df, feat_cols, target):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import make_scorer

    X_train, X_val, y_train, y_val = train_test_split(df[feat_cols], df[target], test_size=0.2, random_state=42)
    f1 = make_scorer(f1_score , average='macro')

    param_grid = {
        'n_estimators': [500, 800, 2000],
        'max_depth' : [8, 10, 20, None],
        'max_features': ['log2'],
        'criterion' :['entropy'],
        'class_weight' : [
            None,
            {0:0.25,1:0.25,2:0.5}, # target=2 more weight
            {0:0.25,1:0.5,2:0.25}, # target=1 more weight
            {0:0.2,1:0.4,2:0.4},   # target=1&2 more weight
        ],
    }
    rf = RandomForestClassifier()
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring=f1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Best score:", gs.best_score_)

    model = RandomForestClassifier(**gs.best_params_)

    model.fit(X_train, np.ravel(y_train))
    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc   = accuracy_score(y_val, val_pred)
    train_f1  = f1_score(y_train, train_pred, average='macro')
    val_f1    = f1_score(y_val, val_pred, average='macro')

    print("F1  (TRAIN | VAL): ", round(train_f1, 4), "|", round(val_f1, 4))
    print("ACC (TRAIN | VAL): ", round(train_acc, 4), "|", round(val_acc, 4))

    return model

def main():
    df, df_test = load_data()
    df, df_test, feat_cols, target = change_column_names(df, df_test)

    model = train_model(df, feat_cols, target)

    # Create submission
    test_pred = model.predict(df_test)
    test_pred = pd.DataFrame(test_pred, columns=['final_status'])
    test_pred.to_csv("predictions.csv", header=True, index=False)
    test_pred.to_json("predictions.json", orient='columns')

    return 0

if __name__ == '__main__':
    sys.exit(main())