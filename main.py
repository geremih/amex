import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt


def clean_helper(df):
    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey',  'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos'] :
        df[column] = df[column].fillna(0)

    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']:
        df[column] = df[column].str.rstrip().str.replace(',','').str.replace('$','').astype(int)
    df['p_centaur'] = (df['previous_vote'] == 'CENTAUR').astype(int)
    df['p_ebony'] = (df['previous_vote'] ==  'EBONY').astype(int)
    df['p_tokugawa'] = (df['previous_vote'] == 'TOKUGAWA').astype(int)
    df['p_cosmos'] = (df['previous_vote'] == 'COSMOS').astype(int)
    df['p_odyssey'] = (df['previous_vote'] == 'ODYSSEY').astype(int)
    return df

def clean_training(df):

    df.columns = ['Citizen_ID', 'actual_vote', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']
    df = clean_helper(df)
    df['actual_vote'] = df['actual_vote'].astype('category')
    return df

    
def clean_testing (df):
    df.columns = ['Citizen_ID', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']
    return clean_helper(df)


def get_features(df):
    return  df[['p_centaur', 'p_ebony', 'p_tokugawa', 'p_odyssey', 'p_cosmos','d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey',  'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos']]


def testing(model):
    tf = pd.read_csv('testing.csv')
    tf = clean_testing(tf)
    X = get_features(tf)
    t = model.predict(X)
    ids = tf['Citizen_ID']
    t = [categories[i] for i in t]
    pd.DataFrame({'y': t},index=ids).to_csv('alpha_kappa_iit_kgp_2.csv')

def metric(X,y):
    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    print 'predicted = %s' %(predicted)

    # generate evaluation metrics
    print 'metrics.accuracy_score(y_test, predicted) = %s' %(metrics.accuracy_score(y_test, predicted))


    #evaluate the model using 10-fold cross-validation
    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print 'cross_val_score = %s' %(scores)    



df = pd.read_csv('training.csv')
df = clean_training(df)
categories = df['actual_vote'].cat.categories
df['actual_vote'] = df['actual_vote'].cat.rename_categories(np.arange(5))


y = np.ravel(df['actual_vote'])


X = get_features(df)



print 'X.shape = %s  y.shape = %s' %(X.shape,y.shape)
model = LogisticRegression()
model = model.fit(X, y)
print model.score(X, y)



metric(X,y)
#testing(model)
