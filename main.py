import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn import svm
from  sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier as Forest
import xgboost as xgb

def clean_helper(df):
    for column in ['p_voted','n_unique_p','d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey',  'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs'] :
        df[column] = df[column].fillna(0)

    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']:
        df[column] = df[column].str.rstrip().str.replace(',','').str.replace('$','').astype(int)
    df['p_centaur'] = (df['previous_vote'] == 'CENTAUR').astype(int)
    df['p_ebony'] = (df['previous_vote'] ==  'EBONY').astype(int)
    df['p_tokugawa'] = (df['previous_vote'] == 'TOKUGAWA').astype(int)
    df['p_cosmos'] = (df['previous_vote'] == 'COSMOS').astype(int)
    df['p_odyssey'] = (df['previous_vote'] == 'ODYSSEY').astype(int)
    del df['previous_vote']
    
    dsum = df['d_centaur']+df['d_ebony']+ df['d_tokugawa']+ df['d_odyssey']+ df['d_cosmos']
    df['d_centaur'] = df['d_centaur'].div(dsum, fill_value=0 )
    df['d_ebony'] = df['d_ebony'].div(dsum, fill_value=0 )
    df['d_tokugawa'] = df['d_tokugawa'].div(dsum, fill_value=0 )
    df['d_cosmos'] = df['d_cosmos'].div(dsum, fill_value=0 )
    df['d_odyssey'] = df['d_odyssey'].div(dsum, fill_value=0 )

    ssum = df['s_centaur']+df['s_ebony']+ df['s_tokugawa']+ df['s_odyssey']+ df['s_cosmos']
    df['s_centaur'] = df['s_centaur'].div(ssum, fill_value=0 )
    df['s_ebony'] = df['s_ebony'].div(ssum, fill_value=0 )
    df['s_tokugawa'] = df['s_tokugawa'].div(ssum, fill_value=0 )
    df['s_cosmos'] = df['s_cosmos'].div(ssum, fill_value=0 )
    df['s_odyssey'] = df['s_odyssey'].div(ssum, fill_value=0 )


    df['n_centaur'] = df['n_centaur'].div(df['n_rallies'], fill_value=0 )
    df['n_ebony'] = df['n_ebony'].div(df['n_rallies'], fill_value=0 )
    df['n_tokugawa'] = df['n_tokugawa'].div(df['n_rallies'], fill_value=0 )
    df['n_cosmos'] = df['n_cosmos'].div(df['n_rallies'], fill_value=0 )





















    df['n_odyssey'] = df['n_odyssey'].div(df['n_rallies'], fill_value=0 )


    df['region'] = df['region'].fillna(57).astype(int)

    df['income'] = df['income'].fillna(df['income'].mean())
    df['income'] = df['income']/ df['income'].max()
    ages = pd.unique(df['age'].ravel())
    for i,a in enumerate(ages):
        df['a' +str(i)] = (df['age'] == a).astype(int)

    edus = pd.unique(df['education'].ravel())
    
    for i,a in enumerate(edus):
        df['e' +str(i)] = (df['education'] == a).astype(int)
        
    df['region'] = df['region'].fillna(57)
    for i in range(1,58):
        df['r' + str(i)] = (df['region'] == i).astype(int)
    #we are not making unknown region a feature
    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey',  'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos'] :
        df[column] = df[column].fillna(0)
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
    regions = ['r' + str(i) for i in range(1,58)]
    ages = ['a' + str(i) for i in range(0,5)]
    edus = ['e' + str(i) for i in range(0,4)]
    return  df[['p_centaur', 'p_ebony', 'p_tokugawa', 'p_odyssey', 'p_cosmos','d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey',  'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies','n_cosmos','n_unique_p','income', 'r_years','n_unique_p', 'p_voted', 'politics'] + regions + ages + edus]



def get_model():
#    return LogisticRegression()
#    return svm.SVC()
#    return GradientBoostingClassifier(n_estimators=200, max_depth=4)
    return xgb.XGBClassifier(max_depth = 4, n_estimators=200)


def testing(model):
    tf = pd.read_csv('testing.csv')
    tf = clean_testing(tf)
    X = get_features(tf)
    X = fix_data(X)
    t = model.predict(X)
    ids = tf['Citizen_ID']
    t = [categories[i] for i in t]
    pd.DataFrame({'y': t},index=ids).to_csv('alpha_kappa_iit_kgp_3.csv')

def metric(X,y):
    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = get_model()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    print 'predicted = %s' %(predicted)

    # generate evaluation metrics
    print 'metrics.accuracy_score(y_test, predicted) = %s' %(metrics.accuracy_score(y_test, predicted))


    #evaluate the model using 10-fold cross-validation
    scores = cross_val_score(get_model(), X, y, scoring='accuracy', cv=5)
    print 'cross_val_score = %s' %(scores)
    print 'cross val average ', np.mean(scores)

def fix_data(X):
    #duplicate n_unique_p
    temp = X['n_unique_p']
    del X['n_unique_p']
    X['n_unique_p'] = temp.ix[:,0]
    
    #making columns strictly alphanumeric
    oldcols = X.columns
    newcols = []
    for col in oldcols:
        newcols.append(col.replace('_',''))
    X.columns = newcols
    return X

df = pd.read_csv('training.csv')
df = clean_training(df)
categories = df['actual_vote'].cat.categories
df['actual_vote'] = df['actual_vote'].cat.rename_categories(np.arange(5))


y = df['actual_vote']


X = get_features(df)
print X.columns


print 'X.shape = %s  y.shape = %s' %(X.shape,y.shape)
#model = LogisticRegression()
model = get_model()

X = fix_data(X)

model = model.fit(X, y)
print model.score(X, y)

#metric(X,y)
testing(model)
