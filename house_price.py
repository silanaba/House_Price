######### Import Library #########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



####### Import DataSet  ###########
### Data Link :  https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
df = pd.read_csv("kc_house_data.csv")



##########  EDA ###############

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Columns #####################")
    print(dataframe.columns)
    print("##################### Info #####################")
    print(dataframe.info())


check_df(df)



######## Drop ID  And DATE  ############
df.drop('id', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)

df.head()
df.columns

#######  Astype ############

df['bathrooms'] = df['bathrooms'].astype(int)
df['floors'] = df['floors'].astype(int)

df.info()


#### Data Visualizing  ####

sns.scatterplot(x=df["bathrooms"], y=df["price"])
plt.show()


sns.scatterplot(x=df["bedrooms"], y=df["price"])
plt.show()

df["bedrooms"].max()
df.drop(df[df['bedrooms'] > 30].index, inplace=True)


sns.barplot(x=df["waterfront"], y=df["price"])
plt.show()

sns.scatterplot(x=df["sqft_living"], y=df["price"])
plt.show()

sns.pairplot(df, hue="price")
plt.show()


###########  Corr ##############
corr = df.corr()
corr
sns.heatmap(corr, cmap="RdBu")
plt.show()



############# Model ####################

X = df.drop("price",axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



models = {
    "Linear Regression" : LinearRegression(),
    "Lasso" : Lasso(),
    "Ridge" : Ridge(),
    "K Neighbors Regressor" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor" : RandomForestRegressor(),
    "Adaboost Regressor" : AdaBoostRegressor(),
    "Gradient Boost Regressor" : GradientBoostingRegressor(),
    "XGBoost Regressor" : XGBRegressor()
}


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    print("Model performance for Training Set")
    print("Root Mean Squared Error: ", model_train_rmse)
    print("Mean Absolute Error: ", model_train_mae)
    print("R2 Score: ", model_train_r2)

    print("-----------------------------------")

    print("Model performance for Test Set")
    print("Root Mean Squared Error: ", model_test_rmse)
    print("Mean Absolute Error: ", model_test_mae)
    print("R2 Score: ", model_test_r2)

    print("-----------------------------------")
    print("\n")



model = XGBRegressor()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
model_train_mae, model_test_mae, model_test_r2 = evaluate_model(y_train, y_train_pred)

param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 9],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid = GridSearchCV(XGBRegressor(), param_grid, refit = True,verbose = 3,n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_params_


#{'colsample_bytree': 0.8,
# 'learning_rate': 0.1,
# 'max_depth': 6,
# 'n_estimators': 200,
# 'subsample': 0.8}


y_pred_grid=grid.predict(X_test)
mae=mean_absolute_error(y_test,y_pred_grid)
score=r2_score(y_test,y_pred_grid)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred_grid)
plt.show()



best_model = grid.best_estimator_


#### Save Model

import joblib

joblib.dump(best_model, 'model.pkl')
print("Save Model")
