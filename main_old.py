import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# 1. Load the data
housing=pd.read_csv("Housing.csv")

#2. Create a Stratified test set
housing['income_cat'] = pd.cut(housing["median_income"],
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index].drop("income_cat", axis=1)  #We will work on this data
    strat_test_set = housing.iloc[test_index].drop("income_cat", axis=1)  #Set aside the test data

# We will work on the copy of training data
housing=strat_train_set.copy()

# 3. Seperate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
print(housing, housing_labels)

# 4.Seperate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Lets make pipeline 
# for the numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# for the categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Construct the full Pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# 6. Tranform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)


# 7. Train the model

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f"The root mean squared error for Linear Regression is {lin_rmse}")
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(lin_rmses).describe())


#Decision Tree Model
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds=dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"The root mean squared error for Decision Tree Model is {tree_rmse}")
# print(pd.Series(dec_rmses).describe())

#Random Forest
ran_reg=RandomForestRegressor()
ran_reg.fit(housing_prepared, housing_labels)
ran_preds=ran_reg.predict(housing_prepared)
# ran_rmse = root_mean_squared_error(housing_labels, ran_preds)
# print(f"The root mean squared error for Random Forest is {ran_rmse}")
ran_rmses = -cross_val_score(ran_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(ran_rmses).describe())
r2 = r2_score(housing_labels, ran_reg.predict(housing_prepared))
print(f"R² score (Random Forest): {r2:.4f}")
print(f"Accuracy-like percentage (Random Forest): {r2 * 100:.2f}%")

