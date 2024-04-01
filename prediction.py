
# %% [markdown]
# <a id="1"></a> <br>
# # Step 1: Load and Explore the Data
# First, we import necessary libraries and load the datasets:

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# %%
train_df.head()

# %%
test_df.head()

# %% [markdown]
# <a id="2"></a> <br>
# # Step 2: Visualizations 
# ## 1. Distribution of Sale Prices
# > Understanding how the sale prices are distributed can help identify skewness and outliers.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Distribution of Sale Prices
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# ## 2. Correlation Heatmap
# > A heatmap of correlations between numerical features and the sale price to identify which variables are most related to the price.

# %%
# Correlation matrix of numerical features
corr_matrix = train_df.select_dtypes(include=['int64', 'float64']).corr()

# Heatmap of correlations
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# %% [markdown]
# ## 3. Boxplot of Sale Prices by Overall Quality
# > This can show the relationship between the overall quality of a house and its sale price.

# %%
# Boxplot of Sale Prices by Overall Quality
plt.figure(figsize=(12, 8))
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df)
plt.title('Sale Price Distribution by Overall Quality')
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price')
plt.show()


# %% [markdown]
# <a id="3"></a> <br>
# # Step 3: Preprocess the Data
# > Handle missing values, encode categorical variables, and scale numerical features:

# %%
# Separate target variable and predictors
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = np.log(train_df['SalePrice'])  # Transform target variable with logarithm
X_test = test_df.drop(['Id'], axis=1)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])


# %%
X_train.columns

# %% [markdown]
# <a id="4"></a> <br>
# # Step 4: Define the Model and Bundle Preprocessing and Modeling Code in a Pipeline

# %% [markdown]
# > We use a RandomForestRegressor for this task:

# %%
model = RandomForestRegressor(n_estimators=100, random_state=0)


# %% [markdown]
# > This simplifies fitting and predictions:

# %%
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Train the model
my_pipeline.fit(X_train, y_train)


# %% [markdown]
# <a id="5"></a> <br>
# # Step 5: Predict and Prepare Submission
# > Predict using the test dataset and prepare the dataframe:

# %%
# Predictions in log scale
predictions_log_scale = my_pipeline.predict(X_test)

# Convert predictions back from log scale
predictions = np.exp(predictions_log_scale)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})

predictions_df.head()

# %%
#save pickle file
import pickle
# save the model to disk
filename = 'model.pkl'
pickle.dump(my_pipeline, open(filename, 'wb'))



