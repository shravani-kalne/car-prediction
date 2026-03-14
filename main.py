import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("used_cars.csv")

# Handle missing categorical values
cat_cols = ['fuel_type','accident','clean_title']
imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer.fit_transform(df[cat_cols])

# Clean mileage column
df['milage'] = df['milage'].str.replace(',', '', regex=True)
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False)
df['milage'] = df['milage'].astype(float)

# Handle missing numeric values
num_imputer = SimpleImputer(strategy='mean')
df[['milage']] = num_imputer.fit_transform(df[['milage']])

# Clean price column
df['price'] = df['price'].str.replace('$','',regex=False)
df['price'] = df['price'].str.replace(',','',regex=False)
df['price'] = df['price'].astype(float)

# Features and target
X = df.drop(columns=['price'])
y = df['price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Encode categorical variables
categorical_cols = X_train.select_dtypes(include='object').columns

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
model = SVR(kernel='rbf')

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("RMSE:",rmse)
print("R2:",r2)

# Save model files
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))
pickle.dump(X_train.columns,open("columns.pkl","wb"))

print("Model saved successfully")