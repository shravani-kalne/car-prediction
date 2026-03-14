readme_content = """
Used Vehicle Market Price Prediction using SVM

This project predicts the price of a used vehicle using machine learning.

The dataset used in this project is used_cars.csv which contains vehicle
specifications such as brand, model, model year, mileage, fuel type,
engine type, transmission type, exterior color, interior color,
accident history and clean title information.

The dataset is first preprocessed by handling missing values and
converting text data into numeric format.

Categorical features are converted into numerical values using
one-hot encoding.

The dataset is split into training data and testing data.

The machine learning algorithm used in this project is
Support Vector Machine (SVM).

The model is trained using the training dataset and evaluated
using performance metrics such as MAE, RMSE and R2 score.

After training, the model is saved as model.pkl and the scaler
is saved as scaler.pkl.

A Streamlit web application is created to allow users to enter
vehicle details and predict the estimated market price of the vehicle.

Steps to run the project

1 Install required libraries
pip install pandas numpy scikit-learn streamlit

2 Train the model
python main.py

3 Run the Streamlit application
python -m streamlit run app.py
"""

with open("README.md", "w") as file:
    file.write(readme_content)

print("README file created successfully")