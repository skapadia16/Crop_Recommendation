#🌱 Agriculture Crop Recommendation Using Machine Learning

A Machine Learning project that recommends the most suitable crop based on soil and environmental conditions such as Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall.

The project uses a Random Forest Classifier to predict the recommended crop.

#📌 Project Overview

Choosing the right crop is an important decision in agriculture. Different crops require different soil nutrients and environmental conditions.

This project uses historical agricultural data to train a Machine Learning model that can recommend a suitable crop based on given agricultural parameters.

Input Features

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature

Humidity

Soil pH

Rainfall

Output

The model predicts the most suitable crop, for example:

Recommended Crop: Rice

#🎯 Objectives

Analyze agricultural data.

Perform data preprocessing and feature analysis.

Identify the relationship between soil/environmental conditions and crops.

Train a Random Forest classification model.

Evaluate model performance.

Predict the suitable crop for new input data.

#🛠️ Technologies Used

Python

Pandas — Data manipulation

NumPy — Numerical operations

Matplotlib — Data visualization

Seaborn — Statistical visualization

Scikit-learn — Machine Learning

Jupyter Notebook — Development and experimentation

Stremlit

#📂 Project Structure
Agriculture-Crop-Recommendation/
│
├── data/
│   └── crop_recommendation.csv
│
├── notebooks/
│   └── crop_recommendation.ipynb
│
├── README.md
│
└── requirements.txt

#📊 Dataset

The dataset contains agricultural and environmental parameters along with the corresponding crop label.

Feature	Description
N	Nitrogen content in soil
P	Phosphorus content in soil
K	Potassium content in soil
temperature	Temperature
humidity	Humidity
ph	Soil pH
rainfall	Rainfall
label	Recommended crop
#🔍 Data Preprocessing

The following steps were performed before model training:

Loaded the CSV dataset using Pandas.

Inspected the dataset structure.

Checked for missing values.

Checked feature data types.

Separated input features and target variable.

Split the dataset into training and testing sets.

Example:

import pandas as pd

df = pd.read_csv("crop_recommendation.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

#🧹 Feature and Target Separation

The label column is used as the target variable.

X = df.drop("label", axis=1)
y = df["label"]


Where:

X → Input features

y → Target crop

✂️ Train-Test Split

The dataset is divided into training and testing data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#🌳 Machine Learning Model
Random Forest Classifier

A Random Forest Classifier is used for crop recommendation.

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

#📈 Model Evaluation

Predictions are generated using the test dataset:

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

Evaluation Metrics

The model can be evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Add your actual model accuracy here after training the final model.

Model Accuracy: XX.XX%

#🌳 Decision Tree Visualization

Since Random Forest consists of multiple Decision Trees, one individual tree can be visualized:

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    max_depth=3
)

plt.show()


This visualization helps understand how an individual tree makes classification decisions.

#🔮 Crop Prediction

The trained model can predict a crop from new agricultural conditions.

Example:

new_data = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]

prediction = model.predict(new_data)

print("Recommended Crop:", prediction[0])


Example output:

Recommended Crop: rice

💡 Future Improvements

The project can be extended with:

🌦️ Real-time weather data

📍 Location-based crop recommendations

💧 Smart irrigation prediction

🧪 Fertilizer recommendation

🌿 Plant disease detection using Computer Vision

📱 Web or mobile application

🤖 AI-powered farming assistant

☁️ Cloud deployment

🚀 Installation

Clone the repository:

git clone https://github.com/your-username/Agriculture-Crop-Recommendation.git


Navigate to the project:

cd Agriculture-Crop-Recommendation


Install the required libraries:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook

#📦 Requirements

Create a requirements.txt file:

pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter


Then install:

pip install -r requirements.txt

#📌 Key Learning Outcomes

Through this project, I learned:

Data loading and exploration using Pandas

Handling and checking missing values

Feature and target separation

Train-test splitting

Classification using Random Forest

Model evaluation

Data visualization

Decision Tree visualization

Making predictions with a trained ML model

#👨‍💻 Author

Shreyas Kapadiya

⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
