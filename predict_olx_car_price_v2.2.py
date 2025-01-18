import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("Car Price Prediction App")

# File Upload Section
uploaded_file = st.file_uploader("Upload Training Data (Excel file)", type=["xlsx", "xls"])

if uploaded_file:
    # Read the Excel file
    data = pd.read_excel(uploaded_file)

    # Adding Unnamed columns
    data['Age'] = 2024-data['Year']
    # Drop Unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Year')]

    # Show raw data
    st.subheader("Raw Data")
    st.write(data)

    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis")

    # Descriptive Statistics
    st.write("### Descriptive Statistics")
    st.write(data.describe())

    # Data Visualization
    st.write("### Data Visualizations")
    
    st.write("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['Price'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("Pairplot")
    fig = sns.pairplot(data)
    st.pyplot(fig)

    st.write("Correlation Heatmap")
    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        st.write("No numeric columns found for correlation heatmap.")
    else:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)    

    # Data Preprocessing
    st.subheader("Data Preprocessing")

    st.write("### Handling Missing Values")
    missing_values_count = data.isnull().sum()
    st.write(missing_values_count)

    numeric_features = data.select_dtypes(include=[np.number])
    numeric_features = numeric_features.drop('Price',axis=1).columns.tolist()
    categorical_features = data.select_dtypes(exclude=[np.number]).columns.tolist()

    # Imputation and Encoding Pipelines
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define Target and Features
    
    # Clean column names
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
    
    target = 'Price'
    features = [col for col in data.columns if col != target]

    X = data[features]
    y = data[target]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training and Evaluation
    st.subheader("Model Training and Evaluation")

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        st.write(f"### {model_name}")
        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
        st.write("R-squared Score:", r2_score(y_test, predictions))

    # Single Input Prediction
    st.subheader("Single Input Prediction")

    st.write("Input Data:")

    inputs = {}
    for feature in features:
        if feature in numeric_features:
            inputs[feature] = st.number_input(f"Enter {feature}:", value=0.0)
        else:
            inputs[feature] = st.text_input(f"Enter {feature}:", value="")

    if st.button("Predict Price"):
        user_input = pd.DataFrame([inputs])
        st.write(user_input)        

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])
            pipeline.fit(X, y)  # Fit on all data for prediction
            prediction = pipeline.predict(user_input)
            st.write(f"Predicted Price ({model_name}):", prediction[0])