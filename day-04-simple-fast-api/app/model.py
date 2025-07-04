import joblib
import numpy as np

model = joblib.load("/Users/ariopurba37/Documents/dream/coding/50-days-of-code/day-04-simple-fast-api/model/iris_model.joblib")
target_names = ["setosa", "versicolor", "virginica"]

def predict_species(features):
    data = np.array([[features.sepal_length, features.sepal_width,
                      features.petal_length, features.petal_width]])
    prediction = model.predict(data)[0]
    return target_names[prediction]
