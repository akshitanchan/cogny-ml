import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# loading data
def load_data(data_dir):
    data = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):  
            for person_dir in os.listdir(class_path):
                person_path = os.path.join(class_path, person_dir)
                if os.path.isdir(person_path):  
                    person_data = pd.DataFrame()
                    for file in os.listdir(person_path):
                        if file.endswith('.csv') and file != '.DS_Store':
                            df = pd.read_csv(os.path.join(person_path, file))
                            person_data = pd.concat([person_data, df])

                    aggregated_data = person_data[['Theta', ' Low_beta', ' High_beta']].mean()  

                    data.append(aggregated_data.values)
                    labels.append(class_dir)  
    return np.array(data), labels

# managing oversampling
X, y = load_data('data/')
ros = RandomOverSampler(random_state=42)

# splitting data
X_resampled, y_resampled = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

# models
models = {
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# training models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} F1 Score: {f1}")
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"{name} Precision: {precision}")
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"{name} Recall: {recall}")

# cross validation
for name, model in models.items():
    skf = StratifiedKFold(n_splits=4)
    scores = cross_val_score(model, X, y, cv=skf)
    print(f"{name} Cross validation scores: {scores}")
    print(f"{name} Cross validation mean score: {scores.mean()}")

# classification report
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} classification report:")
    print(classification_report(y_test, y_pred))

# predictions
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Predictions: {y_pred}")