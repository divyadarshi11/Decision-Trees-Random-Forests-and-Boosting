#Divya Darshi
#1002090905

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from Classification import DecisionTree, RandomForest, AdaBoost

# Load the Titanic dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

original_train = train_data.copy() 

full_data = [train_data, test_data]

train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] 

# Data Preprocessing
#columns_to_drop = ['Name', 'Ticket', 'Cabin']
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Sex']
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)

# Fill missing values
for dataset in [train_data, test_data]:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# Convert categorical features to numerical
label_encoder = LabelEncoder()
for dataset in [train_data, test_data]:
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['Embarked'] = label_encoder.fit_transform(dataset['Embarked'])

# Feature Selection
X = train_data.drop(['PassengerId', 'Survived'], axis=1)
y = train_data['Survived']

# Split the Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=0)

# Convert X, y to numpy arrays
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()

# Decision Tree Classifier
dt_classifier = DecisionTree()
dt_classifier.fit(X_train, y_train)
dt_predictions_val = dt_classifier.predict(X_val)
dt_val_accuracy = accuracy_score(y_val, dt_predictions_val)

# Random Forest Classifier
rf_classifier = RandomForest(num_learners=50, random_state=0)
rf_classifier.fit(X_train, y_train)
rf_predictions_val = rf_classifier.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, rf_predictions_val)

# AdaBoost Classifier with Decision Tree as the weak learner
adaboost_classifier = AdaBoost(weak_learner=DecisionTree(max_depth=2), num_learners=50, learning_rate=0.1, random_state=0)
adaboost_classifier.fit(X_train, y_train)
adaboost_predictions_val = adaboost_classifier.predict(X_val)
adaboost_val_accuracy = accuracy_score(y_val, adaboost_predictions_val)

# Print accuracies
results_df = pd.DataFrame({
    'Classifier': ['Decision Tree', 'Random Forest', 'AdaBoost'],
    'Validation Accuracy': [dt_val_accuracy, rf_val_accuracy, adaboost_val_accuracy]
})

# Print results in tabular form
print(results_df)
print("")

# Preprocess test data
test_data = test_data.astype(float)  # Ensure all features are of numeric data types
X_test = test_data.drop('PassengerId', axis=1).to_numpy()  # Convert X_test to numpy array

# Predictions for the test data
dt_predictions_test = dt_classifier.predict(X_test)
rf_predictions_test = rf_classifier.predict(X_test)
adaboost_predictions_test = adaboost_classifier.predict(X_test)

# Prepare the submission DataFrame
prediction_data = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Age': test_data['Age'],
    'Sex': test_data['Sex'].map({0: 'male', 1: 'female'}),
    'Survived_DT': dt_predictions_test,
    'Survived_RF': rf_predictions_test,
    'Survived_AdaBoost': adaboost_predictions_test
})

# Save the prediction to CSV file
prediction_data.to_csv('prediction.csv', index=False)
print("File prediction.csv saved.")
