import pandas
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Drop / fill some null values & do some feature engineering
#
def cleanTestDataset(dataset):
    # Drop unnecessary columns
    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
    dataset.dropna(subset=['Age'], inplace=True)

    # Convert sex to isMale, with 0 for no and 1 for yes
    dataset['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

    # S is the most embarked from location, use this to fill the unknown onces.
    dataset['Embarked'].fillna('S', inplace=True)

    # Convert the Embarked letter to integers
    dataset['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    # Create new column family size
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

    # Finally drop the SibSp and Parch columns.
    dataset.drop(['SibSp', 'Parch'], inplace=True, axis=1)


# For the Kaggle test dataset we cannot drop empty rows (as we should have a classification for every passenger)
# Therefore there are 2 clean dataset methods
#
def cleanRealDataset(dataset):
    # Drop unecessary columns
    dataset.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

    # Convert sex to isMale, with 0 for no and 1 for yes
    dataset['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

    # S is the most embarked from location, use this to fill the unknown onces.
    dataset['Embarked'].fillna('S', inplace=True)

    # Fill missing age to the median age.
    dataset['Age'].fillna(30, inplace=True)

    # There is 1 missing fare price, set it to the mean fare price
    real_dataset['Fare'].fillna(40, inplace=True)

    # Convert the Embarked letter to integers
    dataset['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    # Create new column family size
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

    # Finally drop the SibSp and Parch columns.
    return dataset.drop(['PassengerId', 'SibSp', 'Parch'], inplace=False, axis=1)


# Given a model and a test dataset returns an float >= 0 <= 100 that indicates the percentage of correct classifications
#
def getModelAccuracy(model, test_dataset):
    predictions = model.predict(test_dataset)

    wrong = index = 0
    for i, row in test.iterrows():
        if (predictions[index] != row['Survived']):
            wrong += 1
        index += 1

    return (1 - float(wrong) / float(index)) * 100



dataset = pandas.read_csv('./train.csv')
cleanTestDataset(dataset)

train, test = train_test_split(dataset, test_size=0.2)

Y_train = train['Survived']
X_train = train.drop(['Survived'], inplace=False, axis=1)

Y_test = test['Survived']
X_test = test.drop(['Survived'], inplace=False, axis=1)

# Create some models
models = []
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('KNeighborsClassifier', KNeighborsClassifier(3)))
models.append(('SVC', SVC(kernel="rbf", C=0.025, probability=True)))
models.append(('NuSVC', NuSVC(probability=True)))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))

# Print accuracy on the train / test data for each model
for name, model in models:
    model.fit(X_train, Y_train)
    accuracy = getModelAccuracy(model, X_test)

    print("%s accuracy: %d%%" % (name, accuracy))
print("")

# This one performed best :)
model = GradientBoostingClassifier()
model.fit(X_train, Y_train)

# Time to start for real with the kaggle test data.
real_dataset = pandas.read_csv('./test.csv')
cleaned_real_dataset = cleanRealDataset(real_dataset)

# The  magic stuff here.
predictions = model.predict(cleaned_real_dataset)

# Write the predictions to the output csv file (as is required by Kaggle)
with open('output.csv', 'w') as output:
    w = csv.writer(output)
    w.writerow(['PassengerId', 'Survived'])

    index = 0
    for i, row in real_dataset.iterrows():
        #print ("PassengerId: %d = %r" % (row['PassengerId'], predictions[index]))
        w.writerow([int(row['PassengerId']), predictions[index]])
        index += 1

print("Predictions have been written to output.csv")
