# all imports required
import numpy as np
from collections import Counter
import pandas as pd
import random

# defining the k nearest neighbors classifier algorithm
# we pass the training data and the test data(predict) as list
# we pass k number of neighbors with 3 neighbors as default
def k_neighbors_classifier(trainingData, predict, k=3):
    
    # declaring a list of euclidean distances
    distances = []
        
    # calculating the euclidean distances and storing it in the list 'distances'
    for group in trainingData:
        for features in trainingData[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            
            # calculated distances are stored in the list along with their respective groups
            distances.append([euclidean_distance, group])
             
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
  
    return vote_result, confidence

# url of the source file
url = 'https://raw.githubusercontent.com/Customer-Churn-Prediction/Project-Machine-Learning/master/Churning.csv'

# reading the source file and storing it in a pandas DataFrame
df = pd.read_csv(url)

# the 'Churn' column is not the last column in the original dataset
# appending the 'Churn' data i.e the target containing the two groups 0 and 1 as the last column
# storing the 'Churn' in another DataFrame variable
target = df['Churn']

# deleting unnecessary columns and the target column from the DataFrame
df.drop(['Churn', 'Phone', 'State','Area Code'], axis = 1, inplace=True)

# now we append the target data as the last column 
df['Churn'] = target

# storing the datas present in the DataFrame as list
full_data = df.astype(float).values.tolist()

# shuffling the data
random.shuffle(full_data)

# setting the percentage of test data as 30%
test_size = 0.3

# train_set and test_set are two dictionaries and their keys represent the two groups or classes
train_set, test_set = {0:[], 1:[]}, {0:[], 1:[]}

# splitting the data into train and test sets
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# calculating the accuracy
for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_neighbors_classifier(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)