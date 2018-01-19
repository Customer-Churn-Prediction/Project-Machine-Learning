# all imports required
import numpy as np
from collections import Counter
import pandas as pd
import random

# defining the k nearest neighbors classifier algorithm
# we pass the training data and the test data as list
# we pass k number of neighbors with 3 neighbors as default
def k_neighbors_classifier(train_set, test_set, k=3):
    
    # declaring a list for calculating and storing euclidean distances
    distances = []
    
    # calculating the euclidean distances and storing it in the list 'distances'
    for group in train_set:
        for features in train_set[group]:
            euclidean_dist = np.linalg.norm(np.array(features)-np.array(test_set))
            
            # calculated distances are stored in the list along with their respective groups
            distances.append([euclidean_dist, group])
    
    # sorting the 'distances' and storing the names of the k nearest neighboring groups in a list
    neighbors = [i[1] for i in sorted(distances)[:k]]
    
    # prediction of the most common neighbor and finding the probability of the prediction
    # the most_common() method stores the data in a list of a tuple, 
    # in this case we could write the tuple of list as: [(group_name, frequency)]
    # therefore we choose the first element as the predicted group
    prediction = Counter(neighbors).most_common(1)[0][0]
    
    # we know k is the total no. of neighbors
    # so we find the probability of our predicted group by division: (frequency / k)
    probability = Counter(neighbors).most_common(1)[0][1] / k
  
    return prediction, probability

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
data_list = df.astype(float).values.tolist()

# shuffling the data so that the training and testing data can be chosen at random
random.shuffle(data_list)

# setting the percentage of test data as 30%
test_size = 0.3

# train_set and test_set are two dictionaries and their keys represent the two groups or classes
# here it is 0(churn = False) or 1(churn = True)
train_set, test_set = {0:[], 1:[]}, {0:[], 1:[]}

# splitting the data into train and test sets
# selecting from the beginning upto the last 30% of data i.e. the first 70%
train_data = data_list[:-int(test_size*len(data_list))]

# selecting the last 30% of the data
test_data = data_list[-int(test_size*len(data_list)):]
# storing the grouping the data according to their group or class 0 or 1 and 
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
confusion_matrix = [[0, 0], [0, 0]]

# calculating the accuracy
for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_neighbors_classifier(train_set, data, k=5)
        if group == vote:
            correct += 1
            confusion_matrix[group][group] += 1
        else:
            confusion_matrix[group][vote] += 1
        total += 1
        
confusion_df = pd.DataFrame(data=confusion_matrix, 
                            columns=['Predicted False', 'Predicted True'],
                            index = ['Actual False', 'Actual True'])
print('Total:', total)
print('Correct:', correct)
print('Confusion Matrix:\n', confusion_df)
print('Accuracy:', correct/total)

'''
Output:
Total: 999
Correct: 871
Confusion Matrix:               
              Predicted False  Predicted True
Actual False              835              16
Actual True               112              36
Accuracy: 0.8718718718718719
'''