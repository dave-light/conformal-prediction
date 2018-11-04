
# Assignment 1, implementation of KNN


```python
import numpy as np
import math
import time

```

## Part 1
Loading data into Python


```python
#load iris data
from sklearn.datasets import load_iris
iris = load_iris()
```


```python
#now importing ionosphere data

#34 features so usecols=np.arrange(34) take first 34 as column
X = np.genfromtxt("ionosphere.txt", delimiter=",",
usecols=np.arange(34))

#each label is of type int, features are not. Only labels will be generated
y = np.genfromtxt("ionosphere.txt", delimiter=",",
usecols=34, dtype='int')
```

## Part 2
Splitting the data into training and test sets


```python
# split iris
from sklearn.model_selection import train_test_split
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(iris['data'],
iris['target'], random_state=0)
```


```python
# split ionosphere
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
y, random_state=0)
```


```python
'''def knn_fit(features, labels):

    columns = features.shape[1] + 1
    length = len(features)
    labelled_samples = np.zeros((length,columns))
    for x in range(length):
        labelled_samples[x] = np.concatenate([features[x], [labels[x]]])
    return labelled_samples
    
#knn_fit(X_train_iris, y_train_iris)
'''

def knn_fit(features, labels):

    columns = features.shape[1] + 1
    length = len(features)
    labelled_samples = np.zeros((length,columns))
    for x in range(length):
        labelled_samples[x] = np.concatenate([features[x], [labels[x]]])
    return labelled_samples
    
#knn_fit(X_train_iris, y_train_iris)

```

# Part 3
Implementing the Nearest Neighbour method


```python
''' Reasoning; the average difference between all 34 features in ionoshpere 
    provides x co-ordinate (4 features for iris).
    The euclidean distance is calculated using x value and y value.
'''


"""takes two vectors sample[x,y] and neighbour[x,y]
   takes one scalar 'l' of type int, limiting number of attributes i.e. not including labels
"""
#calculates euclidean distance
def distance(x1, x2, num_of_features):
    distance = 0
    for i in range(num_of_features):
        distance += ((x1[i] - x2[i])**2)
    return math.sqrt(distance)
    
```


```python
#finds distance from nearest neighbour 

def calc_nearest(distance_list):
    current_minimum = math.inf
    length = len(distance_list)
    index = math.inf
    for x in range(length):
        if current_minimum > distance_list[x]:
            current_minimum = distance_list[x]
            index = x
    return index #returns index of min distance
```


```python
#predictions

def predict(test_sample, train_set, num_of_features):
    distance_list = []
    length = len(train_set)
    for x in range(length):
        #calls distance() and appends to distance_list
        distance_list.append(distance(test_sample, train_set[x], num_of_features))
    nearest_neighbour = calc_nearest(distance_list) #index of neareset neighbour in training set 
    predicted_label = train_set[nearest_neighbour][-1]
    return predicted_label
```


```python
#ERROR RATE
def test_error_rate(predictions, actual_labels):
    return np.mean(predictions != actual_labels)
```

## Conformity Measure


```python
def divide(numerator, denominator):

    try:
        return numerator/denominator
    except ZeroDivisionError:
        return 0   
'''    if numerator == 0.000:
        return 0
    elif denominator == 0.000:
        return numerator/1
    else:
        return numerator/denominator
'''
```




    '    if numerator == 0.000:\n        return 0\n    elif denominator == 0.000:\n        return numerator/1\n    else:\n        return numerator/denominator\n'



## Calculate confirmity scores for entire training set
Calculate the scores for the whole training set and store them.


```python
# CONFORMITY SCORES
def conformity(neighbours_set, num_features):

    length = len(neighbours_set)
    conformity_scores = []
    for x in range(length):
        n = 0
        distance_list = []
        distance_to_diff = 1
        diff_flag = 0
        distance_to_same = 1
        same_flag = 0
        conf_score = 0
        for i in range(length):
            if x == i:
                n = i #store the index of the neighbour being operated on
            distance_list.append(distance(neighbours_set[x], neighbours_set[i], num_features))
        
        current_minimum = math.inf
        index = 0 #index of nearest neighbour
        for j in range(length):
            if current_minimum > distance_list[j]:
                if j != n: #ensure not storing distance to itself
                    current_minimum = distance_list[j]
                    index = j
            
        if neighbours_set[index][-1] == neighbours_set[x][-1]:
            distance_to_same = current_minimum
            same_flag = 1

        else:
            distance_to_diff = current_minimum
            diff_flag = 1

        
        if same_flag == 1:
            current_minimum = math.inf
            for j in range(length):
                if current_minimum > distance_list[j]:
                    if j != n and neighbours_set[j][-1] != neighbours_set[x][-1]: #ensure labels aren't the same
                        current_minimum = distance_list[j]
            distance_to_diff = current_minimum

        
        if diff_flag == 1:
            current_minimum = math.inf
            for j in range(length):
                if current_minimum > distance_list[j]:
                    if j != n and neighbours_set[j][-1] == neighbours_set[x][-1]: #ensure labels are the same
                        current_minimum = distance_list[j]            
            distance_to_same = current_minimum

        conf_score = divide(distance_to_diff, distance_to_same)
        conformity_scores.append(conf_score)

    return conformity_scores
```

## Calculate p-value
Re-compute conformity score for single test_sample\[label\], the nearest neighbour of the same, and the nearest neighbour of different. 

Rank *test_sample\[label\]*

Return *rank  /  n +1*


```python
def calculate_p_val(conformity_set, train_set, p_training_set):
    # call distance on p_training_set[-1] from p_training_set[i++ (nott including -1)]  and store in min_heap
    
    #shape of training_set arr[item][features, features, label, conformity_score]
    
    #CALCULATE NEW CONFORMITY SCORE FOR ADDED LABEL
        #if p_training_set[x][-1] == p_training_set[-1][-1]
    
    length = len(conformity_set) #length of original set. Ignores test_item 
    conformity_score_for_label = 0
    num_features = len(train_set[0])-1
    
    distance_list = []
    distance_to_diff = 1
    diff_flag = 0
    distance_to_same = 1
    same_flag = 0
    conf_score = 0
    for i in range(length):
        distance_list.append(distance(train_set[i], p_training_set[-1], num_features))

    current_minimum = math.inf
    index = 0 #index of nearest neighbour
    for j in range(length):
        if current_minimum > distance_list[j]:
            current_minimum = distance_list[j]
            index = j

    if train_set[index][-1] == p_training_set[-1][-1]:
        distance_to_same = current_minimum
        same_flag = 1

    else:
        distance_to_diff = current_minimum
        diff_flag = 1

    index_to_same = 0
    index_to_diff = 0
    if same_flag == 1:
        current_minimum = math.inf
        for j in range(length):
            if current_minimum > distance_list[j]:
                if train_set[j][-1] != p_training_set[-1][-1]: #ensure labels aren't the same
                    current_minimum = distance_list[j]
                    index_to_diff = j
        distance_to_diff = current_minimum


    if diff_flag == 1:
        current_minimum = math.inf
        for j in range(length):
            if current_minimum > distance_list[j]:
                if train_set[j][-1] == p_training_set[-1][-1]: #ensure labels are the same
                    current_minimum = distance_list[j]
                    index_to_same = j
        distance_to_same = current_minimum

    conf_score = divide(distance_to_diff, distance_to_same)
    #update conformity set
    
    rank = length+1
    for x in range(length):
        if conf_score <= conformity_set[x]:
            rank -= 1
    
    rank /= (length +1)
    #print('rank'+str(rank))
    return rank
```

## Calculate false p-value
Sum total p-values and subtract true p-value


```python
# calculate false p value for each sample

def p_val(conformity_set, train_set, test_sample, labels_set):
    average_p = 0
    
    l = len(labels_set)

    train_set_plus1 = len(train_set)+1
    
    false_p = 0
    true_p = 0
    current_p = -math.inf
    for i in range(len(labels_set)):
        p_training_set = np.empty(train_set.shape)
        test_sample[-1]=labels_set[i]
        p_training_set = np.concatenate([train_set, [test_sample]])
        new_p = calculate_p_val(conformity_set, train_set, p_training_set)
        false_p += new_p
        if current_p < new_p:
            current_p = new_p

    true_p = current_p
    average_p += ((false_p - true_p) / (l-1))
    return average_p #(per row of samples)
```

# MAIN

## IRIS


```python
start = time.time()
#constants
num_of_features = len(X_test_iris[0])
num_of_samples = len(y_test_iris)
actual_labels = y_test_iris
#train 
iris_train_set = knn_fit(X_train_iris, y_train_iris)
iris_test_set = knn_fit(X_test_iris, y_test_iris)

#predict
predictions = []
for x in range(num_of_samples):
    predictions.append(int(predict(iris_test_set[x], iris_train_set, num_of_features)))

#print(predict(iris_test_set[0], iris_train_set, num_of_features))


# SCORE
print(predictions)
print('test error rate: '+str(test_error_rate(predictions, actual_labels))+'\n')


# P-values
###################
## Calculate conformity scores for iris_train_set
conformity_set = conformity(iris_train_set, num_of_features)




labels_set = [0,1,2]
iris_test_set_labels = iris_test_set
false_p_value =0
temp_p_value = 0
# for all predictions, append p values for all labels

for x in range(len(predictions)):
#    p_values.append(predictions[x])
#    for i in range(len(predictions)):
    temp_p_value += p_val(conformity_set, iris_train_set, iris_test_set[x], labels_set)
false_p_value = (temp_p_value / len(predictions))

#print(temp_p_value)
print('average false p value '+str(false_p_value))
print(time.time() - start, ' seconds')

```

    [2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 2]
    test error rate: 0.02631578947368421
    
    average false p value 0.04471355379599442
    0.1244957447052002  seconds


## IONESPHERE


```python
start = time.time()
#constants
num_of_features = len(X_test[0])
num_of_samples = len(y_test)
actual_labels = y_test
#train 
iono_train_set = knn_fit(X_train, y_train)
iono_test_set = knn_fit(X_test, y_test)

#predict
predictions = []
for x in range(num_of_samples):
    predictions.append(int(predict(iono_test_set[x], iono_train_set, num_of_features)))

#print(predict(iris_test_set[0], iris_train_set, num_of_features))


# SCORE
print(predictions)
print('test error rate: '+str(test_error_rate(predictions, actual_labels))+'\n')


# P-values
###################
## Calculate conformity scores for iris_train_set
conformity_set = conformity(iono_train_set, num_of_features)

labels_set = [1,-1]
#iris_test_set_labels = iris_test_set
false_p_value =0
temp_p_value = 0
# for all predictions, append p values for all labels

for x in range(len(predictions)):
#    p_values.append(predictions[x])
#    for i in range(len(predictions)):
    temp_p_value += p_val(conformity_set, iono_train_set, iono_test_set[x], labels_set)
false_p_value = (temp_p_value / len(predictions))

#print(temp_p_value)
print('average false p value '+str(false_p_value))
print(time.time() - start, ' seconds')

```

    [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1]
    test error rate: 0.14772727272727273
    
    average false p value 0.041021005509641904
    2.934255838394165  seconds

