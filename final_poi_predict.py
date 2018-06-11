#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") )


### Task 2: Remove outliers
features = ["salary", "bonus"]
data_dict.pop("TOTAL")

# ~ data_dict.pop("LAY KENNETH L")
# ~ data_dict.pop("SKILLING JEFFREY K")
# ~ data_dict.pop("FREVERT MARK A")
# ~ data_dict.pop("LAVORATO JOHN J")


data = featureFormat(data_dict, features)
# ~ sal,bon=[],[]
# ~ for i in data:
	# ~ sal.append(i[0])
	# ~ bon.append(i[1])
# ~ sal=sorted(sal,reverse=True)
# ~ bon=sorted(bon,reverse=True)
# ~ sal=(sal[0:3])
# ~ bon=(bon[0:3])
# ~ for k,v in data_dict.items():
	# ~ if(data_dict[k]["salary"] in sal)or(data_dict[k]["bonus"] in bon):
		# ~ print(k)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    
    if all_messages == 'NaNNaN': # occurred when created additive features (all emails)
        all_messages = 'NaN'
    if poi_messages == 'NaNNaN':
        poi_messages = 'NaN'
    if all_messages == 'NaN':
        return 0
    if poi_messages == 'NaN':
        return 0
    if all_messages == 0:
        return 0
    return 1.*poi_messages/all_messages
    return fraction
    
for name in data_dict:
        poi_msg_to = data_dict[name]['from_poi_to_this_person']
        all_msg_to = data_dict[name]['to_messages']
        data_dict[name]['fraction_from_poi'] = computeFraction(poi_msg_to, all_msg_to)
        poi_msg_from = data_dict[name]['from_this_person_to_poi']
        all_msg_from = data_dict[name]['from_messages']
        data_dict[name]['fraction_to_poi'] = computeFraction(poi_msg_from, all_msg_from)
        poi_msg_all = poi_msg_to + poi_msg_from
        all_msg_all = all_msg_to + all_msg_from
        data_dict[name]['fraction_emails_with_poi'] = computeFraction(poi_msg_all, all_msg_all)


### Store to my_dataset for easy export below.
features_list = ["poi","salary","fraction_from_poi","fraction_emails_with_poi","fraction_to_poi", "shared_receipt_with_poi"]

my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier( n_estimators=12,min_samples_split=5)

clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
acc=accuracy_score(pred,labels_test)

print("RF ccuracy = ", acc)

#####
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)

for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
acc=accuracy_score(labels_test, pred)

print ("DT ccuracy = ", acc)

# function for calculation ratio of true positives
# out of all positives (true + false)
print ('precision = ', precision_score(labels_test,pred))

# function for calculation ratio of true positives
# out of true positives and false negatives
print ('recall = ', recall_score(labels_test,pred))


#####
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print ("NB ccuracy = ",accuracy)
print ('precision = ', precision_score(labels_test,pred))
print ('recall = ', recall_score(labels_test,pred))

from sklearn.metrics import classification_report
print(classification_report(pred,labels_test))

param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'

####

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print ("SVM ccuracy = ", acc)
print ('precision = ', precision_score(labels_test,pred))
print ('recall = ', recall_score(labels_test,pred))

pickle.dump(clf, open("my_classifier.pkl", "wb") )
pickle.dump(data_dict, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb") )
