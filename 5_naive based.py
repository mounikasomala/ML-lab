import csv 
import math 
import random 
import statistics

def cal_probability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

dataset=[] 
dataset_size=0

with open('lab5.csv') as csvfile:
    lines=csv.reader(csvfile)
    for row in lines:
        dataset.append([float(attr) for attr in row]) 


dataset_size=len(dataset)
print("Size of dataset is: ",dataset_size)
train_size=int(0.7*dataset_size) 
print(train_size)
X_train=[] 
X_test=dataset.copy()

training_indexes=random.sample(range(dataset_size),train_size)
for i in training_indexes: 
    X_train.append(dataset[i]) 
    X_test.remove(dataset[i])

classes={}
for samples in X_train: 
    last=int(samples[-1]) 
    if last not in classes:
        classes[last]=[] 
    classes[last].append(samples)
# print(classes) 
summaries={}

for classValue,training_data in classes.items(): 
    summary=[(statistics.mean(attribute),statistics.stdev(attribute)) for attribute in zip(*training_data)] 
    del summary[-1]
    summaries[classValue]=summary
print(summaries) 

X_prediction=[]

for i in X_test: 
    probabilities={}
    for classValue,classSummary in summaries.items(): 
        probabilities[classValue]=1 
        for index,attr in enumerate(classSummary): 
            probabilities[classValue]*=cal_probability(i[index],attr[0],attr[1])
	
    best_label,best_prob=None,-1
    for classValue,probability in probabilities.items(): 
        if best_label is None or probability>best_prob:
            best_prob=probability 
            best_label=classValue 
    X_prediction.append(best_label)

correct=0
for index,key in enumerate(X_test):
   if X_test[index][-1]==X_prediction[index]: 
        correct+=1
print("Accuracy: ",correct/(float(len(X_test)))*100)


# output

Size of dataset is:  768
537
{0: [(3.2458100558659218, 2.8999993794473395), (108.92458100558659, 26.680544201713392), (67.64245810055866, 18.633291088746493), (19.68435754189944, 14.97124530560692),
     (68.10893854748603, 99.8149934482663), (30.06145251396648, 7.964942099362565), (0.41851117318435754, 0.3033499493185403), (31.175977653631286, 11.878930976411825)],
 1: [(4.64245810055866, 3.649778597591479), (143.60335195530726, 32.78469411900247), (72.53631284916202, 19.100931497739865), (23.452513966480446, 17.91621360490552),
     (109.31284916201118, 142.1673819724231), (35.61173184357542, 7.425081001763168), (0.5532011173184358, 0.3717459773610709), (36.92178770949721, 11.009930407257672)]}
Accuracy:  67.96536796536796

