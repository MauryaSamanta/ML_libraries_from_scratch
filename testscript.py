import random
from sklearn.datasets import load_iris
from knn import knn

#loaded the iris dataset
data=load_iris()
X=data.data
Y=data.target

#divide the DB into training and testing

combined_list=list(zip(X,Y))
random.seed(42)
random.shuffle(combined_list)

split_index=int(len(combined_list)*0.8)

train_DB=combined_list[:split_index]
test_DB=combined_list[split_index:]

X_train, Y_train=zip(*train_DB)
X_test, Y_test=zip(*test_DB)

#taking input k

input_k=int(input("enter value of nearest neighbours k:"))

custom_knn=knn(X_train, Y_train, input_k)

correct_predictions=0
size=len(X_test)

for i in range(size):
    prediction=custom_knn.predict(X_test[i])
    if(prediction==Y_test[i]):
        correct_predictions+=1

accuracy=correct_predictions/size * 100

print("Accuracy of the custom KNN implementation for {}: {:.2f}%".format(input_k,accuracy))



