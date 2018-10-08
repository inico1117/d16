# d16

from sklearn.datasets import load_wine
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

wine = load_wine()
X = wine.data
y = wine.target
#print(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)
cla = MultinomialNB()
cla.fit(X_train,y_train)
y_pred = cla.predict(X_test)
print(y_test)
print(y_pred)
print(metrics.accuracy_score(y_pred,y_test)) =》 0.8472222222222222              #朴素贝叶斯-多项分布

from sklearn.svm import SVC => 0.3888888888888889                
from sklearn.linear_model import LogisticRegression => 0.9583333333333334          #逻辑回归
from sklearn.linear_model import SGDClassifier => 0.5555555555555556               #梯度下降
from sklearn.neighbors import KNeighborsClassifier => 0.6666666666666666        
