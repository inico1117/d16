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

from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2
plt.figure()
l1, = plt.plot(x,y1,color='red',linewidth=2,linestyle='--',label='up')
#plt.figure()
l2, = plt.plot(x,y2,label='down')
plt.xlim(-1,2)
plt.ylim(-1,3)
plt.xlabel('I am X')
plt.ylabel('I am Y')
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
           ['really bad','bad','normal','good','really good'])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.legend(handles=[l1,l2,],labels=['stra','cur'],loc='best')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,50)
y = 2*x+1
plt.figure(num=1,figsize=(8,5))
plt.plot(x,y)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0,y0,c='b')
plt.plot([x0,x0],[0,y0],'k--',lw=2.5)
plt.annotate(r'2*x+1=%s' %y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),
             textcoords='offset points',fontsize=12,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.show()
