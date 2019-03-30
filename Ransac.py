#!/usr/bin/env python
# coding: utf-8

# In[12]:


#  question 1 understanding of geometrical interpretation of covariance

import cv2
import numpy as n
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as l
f = open("/home/raja/Downloads/data3_new.pkl", 'rb')
mydata=pickle.load(f)
a=n.array(mydata)
at=n.transpose(a)
# finding covariance of the given data set
m=n.cov(at)
# print(m)
# finding the mean of data
mx=n.mean(at[0,:])
my=n.mean(at[1,:])

# eigen decomposition
u,v = l.eig(m)
print(u)
print(v)
t1=u[0]*v[:,0]/35
t2=u[1]*v[:,1]/35
# t1=u[0]*v[:,0]
# print(t1)
# print(t2)
# finding endpoints

endx=mx+t1[0]
endy=my+t1[1]
end1=n.array([endx,endy])
print(end1)
endx2=mx+t2[0]
endy2=my+t2[1]
end2=n.array([endx2,endy2])
# plt.plot([mx,endx],[my,endy],'-')
# plotting
plt.plot([mx,endx],[my,endy],'r-')
plt.plot(at[0,:],at[1,:],'.')
plt.plot([mx,endx2],[my,endy2],'b-')
plt.show()


# In[16]:


#  vertical distance least squares technique
import cv2
import numpy as n
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as l
f = open("/home/raja/Downloads/data3_new.pkl", 'rb')
mydata=pickle.load(f)
a=n.array(mydata)
at=n.transpose(a)
# print(a)
Y=a[:,1]
# print(Y)
Y=Y.reshape(200,1)
b=n.ones((200,1))
x=a[:,0]
x=x.reshape(200,1)
X=n.hstack((x,b))
# print(X.shape)
d=n.matmul(n.transpose(X),Y)
print(d.shape)
s=l.inv(n.matmul(n.transpose(X),X))
B=n.matmul(s,d)
print(B)
slope=B[0,0]
intercept=B[1,0]
print(slope)
print(intercept)
y = intercept + slope *x
plt.plot(x,y,'-')
plt.plot(at[0,:],at[1,:],'.')
plt.show()


# In[20]:


#  orthogonal least square technique
import cv2
import numpy as n
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as l
f = open("/home/raja/Downloads/data2_new.pkl", 'rb')
mydata=pickle.load(f)
a=n.array(mydata)
at=n.transpose(a)
x= a[:,0].reshape(200,1)
xb= n.mean(a[:,0])
yb= n.mean(a[:,1])
d= a[:,0]-xb
d= d.reshape(200,1)
# print(d.reshape(200,1))
e= a[:,1]-yb
e= e.reshape(200,1)
# print(e)
u= n.hstack((d,e))
U= n.matmul(n.transpose(u),u)
# eigen decomposition
y,z= l.eig(U)
print(y)
print(z)
# line equation finding slope and intercept
intercept= (z[0,1]*xb+ z[1,1]*yb)/z[1,1]
slope= -z[0,1]/z[1,1]
print(intercept)
print(slope)
y = intercept + slope *x
plt.plot(x,y,'-')
plt.plot(at[0,:],at[1,:],'.')
plt.show()


# In[25]:


# ransac for best line fit
import cv2
import numpy as n
import matplotlib.pyplot as plt
import pickle
import math as m
from numpy import linalg as l
f = open("/home/raja/Downloads/data2_new.pkl", 'rb')
mydata=pickle.load(f)
A=n.array(mydata)
at=n.transpose(A)
# print(point)
j=False
while j==False:
    ranindex = n.random.randint(200)
    point1=A[ranindex]
    ranindex1 = n.random.randint(200)
    point2=A[ranindex1]
    point=n.vstack((point1,point2))
    x=A[:,0].reshape(200,1)
    y=A[:,1].reshape(200,1)
    y1=((point[1,1]-point[0,1])*((x-point[0,0])/(point[1,0]-point[0,0])))+point[0,1]
    # line equation in ax+by+c form
    a=point[1,1]-point[0,1]
    b=point[0,0]-point[1,0]
    c=(-a*point[0,0])-(b*point[0,1])
#print(a)
#print(b)
# computing orthogonal distance
    if a!=0 and b!=0:
        d=abs((a * x + b * y + c)) / (m.sqrt(a * a + b * b))
# print(d)
    count=0
    inliers = []
    k=0

    for i in d:
        
        if i<7:

            inliers.append(n.ndarray.tolist(A[k]))
            count+=1
        k=k+1
    


    if count >= (0.4*x.size):
         j=True

xInliers = []
yInliers = []

for k in inliers:
    xInliers.append(k[0])
    yInliers.append(k[1])
# print(inliers[:])
plt.plot(x,y1,'-')
plt.plot(at[0,:],at[1,:],'.')
plt.plot(xInliers,yInliers,'.b')


# In[ ]:




