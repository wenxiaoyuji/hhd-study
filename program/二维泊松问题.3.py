#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import math
pi = math.pi

n = input("请输入一个正整数")
n = int(n)
h = 1/n    #把区间[0,1]*[0,1]均匀剖分，步长为h


node = np.zeros(((n+1)*(n+1),2))  # 组装node

for j0 in range(n+1):    
    for i0 in range(n+1):
        node[i0 + (j0)*(n+1),0] = i0*h
        node[i0 + (j0)*(n+1),1] = j0*h
        

        
cell = np.zeros((2*n**2,3))   # 组装cell

for j1 in range(n):   
    for i1 in range(n):
        cell[i1+ j1*n ,0] = i1 + j1*(n+1)
        cell[i1+ j1*n ,1] = i1 + j1*(n+1) + 1  
        cell[i1+ j1*n ,2] = i1 + j1*(n+1) + n + 2
        
for j2 in range(n):
    for i2 in range(n):
        cell[i2+ j2*n + n**2 ,0] = i2 + j2*(n+1) + n + 2
        cell[i2+ j2*n + n**2 ,1] = i2 + j2*(n+1) + n + 1  
        cell[i2+ j2*n + n**2 ,2] = i2 + j2*(n+1)
        


# In[96]:


node


# In[97]:


m = len(node)
m


# In[98]:


cell = cell.astype(int)
cell


# In[99]:


s = len(cell)
s


# In[100]:


A = np.zeros((m,m))
B = np.zeros((m,1))


# In[101]:


face = (h**2)/2   #均匀剖分，每个单元的面积为face

for k in range(s):
    
    a0 = (node[cell[k,1],0]* node[cell[k,2],1] - node[cell[k,2],0]* node[cell[k,1],1])/(2*face)
    
    b0 = (node[cell[k,1],1] - node[cell[k,2],1])/(2*face)
    
    c0 = (node[cell[k,2],0] - node[cell[k,1],0])/(2*face)
    

    a1 = (node[cell[k,2],0]* node[cell[k,0],1] - node[cell[k,0],0]* node[cell[k,2],1])/(2*face)
    
    b1 = (node[cell[k,2],1] - node[cell[k,0],1])/(2*face)
    
    c1 = (node[cell[k,0],0] - node[cell[k,2],0])/(2*face)
    

    a2 = (node[cell[k,0],0]* node[cell[k,1],1] - node[cell[k,1],0]* node[cell[k,0],1])/(2*face)
    
    b2 = (node[cell[k,0],1] - node[cell[k,1],1])/(2*face)
    
    c2 = (node[cell[k,1],0] - node[cell[k,0],0])/(2*face)  
   
    A[cell[k,0],cell[k,0]] += (b0**2 + c0**2)*face   #组装刚度矩阵
    A[cell[k,0],cell[k,1]] += (b0*b1 + c0*c1)*face 
    A[cell[k,0],cell[k,2]] += (b0*b2 + c0*c2)*face 

    A[cell[k,1],cell[k,0]] += (b1*b0 + c1*c0)*face    
    A[cell[k,1],cell[k,1]] += (b1**2 + c1**2)*face 
    A[cell[k,1],cell[k,2]] += (b1*b2 + c1*c2)*face
    
    A[cell[k,2],cell[k,0]] += (b2*b0 + c2*c0)*face    
    A[cell[k,2],cell[k,1]] += (b2*b1 + c2*c1)*face
    A[cell[k,2],cell[k,2]] += (b2**2 + c2**2)*face 
    
    
    
    mi0 = min(node[cell[k,0],0],node[cell[k,1],0],node[cell[k,2],0])  #求每个单元节点的最小横坐标
    ma0 = max(node[cell[k,0],0],node[cell[k,1],0],node[cell[k,2],0])  #求每个单元节点的最大横坐标
    
    mi1 = min(node[cell[k,0],1],node[cell[k,1],1],node[cell[k,2],1])  #求每个单元节点的最小纵坐标
    ma1 = max(node[cell[k,0],1],node[cell[k,1],1],node[cell[k,2],1])  #求每个单元节点的最大纵坐标
    
    def f0(x,y):
        return (2*pi**2)*np.sin(pi*x)*np.sin(pi*y)*(a0+ b0*x+ c0*y)  #定义每个单元节点0的fai函数

    def f1(x,y):
        return (2*pi**2)*np.sin(pi*x)*np.sin(pi*y)*(a1+ b1*x+ c1*y) #定义每个单元节点1的fai函数

    def f2(x,y):
        return (2*pi**2)*np.sin(pi*x)*np.sin(pi*y)*(a2+ b2*x+ c2*y) #定义每个单元节点2的fai函数

    
    B[cell[k,0]] +=(h**2)*(f0(mi0,mi1)+f0(ma0,mi1)+f0(mi0,ma1)+f0(ma0,ma1))/4  #用梯形公式组装载荷向量
    B[cell[k,1]] +=(h**2)*(f1(mi0,mi1)+f1(ma0,mi1)+f1(mi0,ma1)+f1(ma0,ma1))/4
    B[cell[k,2]] +=(h**2)*(f2(mi0,mi1)+f2(ma0,mi1)+f2(mi0,ma1)+f2(ma0,ma1))/4


# In[102]:


A


# In[103]:


np.linalg.det(A)


# In[104]:


np.shape(A)


# In[105]:


B


# In[106]:


(np.mat(A).I)*(np.mat(B))


# In[107]:


A1 = np.linalg.inv(A)
np.dot(A1,B)


# In[108]:


len((np.mat(A).I)*(np.mat(B)))


# In[109]:


g = node[:,0]
g


# In[110]:


h = node[:,1]
h


# In[111]:


def u(x,y):
    return np.sin(pi*x)*np.sin(pi*y)  #真解


# In[112]:


for i in range(m):
    print(u(g[i],h[i]))


# In[ ]:




