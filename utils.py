import numpy as np
import cv2 
import matplotlib.pyplot as plt
from collections import deque

# 用于判断是否需要在(i,j)位置继续搜索
def judge(img,i,j,lmap):
    if i>=0 and i<512:
        if j>=0 and j<512:
            if lmap[i][j]==0 and img[i][j]!=0:
                return True
    return False

# 搜索(i,j) 所处的联通分量，返回该联通分量的的大小
def search(img,i,j,n,lmap):
    re=1
    que=deque()
    lmap[i][j]=n
    que.append((i,j))
    while len(que)>0:
        i,j=que.popleft()
        if judge(img,i+1,j,lmap):
            lmap[i+1][j]=n
            que.append((i+1,j))
            re+=1
        if judge(img,i-1,j,lmap):
            lmap[i-1][j]=n
            que.append((i-1,j))
            re+=1
        if judge(img,i,j+1,lmap):
            lmap[i][j+1]=n
            que.append((i,j+1))
            re+=1
        if judge(img,i,j-1,lmap):
            lmap[i][j-1]=n
            que.append((i,j-1))
            re+=1
    return re

# 将图片划分成若干个联通分量，返回划分结果((512*512)的数组)和各分量的大小
def component(img):
    counter=1
    lmap=np.zeros((512,512))
    lis=[]
    for i in range(512):
        for j in range(512):
            # print((i,j))
            if judge(img,i,j,lmap):
                lis.append(search(img,i,j,counter,lmap))
                counter+=1
    return lmap,lis

# 根据输入的图片计算vrand，接受输入为(512,512)的灰度图片，图片预先二值化，变为仅含0和255两个值
def vrand(img,img2):
    lmap,lis=component(img)
    lmap2,lis2=component(img2)
    P=np.zeros((len(lis),len(lis2)),dtype=float)
    for i in range(512):
        for j in range(512):
            si=round(lmap[i][j]-1)
            sj=round(lmap2[i][j]-1)
            if si>=0 and sj>=0:
                P[si,sj]+=1
    P/=(512*512)
    T=np.sum(P,axis=0)
    T=np.sum(np.square(T))
    S=np.sum(P,axis=1)
    S=np.sum(np.square(S))
    return np.sum(np.square(P))/(0.5*T+0.5*S)

# 计算sum(xlogx)
def nlogn(x):
    tem=x+0.0000001
    tem=tem*np.log(tem)
    return np.sum(tem)

# 根据输入的图片计算vinfo，接受输入为(512,512)的灰度图片，图片预先二值化，变为仅含0和255两个值
def vinfo(img,img2):
    lmap,lis=component(img)
    lmap2,lis2=component(img2)
    P=np.zeros((len(lis),len(lis2)),dtype=float)
    for i in range(512):
        for j in range(512):
            si=round(lmap[i][j]-1)
            sj=round(lmap2[i][j]-1)
            if si>=0 and sj>=0:
                P[si,sj]+=1
    P/=(512*512)
    P+=0.000000001
    T=np.sum(P,axis=0)
    S=np.sum(P,axis=1)
    return (nlogn(P)-nlogn(T)-nlogn(S))/(0.5*(-nlogn(T))+0.5*(-nlogn(S)))
