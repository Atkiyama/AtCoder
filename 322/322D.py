# from functools import lru_cache
# import sys
# sys.setrecursionlimit(10**9)
# # input = sys.stdin.readline
# from decimal import Decimal
# from functools import cmp_to_key
# from collections import Counter
# from itertools import permutations
# from itertools import combinations
# from itertools import combinations_with_replacement
from itertools import product
# from itertools import accumulate
# from itertools import groupby
# from itertools import pairwise
# from copy import deepcopy
# import networkx as nx
# import networkx.algorithms as nxa
# import numpy as np
# import math
# import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict
INF = 10 ** 18
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy=[(1,0),(0,1),(-1,0),(0,-1)]
def main():
    P1=[]
    P2=[]
    P3=[]
    for i in range(4):
        P1.append(input())
    for i in range(4):
        P2.append(input())
    for i in range(4):
        P3.append(input())
        
    p1 = change(P1)
    p2 = change(P2)
    p3 = change(P3)
    # ans=[[0 for _ in range(4)] for _ in range(4)]
    # ans[2][2]=p1[2-1][2-1]
    #print(p2[3][3])
    
    bans=False
    for pi1,pj1,pw1,pi2,pj2,pw2,pi3,pj3,pw3 in product(range(-3,10),range(-3,10),range(4),range(-3,10),range(-3,10),range(4),range(-3,10),range(-3,10),range(4)):
        now_p1=rotate_array(p1,pw1)
        now_p2=rotate_array(p2,pw2)
        now_p3=rotate_array(p3,pw3)
 
        ans=[[0 for _ in range(4)] for _ in range(4)]
        flag=False
        for i in range(4):
            for j in range(4):
                if isOn(i,j,pi1,pj1):
                    ans[i][j]+=now_p1[i-pi1][j-pj1]
                if isOn(i,j,pi2,pj2):
                    ans[i][j]+=now_p2[i-pi2][j-pj2]
                if isOn(i,j,pi3,pj3):
                    ans[i][j]+=now_p3[i-pi3][j-pj3]
                if ans[i][j]!=1:
                    flag=True
        # for i in range(4):
        #     print(ans[i])
        # print()
        if not flag:
            yes()
            return
                
    no()
    # # pi1=0
    # # pj1=1
    # # pw1=1
    # pi2=1
    # pj2=1
    # pw2=0
    # # pi3
    # # pj3
    # # pw3
    # #now_p1=rotate_array(p1,pw1)
    # print(p2)
    # now_p2=rotate_array(p2,pw2)

    
    # ans=[[0 for _ in range(4)] for _ in range(4)]
    # flag=False
    # for i in range(4):
    #     for j in range(4):
    #         # if isOn(i,j,pi1,pj1):
    #         #     ans[i][j]+=now_p1[i-pi1][j-pj1]
    #         if isOn(i,j,pi2,pj2):
    #             ans[i][j]+=now_p2[j-pi2][j-pj2]
    #         # if isOn(i,j,pi3,pj3):
    #         #     ans[i][j]+=now_p3[i-pi3][j-pj3]
            
            
    # for i in range(4):
    #     print(ans[i])
    # print()
     
        
def isOn(i,j,pi,pj):
    # print(i-pi,end=",")
    # print(j-pj)
    return 4>i-pi>=0 and 4>j-pj>=0 
def rotate_array(array, degrees):
    if degrees==0:
        return array
    n = len(array)
    # 90度ごとに回転する回数を計算
    rotations = degrees % 4
    
    # 90度ずつ回転
    for _ in range(rotations):
        # 転置行列を作成
        transposed_array = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                transposed_array[i][j] = array[j][i]
        # 各行を逆順にすることで90度回転
        array = [list(reversed(row)) for row in transposed_array]
    return array   

def change(P):
    p=[]
    for i in P:
        line=[]
        for j in i:
            if j=='#':
                line.append(1)
            else:
                line.append(0)
        p.append(line)
    return p  

def swap(A,i,j):
    tmp=A[i]
    A[i]=A[j]
    A[j]=tmp
    
def basezero(num):
    return num-1

def checkIndex(list,i):
    length=len(list)
    
    if 0<=i<length:
        return True
    else:
        return False

def checkIndex2(list,i,j):
    H=len(list)
    W=len(list[0])
    if 0<=i<H and 0<=j<W:
        return True
    else:
        return False
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())
def yes():
    print("Yes")
def no():
    print("No")
def minusOne():
    print(-1)
    
if __name__ == "__main__":
    main()