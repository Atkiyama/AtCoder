from functools import lru_cache
import sys
sys.setrecursionlimit(10**9)
# input = sys.stdin.readline
from decimal import Decimal
from functools import cmp_to_key
from collections import Counter
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product
from itertools import accumulate
from itertools import groupby
from itertools import pairwise
from copy import deepcopy
import networkx as nx
import networkx.algorithms as nxa
import numpy as np
import math
import heapq
from collections import OrderedDict
import bisect
from collections import deque
from collections import defaultdict

'''
DFSやBFSを使ってグラフの直径を求める問題
1.原点から最も離れた点を求める
2.その点から最も離れた点を求める
https://twitter.com/e869120/status/1377752658149175299
'''
def main():
    
    N=int(input())
    dict=defaultdict(list)
    for i in range(N-1):
        A,B=map(int,input().split())
        A-=1
        B-=1
        dict[A].append(B)
        dict[B].append(A)
    #print(bfs(N,dict,3))
    _,hasi=bfs(N,dict,0)
    #print(hasi)
    ans,_=bfs(N,dict,hasi)
    print(ans+1)
    
        

def bfs(N,dict,start):
    q=deque([start])
    lenDict=defaultdict(int)
    lenDict[start]=0
    length=0
    place=-1
    while q:
        now=q.pop()
        for i in dict[now]:
            if i!=start and lenDict[i]==0:
                q.append(i)
                lenDict[i]=lenDict[now]+1
                if lenDict[i]>length:
                    length=lenDict[i]
                    place=i
        
    return length,place
if __name__ == "__main__":
    main()