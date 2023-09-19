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
https://twitter.com/e869120/status/1377027868518064129
二分探索を使う問題
答えの最大値と最小値が決まっている場合、それらを両端として二分探索をかけると解が求められる
'''
def main():
    N,L=map(int,input().split())
    K=int(input())
    A=[0]+list(map(int,input().split()))
    left=0
    right=L
    
    while right - left > 1:
        mid = (left+right)//2
        if solve(mid,A,K,L,N):
            left=mid
        else:
            right=mid
        
        
    print(left)

def solve(M,A,K,L,N):
    count=0
    pre = 0
    for i in range(1,N+1):
        if A[i]-pre>=M and L-A[i]>=M:
            count+=1
            pre=A[i]
        
    #print(count,end=",")
    # print(count>K,end=",")
    
    return count >=K
if __name__ == "__main__":
    main()