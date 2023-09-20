import math

list_abc=[]
for b in range(1,20001):
    for a in range(1,b+1):
        if a*b > 20000: break
        c = math.sqrt(a**2+b**2)
        if c == int(c): list_abc.append((a,b,int(c)))

print("The number of combinations = %d" % len(list_abc))