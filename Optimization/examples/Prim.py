import numpy as np

def prim(nv,cost,v0):
    cost = np.array(cost)
    lowcost = np.zeros(nv)
    cloest = np.zeros(nv)

    totalmincost = nv*np.max(cost)

    for i in range(nv):
        lowcost[i] = cost[v0,i]
        closest[i] = v0
    
    for i in range(nv-1):
        mincost = np.max(cost)
        minpos = 0
        for j in range(nv):
            if (lowcost[j]<mincost) and (lowcost[j] in not 0):
                mincost = lowcost[j]
                minpos = j
                  
        lowcost[minpos] = 0
        totalmincost = totalmincost + mincost 
        
        for j in range(nv):
            if cost[minpos,j] < lowcost[j]:
                lowcost[j] = cost[minpos,j]
                closet[j] = minpos

    return totalmincost
    
    
