import numpy as np

def Dijkstra(N,graph,s0):
    graph = np.array(graph)
    maxvalue = np.max(graph)
    mark = np.zeros(N)
    dist = maxvalue*np.ones(N)

    dist[s0] = 0

    for i in range(N-1):
        mincost = maxvalue
        for j in range(N):
            if not mark[j] and (dist[j] < mincost):
                mincost = dist[j]
                minpos = j
        
        mark[minpos] = True

        for j in range(N):
            if not mark[j] and (graph[minpos,j] > 0):
                temp = dist[minpos] + graph[minpos,j]
                if temp < dist[j]:
                    dist[j] = temp
    
    return dist
