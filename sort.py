"""Provide unique genome IDs."""

import logging

def fast_non_dominated_sort_grade(non_sort_genomes):
    S=[[] for i in range(0,len(non_sort_genomes))]
    front = [[]]
    n=[0 for i in range(0,len(non_sort_genomes))]
    rank = [0 for i in range(0, len(non_sort_genomes))]

    for p in range(0,len(non_sort_genomes)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(non_sort_genomes)):
            if (non_sort_genomes[p].accuracy > non_sort_genomes[q].accuracy and non_sort_genomes[p].neurons < non_sort_genomes[q].neurons) or (non_sort_genomes[p].accuracy >= non_sort_genomes[q].accuracy and non_sort_genomes[p].neurons < non_sort_genomes[q].neurons) or (non_sort_genomes[p].accuracy > non_sort_genomes[q].accuracy and non_sort_genomes[p].neurons <= non_sort_genomes[q].neurons):
                if non_sort_genomes[q] not in S[p]:
                    S[p].append(non_sort_genomes[q])
            elif (non_sort_genomes[q].accuracy > non_sort_genomes[p].accuracy and non_sort_genomes[q].neurons < non_sort_genomes[p].neurons) or (non_sort_genomes[q].accuracy >= non_sort_genomes[p].accuracy and non_sort_genomes[q].neurons < non_sort_genomes[p].neurons) or (non_sort_genomes[q].accuracy > non_sort_genomes[p].accuracy and non_sort_genomes[q].neurons <= non_sort_genomes[p].neurons):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if non_sort_genomes[p] not in front[0]:
                front[0].append(non_sort_genomes[p])
        
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            # for q in S[p]:
            for q in range(0,len(S)):
                n[q] =n[q] - 1
                if(n[q]==0):
                    rank[q]=i+1
                    if non_sort_genomes[q] not in Q:
                        Q.append(non_sort_genomes[q])
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front
