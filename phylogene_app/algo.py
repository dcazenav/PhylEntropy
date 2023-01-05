from .utils import *
import numpy as np


# UPGMA:
#   Runs the UPGMA algorithm on a labelled table
def UPGMA(table, labels):
    # Until all labels have been joined...
    while len(labels) > 1:
        # Locate lowest cell in the table
        x, y = lowest_cell(table)

        # Join the table on the cell co-ordinates
        val = join_table(table, x, y)

        # Update the labels accordingly
        join_labels(labels, x, y, val)

    # Return the final label
    return labels[0]


def kruskal(table, labels):
    liste_sommet = labels
    triplet = []
    sommet = []
    minimal_tree = []
    index_sommet = {}
    reverse_index = {}
    # print(table)
    # print("labels:", liste_sommet)
    # print("minimal_tree:", minimal_tree)
    # print("index_sommet:", index_sommet)
    for i in range(len(table)):
        for j in range(len(table[i])):
            triplet.append((int(table[i][j]), liste_sommet[j], liste_sommet[i]))
    triplet = tri_fusion(triplet)
    # print("triplet:", triplet)
    for i in range(len(liste_sommet)):
        index_sommet[liste_sommet[i]] = i
        reverse_index[i] = [liste_sommet[i]]
    cpt = 0
    while len(sommet) < len(liste_sommet):
        elmt = triplet[cpt]
        # print("elmt:", elmt)
        # print("index_sommet[elmt[1]] + 2:", index_sommet[elmt[1]], index_sommet[elmt[2]])
        if index_sommet[elmt[1]] != index_sommet[elmt[2]]:
            minimal_tree.append(elmt)
            tab = reverse_index[index_sommet[elmt[1]]]
            # print("tab", tab)
            tab2 = reverse_index[index_sommet[elmt[2]]]
            # print("tab2", tab2)
            for index in tab2:
                tab.append(index)
            reverse_index[index_sommet[elmt[1]]] = tab
            for bct in reverse_index[index_sommet[elmt[1]]]:
                index_sommet[bct] = index_sommet[elmt[1]]
            if elmt[1] not in sommet:
                sommet.append(elmt[1])
            if elmt[2] not in sommet:
                sommet.append(elmt[2])
        cpt += 1
    # print("mimin",minimal_tree)
    return minimal_tree


def neighbor_joining(table, labels):
    N = len(table)
    while N > 1:
        if N != 2:
            r = {}
            min_M = []
            for elmt in labels:
                r[elmt] = 0

            for i in range(len(table)):
                for j in range(len(table[i])):
                    r[labels[i]] += table[i][j]
                    r[labels[j]] += table[i][j]

            M = []
            for i in range(len(table)):
                t = []
                for j in range(len(table[i])):
                    if i != 0:
                        val = table[i][j] - (r[labels[i]] + r[labels[j]]) / (N - 2)
                        t.append(val)
                        if len(min_M) == 0 or val < min_M[0][0]:
                            min_M.insert(0, [val, i, j])

                M.append(t)
            i = min_M[0][1]
            j = min_M[0][2]
        if N != 2:
            val1 = table[i][j] / 2 + (r[labels[j]] - r[labels[i]]) / (2 * (N - 2))
            val2 = table[i][j] - val1
            labels[i] = "(" + labels[j] + ":" + str(val1) + "," + labels[i] + ":" + str(val2) + ")"
        else:
            labels[i] = "(" + labels[j] + ":" + str(table[i][j] / 2) + "," + labels[i] + ":" + str(
                table[i][j] / 2) + ")"
        del labels[j]
        if N != 2:
            for cpt1 in range(len(table)):
                for cpt2 in range(len(table[cpt1])):
                    if cpt1 != i and cpt2 == j:
                        if i > len(table[cpt1]) - 1:
                            var2 = table[i][cpt1]
                        else:
                            var2 = table[cpt1][i]

                        if j > len(table[cpt1]) - 1:
                            var1 = table[j][cpt1]
                        else:
                            var1 = table[cpt1][j]

                        d = (var1 + var2 - table[i][j]) / 2
                        table[cpt1][cpt2] = d

            del table[i]
            for cpt1 in range(len(table)):
                for cpt2 in range(len(table[cpt1])):
                    if cpt2 == i:
                        del table[cpt1][cpt2]

        N -= 1

    return labels[0]


def score_entropy(data):
    l = len(data)
    tab = []
    for elmt in data:
        value, counts = np.unique(elmt, return_counts=True)
        norm_counts = counts / counts.sum()
        val = -sum([x1 * np.log2(x1) for x1 in norm_counts])
        tab.append(1 - val)
    score = (sum(tab) / l) * 100
    tab = [x2 * 100 for x2 in tab]
    return tab, score


def simpson_di(data):
    def p(n, N):
        """ Relative abundance """
        if n == 0:
            return 0
        else:
            return float(n) / N

    N = sum(data.values())

    return sum(p(n, N) ** 2 for n in data.values() if n != 0)
