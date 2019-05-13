from .utils import *


# UPGMA:
#   Runs the UPGMA algorithm on a labelled table
def UPGMA(table, labels):
    # Until all labels have been joined...
    while len(labels) > 1:
        # Locate lowest cell in the table
        x, y = lowest_cell(table)

        # Join the table on the cell co-ordinates
        join_table(table, x, y)

        # Update the labels accordingly
        join_labels(labels, x, y)

    # Return the final label
    return labels[0]


def kruskal(table,labels):

    liste_sommet = labels
    triplet = []
    sommet = []
    minimal_tree = []
    index_sommet = {}
    reverse_index = {}
    for i in range(len(table)):
        for j in range(len(table[i])):
            triplet.append((int(table[i][j]), liste_sommet[j], liste_sommet[i]))
    triplet = tri_fusion(triplet)
    for i in range(len(liste_sommet)):
        index_sommet[liste_sommet[i]] = i
        reverse_index[i] = [liste_sommet[i]]
    cpt = 0
    while len(sommet) < len(liste_sommet):
        elmt = triplet[cpt]
        if index_sommet[elmt[1]] != index_sommet[elmt[2]]:
            minimal_tree.append(elmt)
            tab = reverse_index[index_sommet[elmt[1]]]
            tab2 = reverse_index[index_sommet[elmt[2]]]
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
            labels[i] = "(" + labels[j] + "," + labels[i] + ":" + str(table[i][j]) + ")"
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






