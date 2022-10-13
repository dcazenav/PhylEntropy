
###############################
        # UPGMA #
###############################
# lowest_cell:
#   Locates the smallest cell in the table


def lowest_cell(table):

    # Set default to infinity
    min_cell = float("inf")
    x, y = -1, -1

    # Go through every cell, looking for the lowest
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i][j] < min_cell:
                min_cell = table[i][j]
                x, y = i, j

    # Return the x, y co-ordinate of cell
    return x, y


# join_labels:
#   Combines two labels in a list of labels
def join_labels(labels, a, b,val):
    # Swap if the indices are not ordered
    if b < a:
        a, b = b, a

    # Join the labels in the first index
    labels[a] = "(" + labels[a] + ":"+ str(val)+ "," + labels[b] + ":" + str(val) + ")"

    # Remove the (now redundant) label in the second index
    del labels[b]


# join_table:
#   Joins the entries of a table on the cell (a, b) by averaging their data entries
def join_table(table, a, b):
    # Swap if the indices are not ordered
    val= table[a][b] /2
    if b < a:
        a, b = b, a

    # For the lower index, reconstruct the entire row (A, i), where i < A
    row = []
    for i in range(0, a):
        row.append((table[a][i] + table[b][i]) / 2)
    table[a] = row

    # Then, reconstruct the entire column (i, A), where i > A
    #   Note: Since the matrix is lower triangular, row b only contains values for indices < b
    for i in range(a + 1, b):
        table[i][a] = (table[i][a] + table[b][i]) / 2

    #   We get the rest of the values from row i
    for i in range(b + 1, len(table)):
        table[i][a] = (table[i][a] + table[i][b]) / 2
        # Remove the (now redundant) second index column entry
        del table[i][b]

    # Remove the (now redundant) second index row
    del table[b]

    return val

################################
        # COMMUN #
################################
def reduce_table(table,labelSeq):
    taille1=len(table)
    index_ban=[]
    tab_reduce=[]
    label_reduce=[]
    index_sommet={}
    reverse_index={}
    ensemble=[]
    for i in range(len(labelSeq)):
        index_sommet[labelSeq[i]]=i
        reverse_index[i]= [labelSeq[i]]

    for i in range(taille1):
        verif = 0
        taille2= len(table[i])
        for j in range(taille2):
            if table[i][j] ==0:
                if i not in index_ban:
                    index_ban.append(i)

                tab1=reverse_index[index_sommet[labelSeq[i]]]
                tab2=reverse_index[index_sommet[labelSeq[j]]]
                for elmt in tab1 :
                    if elmt not in tab2:
                        tab2.append(elmt)
                reverse_index[index_sommet[labelSeq[j]]]=tab2
                for bct in reverse_index[index_sommet[labelSeq[j]]]:
                    index_sommet[bct]=index_sommet[labelSeq[j]]
                verif=1

        if verif==0:
            tmp = []
            for cpt in range(taille2):
                if cpt not in index_ban:
                   tmp.append(table[i][cpt])
            tab_reduce.append(tmp)
    deja=[]
    for key,value in index_sommet.items():
        if value not in deja:
            ensemble.append(reverse_index[value])
            deja.append(value)

    for i in range(len(ensemble)):
        chn = ""
        for j in range(len(ensemble[i])):
            chn += ensemble[i][j]
            if j < len(ensemble[i]) - 1:
                chn += "+"
        label_reduce.append(chn)

    return tab_reduce,label_reduce,ensemble

####################
    # kruskal #
####################
def fusion(L_1, L_2):
    L = []
    k, l = len(L_1), len(L_2)
    i, j = 0, 0
    while i < k and j < l:
        if L_1[i][0] <= L_2[j][0]:
            L.append(L_1[i])
            i += 1
        else:
            L.append(L_2[j])
            j += 1
    if i == k and j < l:
        L = L + L_2[j:]
    elif j == l and i < k:
        L = L + L_1[i:]
    return L


def tri_fusion(L):
    if len(L) <= 1:
        return (L)
    else:
        m = len(L) // 2
    return fusion(tri_fusion(L[0:m]), tri_fusion(L[m:]))


def countFreq(arr, n):
    visited = [False for i in range(n)]
    result=[]
    for i in range(n):

        if (visited[i] == True):
            continue

        count = 1
        for j in range(i + 1, n, 1):
            if (arr[i] == arr[j]):
                visited[j] = True
                count += 1
        result.append(count/n)
    return result