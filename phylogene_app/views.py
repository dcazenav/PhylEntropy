from django.shortcuts import render,redirect
from django.http import HttpResponse
import json
import numpy as np
from .algo import *

# Create your views here.

def import_data(request):
    error = False
    info=[]
    if 'import' in request.POST:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            error= True
            return render(request, 'import_data.html', locals())
        else:
            csv_file= request.FILES["csv_file"].read().decode("utf-8").split()
            for elmt in csv_file:
                info.append(elmt.split(";"))
            request.session['info']=info
            return redirect(import_data)

    if 'info' in request.session:
        fichier = request.session['info']
        info_submit = True
    else:
        info_submit = False

    if 'kruskal' in request.POST:
        return redirect(run_algo)

    return render(request,'import_data.html', locals())


def ajax_1(request):

    if request.method == 'POST':
        data = json.loads(request.POST['tasks'])
        data2= json.loads(request.POST['labelSeq'])
        request.session['data_file']=data
        request.session['algo']= request.POST['algo']
        request.session['labelSeq']= data2

        return HttpResponse('')

def run_algo(request):

    if 'data_file' in request.session and 'algo' in request.session:
        data= request.session['data_file']
        algo= request.session['algo']
        labelSeq= request.session['labelSeq']
        nb_bact = len(data[0])
        chaine = []
        tab_distance=[]
        nb_var= len(data)
        #création des chaines sring a partir du caractère de chaque colonnes
        for i in range(nb_bact):
            tmp = ''
            #nombre de variable
            for j in range(len(data)):
                tmp = tmp + data[j][i]
            chaine.append(tmp)
        matrice= (nb_bact,nb_bact)
        matrice = np.zeros(matrice,dtype='int')

        for i in range (len(chaine)):
            for j in range (i+1,len(chaine)):
                count = sum(1 for a, b in zip(chaine[i], chaine[j]) if a != b)
                matrice[i][j] =count
                matrice[j][i] =count
        for j in range (nb_bact):
            new=[]
            for i in range (j):
                if j != 0:
                    new.append(matrice[j][i])
            tab_distance.append(new)
        tab_reduce, label_reduce,index_sommet,reverse_index = reduce_table(tab_distance, labelSeq)
        if algo == "upgma":
            #faire attention a la fontion UPGMA car elle vide les variables tab_reduce et label_reduce
            algo_upgma=UPGMA(tab_reduce,label_reduce)
            return render(request, 'upgma.html', locals())
        if algo == "kruskal":
            minimal_tree=kruskal(tab_reduce,label_reduce)
            return render(request, 'kruskal.html', locals())
        if algo == "neighbor-joining":
            labels,branche,nom_noeud = neighbor_joining(tab_reduce,label_reduce)
            return render(request, 'neighbor_joining.html', locals())
        if algo == "boxplot":
            data = [list(map(int, elmt)) for elmt in data]
            return render(request, 'boxplot.html', locals())
        if algo == "heatmap":
            return render(request, 'boxplot.html', locals())
        if algo == "test":
            # data = [list(map(int, elmt)) for elmt in data]
            return render(request, 'test2.html', locals())







