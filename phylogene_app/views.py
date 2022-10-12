from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
import json
import numpy as np
from .algo import *
from django.utils.safestring import SafeString
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

# test map
# Include the `fusioncharts.py` file that contains functions to embed the charts.
from phylogene_app.static.fusioncharts import FusionCharts
from collections import OrderedDict
# test map

import pandas as pd


# Create your views here.

def import_data(request):
    error = False
    info = []
    if 'import' in request.POST:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            error = True
            return render(request, 'import_data.html', locals())
        else:
            csv_file = request.FILES["csv_file"].read().decode("utf-8").split()
            for elmt in csv_file:
                info.append(elmt.split(";"))
            request.session['info'] = info
            return redirect(import_data)

    if 'info' in request.session:
        fichier = request.session['info']
        info_submit = True
    else:
        info_submit = False

    return render(request, 'import_data.html', locals())


def base(request):
    return render(request, 'base.html')


def intro(request):
    return render(request, 'intro.html')


def ajax_1(request):
    if request.method == 'POST':
        data = json.loads(request.POST['tasks'])
        data2 = json.loads(request.POST['labelSeq'])
        index_entete = json.loads(request.POST['entete'])
        request.session['data_file'] = data
        request.session['algo'] = request.POST['algo']
        request.session['labelSeq'] = data2
        request.session['entete'] = index_entete
        return HttpResponse('')


def run_algo(request):
    if 'data_file' in request.session and 'algo' in request.session:
        data = request.session['data_file']
        fichier = request.session['info']
        entete_colonne_selected = []
        index_entete = request.session['entete']
        algo = request.session['algo']
        labelSeq = request.session['labelSeq']
        nb_bact = len(data[0])
        chaine = []
        rows_bact = []
        tab_distance = []
        nb_var = len(data)
        # entete sélectionnersur la première ligne du fichier
        for j in index_entete:
            entete_colonne_selected.append(fichier[0][j])
        # création des chaines sring a partir du caractère de chaque colonnes
        for i in range(nb_bact):
            tmp = ''
            tmp2 = []
            # nombre de variable
            for j in range(len(data)):
                tmp = tmp + data[j][i]
                tmp2.append(data[j][i])
            chaine.append(tmp)
            rows_bact.append(tmp2)
        matrice = (nb_bact, nb_bact)
        matrice = np.zeros(matrice, dtype='int')

        for i in range(len(chaine)):
            for j in range(i + 1, len(chaine)):
                count = sum(1 for a, b in zip(chaine[i], chaine[j]) if a != b)
                matrice[i][j] = count
                matrice[j][i] = count
        for j in range(nb_bact):
            new = []
            for i in range(j):
                if j != 0:
                    new.append(matrice[j][i])
            tab_distance.append(new)
        tab_reduce, label_reduce, ensemble_seq = reduce_table(tab_distance, labelSeq)

        if algo == "upgma":
            # faire attention a la fontion UPGMA car elle vide les variables tab_reduce et label_reduce
            algo_upgma = UPGMA(tab_distance, labelSeq)
            return render(request, 'upgma.html', locals())

        if algo == "Wordcloud":
            return render(request, 'Wordcloud.html', locals())

        if algo == "kruskal":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'kruskal.html', locals())

        if algo == "Hunter-Gaston":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'Hunter-Gaston.html', locals())

        if algo == "Shannon-Entropy":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'Shannon-Entropy.html', locals())

        if algo == "neighbor-joining":
            labels = neighbor_joining(tab_distance, labelSeq)
            return render(request, 'neighbor_joining.html', locals())

        if algo == "boxplot":
            data = [list(map(int, elmt)) for elmt in data]
            return render(request, 'boxplot.html', locals())

        if algo == "heatmap":
            rows_bact = [list(map(int, elmt)) for elmt in rows_bact]
            return render(request, 'heatmap.html', locals())

        if algo == "Entropy":
            tab, score = score_entropy(data)
            trace = dict(x=[xi for xi in entete_colonne_selected],
                         y=[yi for yi in tab],
                         name="conservation",
                         type='bar')
            info1 = [trace]
            return render(request, 'Entropy.html', locals())

        if algo == "pca":
            panel_color = ['#ff0066', ' #9966ff', ' #ff0000', '#ff9900', '#669900', '#006600', '#cc00ff', '#00ffff',
                           '#ff9900', '#993300']
            colors = {}
            col_qualitative = data[-1]
            classe = []
            cpt = 0
            for i in range(len(col_qualitative)):
                if col_qualitative[i] not in classe:
                    classe.append(col_qualitative[i])
                    colors[col_qualitative[i]] = panel_color[i]
                    cpt += 1
            l = len(entete_colonne_selected) - 1
            df = pd.DataFrame(rows_bact)
            df.columns = [entete_colonne_selected]
            df.dropna(how="all", inplace=True)  # drops the empty line at file-end
            X = df.iloc[:, 0:l].values
            y = df.iloc[:, l].values

            X_std = StandardScaler().fit_transform(X)
            # print(X_std.shape[0])
            # print(X_std)
            #### variable  graphique Axe components ####
            # mean_vec = np.mean(X_std, axis=0, dtype=np.float64)
            # mean_vec = np.mean(X_std, axis=0, dtype=np.dtype('<U928'))
            print(type(mean_vec))
            # pd.options.display.float_format = '{:.20f}'.format
            # pd.set_option('display.float_format', lambda x: '%.5f' % x)
            # print(mean_vec.apply(lambda x: '%.10f' % x, axis=1))
            # format(float('-2.00897500e-16'), 'f')
            test = list(map('{:.80f}'.format, mean_vec))
            testnp = "[{0}]".format('  '.join(map(str, test)))
            testnp2 = np.array(testnp)

            # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
            cov_mat = (X_std - testnp2).T.dot((X_std - testnp2)) / (X_std.shape[0] - 1)
            # print(cov_mat)
            cov_mat = np.cov(X_std.T)
            ##problème exponentiel: print(cov_mat)
            # print(cov_mat)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            # print(eig_vals)
            tot = sum(eig_vals)

            var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
            cum_var_exp = np.cumsum(var_exp)
            trace1 = dict(
                type='bar',
                x=['PC %s' % i for i in range(1, l + 1)],
                y=list(var_exp),
                name='Individual'
            )

            trace2 = dict(
                type='scatter',
                x=['PC %s' % i for i in range(1, l + 1)],
                y=list(cum_var_exp),
                name='Cumulative'
            )

            info1 = [trace1, trace2]
            layout1 = dict(
                title='Explained variance by different principal components',
                yaxis=dict(
                    title='Explained variance in percent'
                ),
                annotations=list([
                    dict(
                        x=1.16,
                        y=1.05,
                        xref='paper',
                        yref='paper',
                        text='Explained Variance',
                        showarrow='False',
                    )
                ])
            )
            #### fin ####
            sklearn_pca = sklearnPCA(n_components=2)
            Y_sklearn = sklearn_pca.fit_transform(X_std)
            info2 = []
            for name, col in zip(classe, colors.values()):
                trace = dict(
                    type='scatter',
                    x=list(Y_sklearn[y == name, 0]),
                    y=list(Y_sklearn[y == name, 1]),
                    mode='markers',
                    name=name,
                    marker=dict(
                        symbol="diamond",
                        color=col,
                        size=12,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.5),
                        opacity=0.8)
                )
                info2.append(trace)

            layout2 = dict(
                xaxis=dict(title='PC1', showline='False'),
                yaxis=dict(title='PC2', showline='False')
            )
            return render(request, 'test2.html', locals())

        if algo == "Global Map":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'chart.html', locals())

        if algo == "Global City Map":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'chartcity.html', locals())


# def test(request):
#     return render(request, 'test.html', locals())

# def testmap(request):
#     return render(request, 'testmap.html', locals())

""" def chart(request):

  # Chart data is passed to the `dataSource` parameter, as dict, in the form of key-value pairs.
  dataSource = OrderedDict()

  # The `mapConfig` dict contains key-value pairs data for chart attribute
  mapConfig = OrderedDict()
  mapConfig["caption"] = "Average Annual Population Growth"
  mapConfig["subcaption"] = "1955-2015"
  mapConfig["numbersuffix"] = "%"
  mapConfig["includevalueinlabels"] = "1"
  mapConfig["labelsepchar"] = ":"
  mapConfig["theme"] = "fusion"

  # Map color range data
  colorDataObj = { "minvalue": "0", "code" : "#FFE0B2", "gradient": "1",
    "color" : [
        { "minValue" : "0.5", "maxValue" : "1", "code" : "#FFD74D" },
        { "minValue" : "1.0", "maxValue" : "2.0", "code" : "#FB8C00" },
        { "minValue" : "2.0", "maxValue" : "3.0", "code" : "#E65100" }
    ]
  }

  dataSource["chart"] = mapConfig
  dataSource["colorrange"] = colorDataObj
  dataSource["data"] = []


  # Map data array
  mapDataArray = [
    ["NA", "0.82", "1"],
    ["SA", "2.04", "1"],
    ["AS", "1.78", "1"],
    ["EU", "0.40", "1"],
    ["AF", "2.58", "1"],
    ["AU", "1.30", "1"]
  ]


  # Iterate through the data in `mapDataArray` and insert in to the `dataSource["data"]` list.
  # The data for the `data` should be in an array wherein each element of the array is a JSON object
  # having the `id`, `value` and `showlabel` as keys.
  for i in range(len(mapDataArray)):
      dataSource["data"].append({"id": mapDataArray[i][0], "value": mapDataArray[i][1], "showLabel": mapDataArray[i][2] })

  # Create an object for the world map using the FusionCharts class constructor
  # The chart data is passed to the `dataSource` parameter.
  fusionMap = FusionCharts("maps/world", "ex1" , "650", "450", "chart-1", "json", dataSource)
 
  # returning complete JavaScript and HTML code, which is used to generate map in the browsers.
  return  render(request, 'chart.html', {'output' : fusionMap.render(), 'chartTitle': 'Simple Map Using Array'})
"""
