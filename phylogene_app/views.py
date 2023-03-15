from detect_delimiter import detect
import os
import random
import uuid
import mttkinter
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns
from django.contrib.auth import get_user_model, login, logout, authenticate
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.utils.safestring import SafeString
from sklearn import svm
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .algo import *
from .form import UserRegistrationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm



# Create your views here.
def register(request):
    if request.user.is_authenticated:
        return redirect('/')

    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/')

        else:
            for error in list(form.errors.values()):
                print(request, error)

    else:
        form = UserRegistrationForm()

    return render(request=request, template_name = "users/register.html", context={"form": form})

@login_required
def custom_logout(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("homepage")

def custom_login(request):
    if request.user.is_authenticated:
        return redirect("homepage")

    if request.method == "POST":
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password"],
            )
            if user is not None:
                login(request, user)
                messages.success(request, f"Hello <b>{user.username}</b>! You have been logged in")
                return redirect("homepage")

        else:
            for error in list(form.errors.values()):
                messages.error(request, error)

    form = AuthenticationForm()

    return render(
        request=request,
        template_name="users/login.html",
        context={"form": form}
        )
def import_data(request):
    error = False
    info = []
    if 'import' in request.POST:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            error = True
            return render(request, 'phylEntropy/import_data.html', locals())
        else:
            csv_file = request.FILES["csv_file"].read().decode("utf-8").split()
            for elmt in csv_file:
                print(detect(elmt))
                info.append(elmt.split(detect(elmt)))
            request.session['info'] = info
            return redirect(import_data)
    if 'info' in request.session:
        fichier = request.session['info']
        info_submit = True
    else:
        info_submit = False

    return render(request, 'phylEntropy/import_data.html', locals())

def intro(request):
    return render(request, 'phylEntropy/intro.html')

def base(request):
    return render(request, 'phylEntropy/base.html')

def aboutphylentropy(request):
    return render(request, "phylEntropy/about.html")

def links(request):
    return render(request, "phylEntropy/links.html")

def credits(request):
    return render(request, "phylEntropy/credits.html")

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

        if algo == "UPGMA":
            # faire attention a la fontion UPGMA car elle vide les variables tab_reduce et label_reduce
            algo_upgma = UPGMA(tab_distance, labelSeq)
            return render(request, 'graph/upgma.html', locals())

        if algo == "Wordcloud":
            return render(request, 'graph/Wordcloud.html', locals())

        if algo == "Pie Chart":
            return render(request, 'graph/pie_chart.html', locals())

        if algo == "Minimun Spanning Tree":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'graph/kruskal.html', locals())

        if algo == "Hunter-Gaston":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'metrics/Hunter-Gaston.html', locals())

        if algo == "Shannon-Entropy":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'metrics/Shannon-Entropy.html', locals())

        if algo == "Neighbor-Joining":
            labels = neighbor_joining(tab_distance, labelSeq)
            return render(request, 'graph/neighbor_joining.html', locals())

        if algo == "Boxplot":
            data = [list(map(int, elmt)) for elmt in data]
            return render(request, 'statistics/boxplot.html', locals())

        if algo == "Heatmap":
            rows_bact = [list(map(int, elmt)) for elmt in rows_bact]
            return render(request, 'statistics/heatmap.html', locals())

        if algo == "Entropy":
            tab, score = score_entropy(data)
            trace = dict(x=[xi for xi in entete_colonne_selected],
                         y=[yi for yi in tab],
                         name="conservation",
                         type='bar')
            info1 = [trace]
            return render(request, 'metrics/Entropy.html', locals())

        if algo == "PCA":
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

            #### variable  graphique Axe components ####
            mean_vec = np.mean(X_std, axis=0, dtype=np.float64)

            test = list(map('{:.80f}'.format, mean_vec))
            testnp = "[{0}]".format('  '.join(map(str, test)))
            testnp2 = np.array(testnp)

            cov_mat = (X_std - testnp2).T.dot((X_std - testnp2)) / (X_std.shape[0] - 1)
            cov_mat = np.cov(X_std.T)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
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
            return render(request, 'graph/test2.html', locals())

        if algo == "Global Map":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'maps/chart.html', locals())

        if algo == "Global City Map":
            minimal_tree = kruskal(tab_reduce, label_reduce)
            minimal_tree = [SafeString(elmt) for elmt in minimal_tree]
            return render(request, 'maps/chartcity.html', locals())

        if algo == "Decision Tree":
            # dataframe ou read_csv ?
            # df remplace tb_data présent dans les script python machine_learning et test
            fichier = request.session['info']

            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)

            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            model_option = DecisionTreeClassifier()
            model_option.fit(X_train, y_train)

            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")
            # filename = str(uuid.uuid4()) + ".dot"
            #
            # tree.export_graphviz(model_option, out_file=filename,
            #                      feature_names=entete_colonne_selected[:-1],
            #                      class_names=sorted(y.unique()),
            #                      label='all',
            #                      rounded=True,
            #                      filled=True)
            predictions = model_option.predict(X_test)
            prediction2 = model_option.predict(x_d)

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index

            htmlfinal = x_d_1.join(numbers)

            score_dt = accuracy_score(y_test, predictions)
            precision_dt = precision_score(y_test, predictions, average='macro')


            # write html to file
            # For accessing the file in a folder contained in the current folder

            file_name = os.path.join('../templates/machine_learning/Decision_Tree.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Decision Tree</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

              <body>
              <div class="container-fluid" style="background-color: #a6aaa9;" >
                <h1 style="text-align: center;" >Decision Tree</h1>
                </div>
              <p> Accuracy Score : {score_dt} </p>  
              <p> Precision Score : {precision_dt} </p>  
             
              
                {table}
                
              </body>
            </html>.
            '''
            html = html_string.format(score_dt=score_dt, precision_dt=precision_dt,
                                      table=htmlfinal.to_html(
                    classes='dataframe display table table-striped table-bordered table-hover responsive nowrap'
                            'cell-border compact stripe'))

            with text_file as f:
                f.write(html)

            text_file.close()
            # return render(request, 'Decision_Tree.html', locals())
            # geeks_object = predictions.to_html()

            # return HttpResponse(geeks_object)
            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html, 'score': score_dt}
            return render(request, 'machine_learning/Decision_Tree.html', context)

        if algo == "Support Vector Machines":
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])

            clf = svm.SVC()

            clf.fit(X, y)

            predictions = clf.predict(X_test)
            svm_prediction = clf.predict(x_d)
            score_svm = accuracy_score(y_test, predictions)
            precision_svm = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index
            htmlfinal = x_d_1.join(numbers)

            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")
            file_name = os.path.join('../templates/machine_learning/Support_Vector_Machines.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Support_Vector_Machines</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                            <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >Support Vector Machines</h1>
                            </div>
                          <p> Accuracy Score : {score_svm} </p>  
                          <p> Precision Score : {precision_svm} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_svm=score_svm, precision_svm=precision_svm, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)

            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'machine_learning/Support_Vector_Machines.html', context)

        if algo == "Random Forest":
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            # valeur par défault de n_estimators(possibilité de choisir la valeur?)
            clf = RandomForestClassifier(n_estimators=5)
            clf.fit(X, y)
            # SVC()
            predictions = clf.predict(X_test)
            rf_prediction = clf.predict(x_d)
            score_rf = accuracy_score(y_test, predictions)
            precision_rf = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index
            htmlfinal = x_d_1.join(numbers)

            # write html to file
            # For accessing the file in a folder contained in the current folder
            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")

            file_name = os.path.join('../templates/machine_learning/Random_Forest.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Random Forest</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                          <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >Random Forest</h1>
                            </div>
                          <p> Accuracy Score : {score_rf} </p>
                          <p> Precision Score : {precision_rf} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_rf=score_rf, precision_rf=precision_rf, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)
            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'machine_learning/Random_Forest.html', context)

        if algo == "Extra Trees":
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            clf = ExtraTreesClassifier(n_estimators=5)
            clf.fit(X, y)
            predictions = clf.predict(X_test)
            rf_prediction = clf.predict(x_d)
            score_et = accuracy_score(y_test, predictions)
            precision_et = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index

            htmlfinal = x_d_1.join(numbers)

            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")
            # write html to file
            # For accessing the file in a folder contained in the current folder
            file_name = os.path.join('../templates/machine_learning/Extra_Trees.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Extra Trees</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                          <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >Extra Trees</h1>
                            </div>
                          <p> Accuracy Score : {score_et} </p>
                          <p> Precision Score : {precision_et} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_et=score_et, precision_et=precision_et, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)
            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'machine_learning/Extra_Trees.html', context)

        if algo == "Ada Boost":
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            clf = AdaBoostClassifier(n_estimators=5)
            clf.fit(X, y)
            predictions = clf.predict(X_test)
            ab_prediction = clf.predict(x_d)
            score_ab = accuracy_score(y_test, predictions)
            precision_ab = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index
            htmlfinal = x_d_1.join(numbers)

            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")

            # write html to file
            # For accessing the file in a folder contained in the current folder
            file_name = os.path.join('../templates/machine_learning/Ada_Boost.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Ada Boost</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                          <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >Ada Boost</h1>
                            </div>
                          <p> Accuracy Score : {score_ab} </p>
                          <p> Precision Score : {precision_ab} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_ab=score_ab, precision_ab=precision_ab, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)
            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'machine_learning/Ada_Boost.html', context)

        if algo == "K Neighbors":
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X, y)
            predictions = clf.predict(X_test)
            rf_prediction = clf.predict(x_d)
            score_knn = accuracy_score(y_test, predictions)
            precision_knn = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index

            htmlfinal = x_d_1.join(numbers)

            # write html to file
            # For accessing the file in a folder contained in the current folder
            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")

            file_name = os.path.join('../templates/machine_learning/K_Neighbors.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>K Neighbors</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                          <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >K Neighbors</h1>
                            </div>
                          <p> Accuracy Score : {score_knn} </p>
                          <p> Precision Score : {precision_knn} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_knn=score_knn, precision_knn=precision_knn, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)

            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'K_Neighbors.html', context)

        if algo == "Nayve Bayes":
            array = ['ND', 'Unknown', 'unknown', 'NA']
            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)
            X = df.drop(columns=['Type'])
            y = df['Type']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            array = ['ND', 'Unknown', 'unknown']
            df.isin(array)
            x_d_1 = []
            x_d_1 = df.loc[df['Type'].isin(array)]
            x_d = df.loc[df['Type'].isin(array)].drop(columns=['Type'])
            clf = GaussianNB()
            clf.fit(X, y)

            predictions = clf.fit(X_train, y_train).predict(X_test)
            score_nb = accuracy_score(y_test, predictions)
            precision_nb = precision_score(y_test, predictions, average='macro')

            # Ajout d'un header
            dfpred = pd.DataFrame(predictions,
                                  columns=['Prediction'])

            # Récupérer la colonne Prediction
            numbers = dfpred["Prediction"]

            # resetting the DataFrame index
            x_d_1 = x_d_1.reset_index(drop=False)

            # get colonne index
            new_index = []
            index_bact = x_d_1['index']
            for i in range(len(index_bact)):
                new_index.append(fichier[index_bact[i]][0])

            x_d_1['index'] = new_index
            htmlfinal = x_d_1.join(numbers)

            # write html to file
            # For accessing the file in a folder contained in the current folder
            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")
            file_name = os.path.join('../templates/machine_learning/Nayves_Bayes.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''<html> <head><title>Nayves Bayes</title></head> <script 
            src="https://code.jquery.com/jquery-3.4.1.slim.min.js" 
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" 
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" 
            crossorigin="anonymous"></script> <script 
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js" 
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" 
            crossorigin="anonymous"></script> <link rel="stylesheet" 
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"/> 

                          <body>
                          <div class="container-fluid" style="background-color: #a6aaa9;" >
                                <h1 style="text-align: center;" >Nayves Bayes</h1>
                            </div>
                          <p> Accuracy Score : {score_nb} </p>
                          <p> Precision Score : {precision_nb} </p>  
                            {table}
                          </body>
                        </html>.
                        '''
            html = html_string.format(score_nb=score_nb, precision_nb=precision_nb, table=htmlfinal.to_html(
                classes='dataframe display table table-striped table-bordered table-hover responsive nowrap '
                        'cell-border compact stripe'))

            with text_file as f:
                f.write(html)

            text_file.close()

            context = {'pred_Type': predictions, 'df': df, 'Unknown_Type': x_d_1, 'html': html}
            return render(request, 'machine_learning/Nayves_Bayes.html', context)

        if algo == "h_clust":
            '''X = np.matrix([[0, 0, 0, 0], [13, 0, 0, 0], [2, 14, 0, 0], [17, 1, 18, 0]])

            names = "0123"
            plot_div = ff.create_dendrogram(X,
                                       orientation='left',
                                       labels=names,
                                       linkagefun=lambda x: sch.linkage(x, "average"), )
            plot_div.update_layout(width=800, height=800)
            plot_div.show()'''

            '''x_data = [0, 1, 2, 3]
            y_data = [x ** 2 for x in x_data]
            plot_div = plot([Scatter(x=x_data, y=y_data,
                                     mode='lines', name='test',
                                     opacity=0.8, marker_color='green')],
                            output_type='div')'''

            '''df = pd.read_csv('https://git.io/clustergram_brain_cancer.csv')

            plot_div = dash_bio.Clustergram(
                data=df,
                column_labels=list(df.columns.values),
                row_labels=list(df.index),
                height=800,
                width=700
            )'''
            new_names = []
            new_fichier = fichier[1:]
            command = ["bash", "your_script_path.sh"]
            for i in range(len(new_fichier)):
                new_names.append(new_fichier[i][0])

            new_X_input = []
            for i in range(len(rows_bact)):
                new_X_input.append(rows_bact[i][:-1])

            array_for_X = np.array(new_X_input)
            X = array_for_X.astype("float64")

            fig = ff.create_dendrogram(X, orientation='left', labels=new_names)
            fig.update_layout(width=2000, height=2000)

            # pc ipg
            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")

            filename = str(uuid.uuid4()) + ".png"
            fig.write_image(BASE_DIR1 + "/static/machine_learning/" + filename)

            # images = Image.open('/home/linuxipg/Documents/PhylEntropy/phylogene_app/static/' + filename)

            # For accessing the file in a folder contained in the current folder

            file_name = os.path.join('../templates/machine_learning/h_clust.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''
                            <html>
                                <head>
                                  <meta charset="utf-8">
                                  <meta name="viewport" content="width=device-width, initial-scale=1">
                                  <title>H_CLust</title>
                                </head>
                                <body>
                                <div>
                                    <a href="../static/machine_learning/{file}" download>Link 1</a>                                                      
                                
                                </body>
                            </html>
                        '''
            html = html_string.format(file=filename)

            with text_file as f:
                f.write(html)

            text_file.close()

            context = {'html': html}
            plt.close()
            return render(request, 'machine_learning/h_clust.html', context)

        if algo == "k_means":
            new_names = []
            new_fichier = fichier[1:]
            for i in range(len(new_fichier)):
                new_names.append(new_fichier[i][-1])

            new_X_input = []
            for i in range(len(rows_bact)):
                new_X_input.append(rows_bact[i][:-1])

            # Load Data
            data = np.array(new_X_input)
            pca = PCA(n_components=0.95)

            # Transform the data
            df = pca.fit_transform(data)
            df.shape

            clusters = np.unique(new_names)

            # Initialize the class object
            kmeans = KMeans(n_clusters=len(clusters))

            # predict the labels of clusters.
            label = kmeans.fit_predict(df)

            # Getting the Centroids
            centroids = kmeans.cluster_centers_
            u_labels = np.unique(label)

            number_of_colors = len(clusters)
            color = np.unique(["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                               for i in range(number_of_colors)])


            # plotting the results:
            for i in u_labels:
                plt.scatter(df[label == i, 0], df[label == i, 1], label=clusters[i], color=color[i], s=80)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)

            # pc ipg
            BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
            os.chdir(BASE_DIR1 + "/static")

            filename = str(uuid.uuid4()) + ".png"
            plt.savefig(BASE_DIR1 + "/static/machine_learning/" + filename, dpi=100, bbox_inches='tight')



            # For accessing the file in a folder contained in the current folder
            file_name = os.path.join('../templates/machine_learning/k_means.html')
            text_file = open(file_name, "w+")

            # OUTPUT AN HTML FILE
            html_string = '''
                                        <html>
                                            <head>
                                              <meta charset="utf-8">
                                              <meta name="viewport" content="width=device-width, initial-scale=1">
                                              <title>K_means</title>
                                            </head>
                                            <body>
                                                <a href="../static/machine_learning/{file}" download>Link 1</a>                                                      

                                            </body>
                                        </html>
                                    '''
            html = html_string.format(file=filename)

            with text_file as f:
                f.write(html)

            text_file.close()

            context = {'html': html}
            plt.close()
            return render(request, 'machine_learning/k_means.html', context)

        if algo == "clustermap":
            new_names = []
            new_fichier = fichier[1:]
            for i in range(len(new_fichier)):
                new_names.append(new_fichier[i][0])

            df = pd.DataFrame(rows_bact,
                              columns=entete_colonne_selected)

            if 'Country' in df:
                df["ID"] = new_names
                df = df.set_index('ID')
                col_type = df['Type']
                del df['Type']
                del df['Country']
                df = df.astype(float)

                color = np.unique(["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                   for i in range(len(df))])
                # Prepare a vector of color mapped to the 'cyl' column
                my_palette = dict(zip(col_type.unique(), color))
                row_colors = col_type.map(my_palette)

                # plot
                sns.clustermap(df,
                               row_colors=row_colors)
                BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
                os.chdir(BASE_DIR1 + "/static")

                filename = str(uuid.uuid4()) + ".png"
                plt.savefig(BASE_DIR1 + "/static/machine_learning/" + filename, dpi=100, bbox_inches='tight')

                # For accessing the file in a folder contained in the current folder

                file_name = os.path.join('../templates/machine_learning/clustermap.html')
                text_file = open(file_name, "w+")

                # OUTPUT AN HTML FILE
                html_string = '''
                                            <html>
                                                <head>
                                                  <meta charset="utf-8">
                                                  <meta name="viewport" content="width=device-width, initial-scale=1">
                                                  <title>clustermap</title>
                                                </head>
                                                <body>
                                                    <a href="../static/machine_learning/{file}" download>Link 1</a>                                                      

                                                </body>
                                            </html>
                                        '''
                html = html_string.format(file=filename)

                with text_file as f:
                    f.write(html)

                text_file.close()

                context = {'html': html}
                plt.close()
                return render(request, 'machine_learning/clustermap.html', context)

            elif 'Country' not in df:
                df["ID"] = new_names
                df = df.set_index('ID')
                col_type = df['Type']
                del df['Type']
                df = df.astype(float)

                color = np.unique(["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                   for i in range(len(df))])
                # Prepare a vector of color mapped to the 'cyl' column
                my_palette = dict(zip(col_type.unique(), color))

                row_colors = col_type.map(my_palette)

                # plot
                sns.clustermap(df,
                               row_colors=row_colors)

                BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
                os.chdir(BASE_DIR1 + "/static")

                filename = str(uuid.uuid4()) + ".png"
                plt.savefig(BASE_DIR1 + "/static/machine_learning/" + filename, dpi=100, bbox_inches='tight')

                # images = Image.open('/home/linuxipg/Documents/PhylEntropy/phylogene_app/static/' + filename)

                # For accessing the file in a folder contained in the current folder

                file_name = os.path.join('../templates/machine_learning/dendro_heat.html')
                text_file = open(file_name, "w+")

                # OUTPUT AN HTML FILE
                html_string = '''
                                            <html>
                                                <head>
                                                  <meta charset="utf-8">
                                                  <meta name="viewport" content="width=device-width, initial-scale=1">
                                                  <title>dendro_heat</title>
                                                </head>
                                                <body>
                                                    <a href="../static/machine_learning/{file}" download>Link 1</a>                                                      

                                                </body>
                                            </html>
                                        '''
                html = html_string.format(file=filename)

                with text_file as f:
                    f.write(html)

                text_file.close()

                context = {'html': html}
                plt.close()
                return render(request, 'machine_learning/dendro_heat.html', context)
