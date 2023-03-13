# Phylentropy


A Django web application to make simple data analysis and data vizualisation.

# Installation

Phylentropy requires python 3.10.

To install it:

* You may first install and activate a new conda environment:

```
conda create --name phylentropyenv python=3.10
source activate phylentropyenv
```

* Install all Phylentropy dependencies:

```
pip3 install -r requirements.txt

```

* Create Django databases

```
python3 manage.py makemigrations
python3 manage.py migrate
```

* Create admin user

```
python manage.py createsuperuser
```

* Run the django server on local network

```
python manage.py runserver 0.0.0.0:<port>
```

## Gérer les fichiers téléchargés

cron et crontab 

```
https://www.digitalocean.com/community/tutorials/how-to-use-cron-to-automate-tasks-ubuntu-1804-fr
```

Pour ajouter la commande que vous souhaitez automatiser

```
crontab -e
``` 
La commande permettant de supprimer les fichiers génénérés

```
*/10 * * * * find /path/to/files -name "*.png" -type f -delete

``` 

# How to use phylEntropy

You will first need to download the git repository on your computer (https://github.com/dcazenav/PhylEntropy).
I’m using Ubuntu, but you can use this tutorial on another operating system (OS).

Then you will find all the installing instructions in the READ.ME, follow them to launch the WEB application.
Load .csv file

We have several option, but you can’t use them without an input file. It must be a .csv file with semicolon as field separator (no tabulation or commas).



(En fonction de l’option choisie, il faudra se référer au formalisme du fichier exemple.)


Une fois votre fichier d’entrée bien en forme, vous pouvez le charger/glisser dans la zone prévue à cet effet.




Après avoir validé en cliquant sur “Load File”, vérifier que votre fichier a bien été pris en compte. 
Si votre fichier a bien été pris en compte, vous devriez vous retrouve face à votre fichier sous forme de tableau







Utiliser les options

Maintenant que votre fichier est bien chargé, vous pouvez utiliser l’une des options. 

La procédure de lancement est la même pour toutes les options excepté pour celles de la rubrique “Statistics”.

Vous retrouverez les instructions de lancement directement sur l’application web.







