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

* Run the django server

```
python manage.py runserver 0.0.0.0:<port>
```
