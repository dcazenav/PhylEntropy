# phylEntropy


A Django web application to make simple data analysis and vizualisation.

# Installation

First, clone this repo:
```
git clone https://github.com/dcazenav/PhylEntropy.git
```

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

Note that you can replace ```<port>``` by an available port number (e.g. 8000) as follows:
```
python manage.py runserver 0.0.0.0:8000
```

## Manage downloaded files (optional)

cron and crontab installation tutorial :

```
https://www.digitalocean.com/community/tutorials/how-to-use-cron-to-automate-tasks-ubuntu-1804-fr
```

To add the command you want to automate :

```
crontab -e
``` 
To list the commands you have automated :

```
crontab -l
``` 
The command to delete the generated files :

```
*/10 * * * * find /path/to/files -name "*.png" -type f -delete

``` 

# How to use phylEntropy

You will first need to download the git repository on your computer (https://github.com/dcazenav/PhylEntropy).
I’m using Ubuntu, but you can use this tutorial on another operating system (OS).

Then you will find all the installing instructions in the READ.ME, follow them to launch the WEB application.

Then load a ".csv" file

Several options are available, but you cannot use them without an input file. It must be a ".csv" file with a semicolon or comma as field separator (no tabs).

Depending on the option chosen, you will have to refer to the formalism of the example file.

Once your input file is in good shape, you can load/drag it into the area provided for this purpose.


After validating by clicking on “Load File”, check that your file has been taken into account.
If your file has been taken into account, you should be faced with your file in the form of a table


### Use options:

Now that your file is uploaded, you can use one of the options.

The launch procedure is the same for all options except for those in the “Statistics” section.

You will find the launch instructions directly on the web application.







