from django.db import models

# Create your models here.
class UserFilesForm(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField("Enter username", max_length=50)
    file = models.FileField()  # for creating file input

    class Meta:
        db_table = "userfile"


# class StudentForm(models.Model):
#     id = models.AutoField(primary_key=True)
#     username = models.CharField("Enter username", max_length=50)
#     file = models.FileField()  # for creating file input
#
#     class Meta:
#         db_table = "student"