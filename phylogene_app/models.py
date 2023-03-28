from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class UserFilesForm(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField("Enter username", max_length=50)
    # file = models.FileField(upload_to='upload/', validators=[FileExtensionValidator( ['csv'] ) ])  # for creating file input
    file = models.FileField("Load .csv file", upload_to='upload/') # for creating file input


    def delete_file(self):
        print('model',self.file)
        self.file.delete()
        super().delete()
    class Meta:
        db_table = "userfile"


# class StudentForm(models.Model):
#     id = models.AutoField(primary_key=True)
#     username = models.CharField("Enter username", max_length=50)
#     file = models.FileField()  # for creating file input
#
#     class Meta:
#         db_table = "student"