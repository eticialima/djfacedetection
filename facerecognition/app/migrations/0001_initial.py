# Generated by Django 3.2.5 on 2021-07-05 02:02

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FaceRecognition',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('record_date', models.DateTimeField(auto_now_add=True)),
                ('image', models.ImageField(upload_to='images/')),
            ],
        ),
    ]
