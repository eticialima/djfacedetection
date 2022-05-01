from django.urls import path 
from recognition import views

urlpatterns = [
    path('',views.index2,name='index2')
]
