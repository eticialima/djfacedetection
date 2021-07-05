from django.shortcuts import render
from django.http import HttpResponse
from app.forms import FaceRecognition

# Create your views here.
def Index(request):
    form = FaceRecognition()
    return render(request, 'index.html', {'form':form})
 