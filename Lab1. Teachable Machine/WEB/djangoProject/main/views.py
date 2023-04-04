import os

from django.shortcuts import render
from django.http import FileResponse, HttpResponseNotFound


# Create your views here.

def index(request):
    return render(request, 'main/index.html')


def model(request):
    return render(request, './model.json')


def weights(request):
    file_path = 'E:\\pythonProject\\djangoProject\\templates\\weights.bin'
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True)
    else:
        return HttpResponseNotFound()


def metadata(request):
    return render(request,'./metadata.json')