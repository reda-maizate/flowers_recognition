from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import datetime
import pandas as pd
import numpy as np
import random
import os

from fastai.train import Learner
from fastai.basic_train import load_learner
from fastai.vision.image import open_image
# import torch

LEARN = load_learner(settings.LEARN_FOLDER, 'export.pkl')

def index(request):
    context = {'a': 1,
               'image': '/media/135831974sunflower.jpg'}
    # predictImage(request, context)
    return render(request, 'index.html', context)

def handled_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f.name)
    fs = FileSystemStorage()
    path = fs.save(name, f)
    url = os.path.join(settings.MEDIA_URL, name)
    return os.path.join(settings.MEDIA_ROOT, path), name, url

def getFile(request):
    return request.FILES['filePath']

def predictImage(request):
    fileObj = getFile(request)
    filePath, fileName, fileUrl = handled_uploaded_file(fileObj)
    print(filePath)
    _,_,outputs = LEARN.predict(open_image(filePath))
    outputs = outputs.numpy()
    # print(fileObj)
    # print(np.around(outputs*100, 3))
    # print(outputs[np.argmax(outputs)])
    # print(LEARN.data.classes[np.argmax(outputs)])
    # print(fileUrl)

    context = {'scores': list(np.around(outputs*100, 3)),
               'prediction_score': np.around(outputs[np.argmax(outputs)]*100, 3),
               'labels': LEARN.data.classes,
               'prediction_label': LEARN.data.classes[np.argmax(outputs)],
               'image': fileUrl}

    return render(request, 'index.html', context)
