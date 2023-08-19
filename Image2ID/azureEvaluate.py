import asyncio
import io
import glob
import os
import sys
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import pickle
import sys
sys.path.append('../..')

from loader import *



TEST=False
if not TEST:
    with open("azureKey.pvt") as f: # Not provided 
        KEY = f.readline().strip()
    ENDPOINT="https://xxxxxxxx.cognitiveservices.azure.com/" # Not provided 

    # Create an authenticated FaceClient.
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

with open("lfwpersonList.pkl", 'rb') as g:
    persons = pickle.load(g)

def getPred(imgFile, personGroup='lfwevaltrain'):
    with open(imgFile,'rb') as imgHandler:
        img = face_client.face.detect_with_stream(imgHandler,detection_model='detection_03')
    if not img:
        print("No img detected:", imgFile)
        return None
    img = img[0] # detect返回一个list，正常情况下只有一个detected img
    res = face_client.face.identify([img.face_id],person_group_id=personGroup,max_num_of_candidates_returned=5,confidence_threshold=0)
    return res

dir_list = [
# Feature Path Here
]

for inv_result_dir in dir_list:
    attributePath = "../dataset/eval_test.csv"
    _, eval_dataloader = init_dataloader(attributePath, inv_result_dir, action='eval', batch_size=1, n_classes=1000, attriID=1, skiprows=1, stream=True) # eval

    hit_top1 = 0
    hit_top5 = 0
    total = len(eval_dataloader)

    print(inv_result_dir)
    for i, (img, label) in enumerate(eval_dataloader, start = 1):
        if i % 5 == 0:
            # print("hit1,hit5,i", hit_top1, hit_top5, i)
            time.sleep(1)
        imgpath = img[0]
        label = round(label.item())
        # print(imgpath, label)
        try:
            res = getPred(imgpath)
        except:
            print("Failure at:", imgpath)
            continue
        if not res:
            continue
        res = res[0]
        res_ids = list(map(lambda x: x.person_id, res.candidates))
        # print(res_ids)
        if persons[label].person_id in res_ids[:1]:
            hit_top1 += 1
        if persons[label].person_id in res_ids[:5]:
            hit_top5 += 1
    print(inv_result_dir)
    print("top 1 acc:", hit_top1 / total)
    print("top 5 acc:", hit_top5 / total)



