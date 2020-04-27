# coding=UTF-8
from django.shortcuts import render
from django.http import HttpResponse
import json
from django.conf import settings
import cv2
from demo_app.algorithm import rectified, person_segmetation, person_key, person_measurement
import base64
import numpy as np
from body_shape_estimation.openpose.openpose import draw_person_pose

def home(request):
    return render(request, 'home.html')


def test(request):
    return render(request, 'test.html')


def update(request):
    return HttpResponse(json.dumps({'name': '李静侃', 'sex': 'cnm'}))


def uploadImg(request):
    image = request.FILES.get('image')
    name = request.POST.get('name')
    fname = settings.MEDIA_ROOT + '/imgs/' + name + '.jpg'
    with open(fname, 'wb') as pic:
        for c in image.chunks():
            pic.write(c)
    return HttpResponse(json.dumps({'data': fname}))


def img_rectification(request):
    img1_dir = settings.MEDIA_ROOT + '/imgs/' + 'image_raw_1.jpg'
    img2_dir = settings.MEDIA_ROOT + '/imgs/' + 'image_raw_2.jpg'
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)
    img1_left = img1[:, 0:1920, :]
    img1_right = img1[:, 1920:, :]
    img2_left = img2[:, 0:1920, :]
    img2_right = img2[:, 1920:, :]
    img1_left_retified, img1_right_retified = rectified(img1_left, img1_right)
    img2_left_retified, img2_right_retified = rectified(img2_left, img2_right)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'image_left_1.jpg', img1_left_retified)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'image_right_1.jpg', img1_right_retified)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'image_left_2.jpg', img2_left_retified)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'image_right_2.jpg', img2_right_retified)
    img1_left_base64 = base64.b64encode(cv2.imencode('.jpg', img1_left_retified)[1]).decode()
    img1_right_base64 = base64.b64encode(cv2.imencode('.jpg', img1_right_retified)[1]).decode()
    img2_left_base64 = base64.b64encode(cv2.imencode('.jpg', img2_left_retified)[1]).decode()
    img2_right_base64 = base64.b64encode(cv2.imencode('.jpg', img2_right_retified)[1]).decode()
    return HttpResponse(json.dumps({'img1_left_base64': img1_left_base64,
                                    'img1_right_base64': img1_right_base64,
                                    'img2_left_base64': img2_left_base64,
                                    'img2_right_base64': img2_right_base64}))


def img_seg(request):
    request_data = request.body
    request_dict = json.loads(request_data.decode('utf-8'))
    id = int(request_dict.get('id'))
    if id == 0:
        img_name = 'image_left_1'
    elif id == 1:
        img_name = 'image_right_1'
    elif id == 2:
        img_name = 'image_left_2'
    elif id == 3:
        img_name = 'image_right_2'
    print(img_name)
    img_path = settings.MEDIA_ROOT + '/imgs/' + img_name + '.jpg'
    img, mask = person_segmetation(img_path)
    np.save(settings.MEDIA_ROOT + '/mask/' + img_name + '.npy', mask)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'seg_' + img_name + '.jpg', img)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'mask_' + img_name + '.jpg', mask * 255)
    img_base64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return HttpResponse(json.dumps({'img_base64': img_base64, 'id': id}))


def img_key(request):
    request_data = request.body
    request_dict = json.loads(request_data.decode('utf-8'))
    id = int(request_dict.get('id'))
    if id == 0:
        img_name = 'image_left_1'
    elif id == 1:
        img_name = 'image_right_1'
    elif id == 2:
        img_name = 'image_left_2'
    elif id == 3:
        img_name = 'image_right_2'
    print(img_name)
    img_path = settings.MEDIA_ROOT + '/imgs/' + img_name + '.jpg'
    img, pose = person_key(img_path)
    np.save(settings.MEDIA_ROOT + '/key_points/' + img_name + '.npy', pose)
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/' + 'key_' + img_name + '.jpg', img)
    img_base64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return HttpResponse(json.dumps({'img_base64': img_base64, 'id': id}))


def measure(request):
    img_name_1 = 'image_left_1'
    img_name_2 = 'image_left_2'

    img1_mask = np.load(settings.MEDIA_ROOT + '/mask/' + img_name_1 + '.npy') * 225
    img1_pose = np.load(settings.MEDIA_ROOT + '/key_points/' + img_name_1 + '.npy')
    img1 = draw_person_pose(cv2.cvtColor(img1_mask, cv2.COLOR_BGR2RGB), img1_pose[np.newaxis, :])
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/final_1.jpg', img1)

    img2_mask = np.load(settings.MEDIA_ROOT + '/mask/' + img_name_2 + '.npy') * 225
    img2_pose = np.load(settings.MEDIA_ROOT + '/key_points/' + img_name_2 + '.npy')
    img2 = draw_person_pose(cv2.cvtColor(img2_mask, cv2.COLOR_BGR2RGB), img2_pose[np.newaxis, :])
    cv2.imwrite(settings.MEDIA_ROOT + '/imgs/final_2.jpg', img2)

    img1_base64 = base64.b64encode(cv2.imencode('.jpg', img1)[1]).decode()
    img2_base64 = base64.b64encode(cv2.imencode('.jpg', img2)[1]).decode()

    height, shoulder, d_left_arm, d_left_forearm, d_right_arm, d_right_forearm,\
    waist_thickness, waist_width, waist_length, d = person_measurement()

    return HttpResponse(json.dumps({'img1_base64': img1_base64,
                                    'img2_base64': img2_base64,
                                    'height': height,
                                    'shoulder': shoulder,
                                    'd_left_arm': d_left_arm,
                                    'd_left_forearm': d_left_forearm,
                                    'd_right_arm': d_right_arm,
                                    'd_right_forearm': d_right_forearm,
                                    'waist_thickness': waist_thickness,
                                    'waist_width': waist_width,
                                    'waist': waist_length,
                                    'd': d,
                                    }))














