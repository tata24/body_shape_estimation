import cv2
import numpy as np
from django.conf import settings
import matplotlib.pyplot as plt

left_camera_matrix = np.array([[1.182439788403686e+03, 0.127117589316293, 1.037795688385696e+03],
                               [0, 1.181978169087777e+03, 5.627070160297170e+02],
                               [0, 0, 1]])

left_distortion = np.array([0.012270434837192, -0.012817018260326, 2.802275694509727e-04, 7.221337383953672e-04, 0.00000])


right_camera_matrix = np.array([[1.188494747570850e+03, 0.464991070014310, 1.120848677818293e+03],
                                [0 ,1.190002761311166e+03, 5.062033809395186e+02],
                                [0, 0, 1]])

right_distortion = np.array([[0.008793996829824, -0.029004917496358, -5.667084369077685e-04, -0.001221810697314, 0.00000]])



R = np.array([[0.999915564138730, 0.011872623340588, -0.005282556969724],
              [-0.011896818196006, 0.999918775514247, -0.004572536593672],
              [-0.005282556969724 , 0.004634996127486, 0.999975592952629]])

T = np.array([-1.664732040098273e+02, 1.365613063167735, -0.251502490097868]) # 平移关系向量

# size = (1280, 960) # 图像尺寸
size = (1920, 1080)


def rectified(img1, img2):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion, size, R,
                                                                      T)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                         cv2.CV_16SC2)
    img1_rectified = cv2.remap(img1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, right_map1, right_map2, cv2.INTER_LINEAR)

    for i in range(3):
        img1_rectified = np.rot90(img1_rectified)
        img2_rectified = np.rot90(img2_rectified)
    np.save(settings.MEDIA_ROOT + '/Q.npy', Q)
    return img1_rectified, img2_rectified

import torch
import torch.backends.cudnn as cudnn
import argparse
import glob
import cv2
import time
import numpy as np


from body_shape_estimation.yolact.modules.build_yolact import Yolact
from body_shape_estimation.yolact.utils.augmentations import FastBaseTransform
from body_shape_estimation.yolact.utils.functions import MovingAverage, ProgressBar
from body_shape_estimation.yolact.utils import timer
from body_shape_estimation.yolact.data.config import update_config
from body_shape_estimation.yolact.utils.output_utils import NMS, after_nms, draw_img


model_weight_path = 'body_shape_estimation/weights/yolact_res101_coco_800000.pth' # path to weight
image_path = 'data_input'   # The folder of images for detecting
mask_path = 'data_seg_output/mask'
masked_path = 'data_seg_output/masked'
pose_seg_path = 'data_pose_seg_output'
show_lincomb = False         # show_lincomb: Whether to show the generating process of masks
no_crop = False              # no_crop: Do not crop output masks with the predicted bounding box.
visual_thre = 0.1            # visual_thre: Detections with a score under this threshold will be removed
traditional_nms = False      # Whether to use traditional nms
visual_top_k = 100           # Further restrict the number of predictions to parse
hide_mask = False            # Whether to display masks
hide_bbox = False            # Whether to display bboxes
hide_score = False           # Whether to display scores

strs = model_weight_path.split('_')
config = f'{strs[-3]}_{strs[-2]}_config'
update_config(config)

def person_segmetation(img_path):
    with torch.no_grad():
        cuda = torch.cuda.is_available()
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        net.load_weights(model_weight_path, cuda)
        net.eval()
        print('Model loaded.\n')

        if cuda:
            net = net.cuda()

        img_name = img_path.split('/')[-1]
        img_origin = torch.from_numpy(cv2.imread(img_path)).float()
        # img_origin_dict[img_name] = img_origin.numpy()
        if cuda:
            img_origin = img_origin.cuda()
        img_h, img_w = img_origin.shape[0], img_origin.shape[1]
        img_trans = FastBaseTransform()(img_origin.unsqueeze(0))
        net_outs = net(img_trans)
        nms_outs = NMS(net_outs, traditional_nms)

        with timer.env('after nms'):
            results = after_nms(nms_outs, img_h, img_w, show_lincomb=show_lincomb, crop_masks=not no_crop,
                                visual_thre=visual_thre, img_name=img_name)
            # mask为01二值图
            class_ids, classes, boxes, masks = results
            # 先转化为numpy处理
            class_ids, classes, boxes, masks = class_ids.numpy(), classes.numpy(), boxes.numpy(), masks.numpy()
            # 只保留person的信息
            person_ids = np.squeeze(np.argwhere(class_ids == 0), axis=1)
            class_ids = class_ids[person_ids]
            classes = classes[person_ids]
            boxes = boxes[person_ids]
            masks = masks[person_ids]

            # 选择score最大的一个person
            if np.size(class_ids) != 0:
                max_score_person_id = np.argmax(classes).reshape(1, )
                class_ids = class_ids[max_score_person_id]
                classes = classes[max_score_person_id]
                boxes = boxes[max_score_person_id]
                masks = masks[max_score_person_id]
                # img_mask_dict[img_name] = masks[0]
                # img_bbox_dict[img_name] = boxes[0]
            results = (torch.from_numpy(class_ids), torch.from_numpy(classes), torch.from_numpy(boxes), torch.from_numpy(masks))
            if cuda:
                torch.cuda.synchronize()
            img_numpy = draw_img(results, img_origin, visual_thre=visual_thre, hide_mask=False,
                                 class_color=False, hide_bbox=False, hide_score=False)
            return img_numpy, masks[0]


from body_shape_estimation.openpose.openpose import Openpose, draw_person_pose
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签

pose_weight_path = 'body_shape_estimation/weights/posenet.pth'
openpose = Openpose(weights_file = pose_weight_path, training = False)
precise = False


def person_key(img_path):
    img_name = img_path.split('/')[-1]
    # read image
    img = cv2.imread(img_path)
    poses, _ = openpose.detect(img, precise=precise)
    # draw and save image
    img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
    # img_pose_dict[img_name] = poses[0]
    return img, poses[0]


def person_measurement():
    image_left_1_mask = np.load(settings.MEDIA_ROOT + '/mask/image_left_1.npy')
    image_right_1_mask = np.load(settings.MEDIA_ROOT + '/mask/image_right_1.npy')
    image_left_2_mask = np.load(settings.MEDIA_ROOT + '/mask/image_left_2.npy')
    # image_right_2_mask = settings.MEDIA_ROOT + '/mask/image_right_2.npy'
    image_left_1_pose = np.load(settings.MEDIA_ROOT + '/key_points/image_left_1.npy')
    # image_right_1_pose = settings.MEDIA_ROOT + '/key_points/image_right_1.npy'
    image_left_2_pose = np.load(settings.MEDIA_ROOT + '/key_points/image_left_2.npy')
    # image_right_2_pose = settings.MEDIA_ROOT + '/key_points/image_right_2.npy'
    Q = np.load(settings.MEDIA_ROOT + '/Q.npy')

    f = Q[2][3]  # 焦距 单位pixel
    Tx = 1 / Q[3][2]  # 两摄像头间距 单位mm
    Cx = -Q[0][3]  # 主点x轴像素坐标
    Cy = -Q[1][3]  # 主点y轴像素坐标

    mask_shicha = []
    for i in range(0, 1080):
        t1 = 0
        t2 = 0
        for j in range(1920):
            if image_left_1_mask[j][i] == 1:
                t1 = j
                break
        for j in range(1920):
            if image_right_1_mask[j][i] == 1:
                t2 = j
                break
        if t1 != 0 and t2 != 0:
            mask_shicha.append(t1 - t2)
    mask_shicha = np.array(mask_shicha)
    shicha = stats.mode(mask_shicha)[0][0]

    # 如果两个摄像头竖直放置，需要旋转图片，主点坐标发生改变
    if True:
        Cx = image_left_1_mask.shape[1] + Q[1][3]
        Cy = -Q[0][3]

    d = f * Tx / shicha  # 深度
    length = d / f  # 单像素点对应的实际长度 单位mm

    print("两摄像头间距:", Tx)
    print("深度：", d)
    print("单像素点对应的实际长度:", length)

    # 手臂长度计算
    d_left_arm = np.sqrt((image_left_1_pose[5][0] - image_left_1_pose[6][0]) ** 2 +
                         (image_left_1_pose[5][1] - image_left_1_pose[6][1]) ** 2) * length
    print("左大臂长度：", d_left_arm)
    d_left_forearm = np.sqrt((image_left_1_pose[6][0] - image_left_1_pose[7][0]) ** 2 +
                             (image_left_1_pose[6][1] - image_left_1_pose[7][1]) ** 2) * length
    print("左小臂长度：", d_left_forearm)
    d_right_arm = np.sqrt((image_left_1_pose[2][0] - image_left_1_pose[3][0]) ** 2 +
                          (image_left_1_pose[2][1] - image_left_1_pose[3][1]) ** 2) * length
    print("右大臂长度：", d_right_arm)
    d_right_forearm = np.sqrt((image_left_1_pose[3][0] - image_left_1_pose[4][0]) ** 2 +
                              (image_left_1_pose[3][1] - image_left_1_pose[4][1]) ** 2) * length
    print("右小臂长度：", d_right_forearm)


    # 肩宽计算 求出肩部关键点的水平连线与mask的交点
    # 计算左肩mask关键点
    left_x = int(image_left_1_pose[5][0])
    left_y = int(image_left_1_pose[5][1])
    while image_left_1_mask[left_y][left_x] == 1:
        left_x = left_x + 1
    left_x = left_x - 1
    # 计算右肩mask关键点
    right_x = int(image_left_1_pose[2][0])
    right_y = int(image_left_1_pose[2][1])
    while image_left_1_mask[right_y][right_x] == 1:
        right_x = right_x - 1
    right_x = right_x + 1

    d_shoulder = (left_x - right_x) * length
    print("肩宽：", d_shoulder)

    # 身高计算
    # 求出左视图人体mask最高点
    left_top = 0
    for i in range(image_left_1_mask.shape[0]):
        for j in range(image_left_1_mask.shape[1]):
            if image_left_1_mask[i][j] == 1:
                left_top = i
                break
        if left_top != 0:
            break

    foot = (image_left_1_pose[10][1] + image_left_1_pose[13][1]) / 2

    height = (foot - left_top) * length + 100
    print("身高：", height)

    """侧视图求臀厚"""
    left_waist_x = image_left_2_pose[11][1]  # 侧视图左臀的y坐标
    right_waist_x = image_left_2_pose[8][1]  # 侧视图右臀的y坐标

    waist_x = int(max(left_waist_x, right_waist_x))  # 侧视图，选择更大的y坐标

    # 水平延长，找到与轮廓的左右交点，交点的连线就是臀厚
    left_point = 0
    right_point = 0
    for i in range(image_left_2_mask.shape[1]):
        if image_left_2_mask[waist_x][i] == 1:
            left_point = i
            break
    for i in range(image_left_2_mask.shape[1] - 1, -1, -1):
        if image_left_2_mask[waist_x][i] == 1:
            right_point = i
            break
    waist_thickness = (right_point - left_point) * length  # 臀厚
    print("臀厚：", waist_thickness)

    """正视图求臀宽"""
    left_waist_x = int(image_left_1_pose[11][1])  # 正视图左臀的y坐标
    right_waist_x = int(image_left_1_pose[8][1])  # 正视图右臀的y坐标

    waist_x = int((left_waist_x + right_waist_x) / 2)  # 正视图，选择更大的y坐标

    # 在图片中向左延长找到左交点（右臀）
    for i in range(int(image_left_1_pose[8][0]), -1, -1):
        if image_left_1_mask[right_waist_x][i] == 0:
            left_point = i
            break

    # 在图片中向右延长找到右交点（左臀）
    for i in range(int(image_left_1_pose[11][0]), image_left_2_mask.shape[1], 1):
        if image_left_1_mask[left_waist_x][i] == 0:
            right_point = i
            break

    waist_width = (right_point - left_point - 2) * length  # 臀宽
    print("臀宽：", waist_width)

    # 根据论文中《基于数字图像的青年男体二维非接触式测量系统研究》中的臀围回归模型，来计算臀围
    a0 = 0
    a1 = 0
    a2 = 0
    level = waist_width / waist_thickness

    if level < 1.31:
        a0 = 1.12
        a1 = 1.87
        a2 = 1.325
    elif level >= 1.31 and level < 1.41:
        a0 = 7.832
        a1 = 2.442
        a2 = 0.696
    elif level >= 1.41 and level < 1.51:
        a0 = 16.623
        a1 = 1.434
        a2 = 1.194
    elif level >= 1.51 and level < 1.61:
        a0 = 80.518
        a1 = -5.273
        a2 = 3.442

    print("level:", level)
    waist_length = a0 + waist_thickness * a1 + waist_width * a2  # 臀围
    print("臀围：", waist_length)
    print(mask_shicha.tolist())
    return height, d_shoulder, d_left_arm, d_left_forearm, \
           d_right_arm, d_right_forearm, waist_thickness, waist_width, waist_length, d



















