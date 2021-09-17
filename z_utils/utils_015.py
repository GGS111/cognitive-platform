from z_utils.utils_track_00 import TrackObjects

import numpy as np
import cv2
from scipy.sparse.linalg import lsqr




import torch

from PIL import Image
from scipy.spatial.distance import cosine



USE_FACE_DETECTION = 0 # 0 - использовать detect_face, 1 - использовать facenet

#if not USE_FACE_DETECTION:
    #from FaceNet import detect_face

# Коррекция центра баундин бокса по опт. флоу
def correct_opt_flow_02(q0, object_, pts, q, thr_):
    pts = np.array([pts[i][min(q, len(pts[i]) - 1)] for i in
                    range(len(object_))])
    diff_opt_fl = q0 - pts
    n_diff_opt_fl = np.linalg.norm(diff_opt_fl, axis=1)
    i_d_big = np.where(n_diff_opt_fl > thr_)
    q0[i_d_big, :] = pts[i_d_big, :]
    return q0

# Предиктор. Используется для детекта лица + отрисовки квадрата.
def facenet_pytorch_application(predictor, imageIn):
    predictor.run_00(imageIn)
    encodings = predictor.encodings
    imageOut = predictor.ImVisu(imageIn)
    par_all = predictor.Struct_00()
    return par_all, imageOut, encodings

# Получаем пулл данных о задетекченном человеке.
def X2pts_01(XXX):
    if type(XXX) != type(None):
        pts = {'coords': [], 'size': [], 'class': [], 'vector': [], 'owned': [], 'confidences': [], \
            'h_': [x['h'] for x in XXX], 'l_': [x['w'] for x in XXX]}
        pts['coords'] = np.array([x['center'] for x in XXX], dtype=np.float32)
        pts['class'] = [x['class'] for x in XXX]
        pts['confidences'] = [x['confidences'] for x in XXX]
        pts['size'] = [x['w'] * x['h'] for x in XXX]

        return pts
    else:
        print('NoFaces4')

# Получаем центры каждого лица + Его класс + Его Айди
def object2pts(object_k):


    pts = {'coords': [], 'class': []}
    pts['coords'] = np.array([x['history'][-1] for x in object_k], dtype=np.float32)
    pts['class'] = [x['class'] for x in object_k]
    pts['id']=[x['id'] for x in object_k]
    return pts

class ObjectDescriptor:
    # Выбор модели. Используем Facenet - 11
    def __init__(self, patch_json, path_to_w, flag_model):
        self.flag_model = flag_model
        self.cannal = 0  # 0-HUE,  1-яркость
        self.null_ = 0
        if flag_model == 2:  # гисторамма
            self.h = 24
            self.w = 32
            self.model_descriptor = None
        elif flag_model == 10:  # нулевая модель
            self.null_ = 1
        elif flag_model == 11:  # Facenet
            self.null_ = 5
    
    # Ресайз имейджа.
    def resize_00(self, im):
        return np.maximum(0, np.array(Image.fromarray(im).resize([self.w, self.h], Image.BICUBIC), dtype='uint8'))

    # Получаем вектора
    def descriptor_one_object_00(self, im, khown_vector = None):
        if self.null_ == 0:
            if self.flag_model == 2:
                cannal = self.cannal  # 0-HUE,  2-яркость
                k = 20
                color_image_hvs_ = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2HSV)
                hs = np.histogram(color_image_hvs_[:, :, cannal], bins=range(1, 255, k), range=None, normed=None,
                                  weights=None, density=True)[0]

            elif self.flag_model == 1 or self.flag_model == 3:  ## модель TLOSS
                X0 = self.resize_00(im)
                X1 = torch.unsqueeze(X0, 0)
                hs = self.model_descriptor.predict(X1)[0]
        # Получаем верхний и правый размер баундинг бокса. И получаем набор векторов описывающий лицо.
        elif self.null_ == 5:
            bbox = [(
                int(im.shape[0]),  # top
                int(im.shape[1]),  # right
                int(0),  # bottom
                int(0)  # left
            )]
            if USE_FACE_DETECTION:
                hs = face_encodings(im, bbox)
            else:
                if khown_vector is not None:
                    hs = khown_vector
                else:
                    hs = None
            pass
        else:
            hs = np.array([1, 1, 1, 1, 1])
            print('SELF NULL  = ', self.null_)
        return hs

# Иницилизируем параметры
def init_000(flag_prediction, cap, predictor, list_params, LABELS, COLORS):
    pnet = rnet = onet = None


    count_no_init = 0
    cannals_hist = list_params['cannals_hist']
    flag_classify_traj = list_params['flag_classify_traj']
    path_to_w_classify_traj = list_params['path_to_w_classify_traj']
    flag_prediction = list_params['flag_prediction']
    path_to_prediction = list_params['path_to_prediction']
    quality_width = list_params['quality_width']
    resize_coef = list_params['resize_coef']
    confidence_trashhold = list_params['confidence_trashhold']
    input_type_flag = list_params['input_type_flag']
    predictor_type = list_params[
        'predictor_type']  # 0 - yolo, 1 - detectron2 (cadet), 2 - detectron2 (pedet), 3 - facenet (?)
    
    # Выбор предиктора. Используем 4 - facenet_pytorch
    while (cap.isOpened()):
        ret, frame = cap.read()
        if input_type_flag != 2 or input_type_flag != 3:
            #resize_coef = frame.shape[1]/frame.shape[0]
            quality_height = int(quality_width/resize_coef)
            frame = cv2.resize(frame, (quality_width,quality_height))
        imageIn = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        XXX = frame1 = None
        try:
            if predictor_type == 4:  # facenet_pytorch
                XXX, frame1, _ = facenet_pytorch_application(predictor, imageIn)
        except:
            pass

        print('no init_000:', count_no_init)
        count_no_init += 1

        # Получаем центры + класс айди.
        try: 
            if len(XXX) > 0:
                if predictor_type == 4: # facenet_pytorch
                    pts = pts_init_facenet_pytorch(predictor, frame)

                U = [(x['w'], x['h']) for x in XXX]
                shape_ = []
                for lkj in range(len(U)):
                    shape_.append([U[lkj][0], U[lkj][1]])

                break
        except:
            print('People_No1')

    TrackObjects000 = TrackObjects(pts, flag_prediction, path_to_prediction,
                                   flag_classify_traj, path_to_w_classify_traj,
                                   LABELS, COLORS,confidence_trashhold)
    TrackObjects000.cannals_hist = cannals_hist
    TrackObjects000.im_w = frame.shape[1]
    TrackObjects000.im_h = frame.shape[0]

    return TrackObjects000

# Сборка параметров баундинг бокса. Координаты, w,h,класс. И owned - Детект/недект. Работает только на первом кадре.
def pts_init_facenet_pytorch(predictor, imageIn):
    try:
        pts = {'coords': [], 'class': [], 'owned': [], 'w': [], 'h': []}
        XXX, frame1, _ = facenet_pytorch_application(predictor, imageIn)
        pts['coords'] = np.array([x['center'] for x in XXX], dtype=np.float32)
        pts['confidences'] = [x['confidences'] for x in XXX]
        pts['class'] = [x['class'] for x in XXX]
        pts['w'] = [x['w'] for x in XXX]
        pts['h'] = [x['h'] for x in XXX]
        return pts
    except:
        print('People_No')


