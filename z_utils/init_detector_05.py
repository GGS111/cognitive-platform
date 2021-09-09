from z_utils.utils_015 import *
import os
import torch
import time
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from random import shuffle
import numpy as np



### Сначала идут выбор предикторов. cocos, detectron или detector_facenet_pytorch
#### cocos ###################################################
#Не очень понял
def detector_cocos(conf_threshold , nms_threshold ): # cервак
    path_0 = './'
    configPath = path_0 + 'data/yolov3.cfg'
    weightsPath = path_0 + 'data/yolov3.weights'
    LABELS = []
    with open(path_0 + 'data/coco.names', 'r') as f:
        for line in f.readlines():
            LABELS.append(line.strip('\r\n'))
    # thr_confidence = 0.5
    # threshold_iou = 0.1  ## 0-квадратики не пересекаются, 1- могут пересекаться
    img_size = 416
    COLORS_ = np.random.randint(100, 255, size=(1100, 3), dtype="uint8")
    predictor_cocos = Predictor_01(1, configPath, weightsPath, conf_threshold, nms_threshold, img_size, LABELS, COLORS_)

    return predictor_cocos


#### detectron cadet ###################################################
#Не очень понял
def detector_detectron_cadet(thr_confidence=0.8, model="pedet"): # cервак
    COLORS_ = np.random.randint(100, 255, size=(1100, 3),dtype="uint8")
    LABELS=['car','car1']
    path0 = '../detectron2_src/'
    if model == 'cadet':
        LABELS = ['car', 'car1']
    else:
        LABELS = ['man', 'human']
    config_path = path0 + 'test_data/' + model + '/config.yaml'
    weights_path = path0 + 'test_data/' + model + '/model_final.pth'
     
    predictor_detectron = Detectron_cadet_00(config_path, weights_path, thr_confidence,  LABELS, COLORS_)    
    
    return predictor_detectron


class Detector_Facenet_pytorch:

    def __init__(self, thr_confidence, LABELS, COLORS_, video_adapter, pathToWeights='vggface2'):
        self.weights_path = pathToWeights
        self.device = 'cuda:0'
        self.confidence_threshold = thr_confidence
        self.classIDs = []
        self.bboxes_ = []
        self.scores_ = []
        self.encodings_ = []
        self.video_adapter = video_adapter

        self.LABELS = LABELS
        self.COLORS_ = COLORS_

        self.device = torch.device('cuda:0' if video_adapter == 1 else 'cpu')
        print('Running on device: {}'.format(self.device))

        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        if video_adapter == 1:
            self.resnet.cuda()


    def run_00(self, image):
        start = time.time()
        img = Image.fromarray(image)

        # Detect faces
        self.bboxes_, self.scores_ = self.mtcnn.detect(img)
        # print('ДО ФИЛЬТРАЦИИ')
        # print('self.bboxes, self.scores',self.bboxes_, self.scores_)

        # end = time.time()
        # print("[INFO] Bbox {:.6f} seconds".format(end - start))

        # start = time.time()
        #ЧАСТЬ КОДА С ПЕРЕВОДОМ. Смотреть отсюда до self.encodings. То что закомичено используется для тестов скорости.
        to_int = np.vectorize(np.int)
        if type(self.bboxes_) != type(None):
            self.bboxes_ = to_int(self.bboxes_)
            img_cropped = self.mtcnn(img)
            # img_cropped.to(self.device, torch.float)
            if self.video_adapter == 1:
                img_cropped = img_cropped.cuda()
        #ТУТ ЗАКАНЧИВАЕТСЯ ЭТОТ КУСОК, который занимает такое же время как и детекция лица.
            # end = time.time()
            # print("[INFO] changes {:.6f} seconds".format(end - start))
            # Calculate embedding (unsqueeze to add batch dimension)
            # start = time.time()
            self.encodings_ = self.resnet(img_cropped).detach().cpu().numpy()

            #ФИЛЬТРАЦИЯ ПО КОНФИДЕНЦ ДЛЯ УДАЛЕНИЯ ЛОЖНЫХ ДЕТЕКЦИЙ
            result = []
            for idx, val in enumerate(self.scores_):
                if val < self.confidence_threshold:
                    result.append(idx)
            index_set = tuple(result)
            self.scores = np.delete(self.scores_, index_set)
            self.bboxes = np.delete(self.bboxes_, index_set,0)
            self.encodings = np.delete(self.encodings_, index_set,0)

            # print('ПОСЛЕ ФИЛЬТРАЦИИ')
            # print('self.bboxes, self.scores',self.bboxes, self.scores)
            # print('self.encodings',self.encodings)

            end = time.time()
            print("[INFO] Facenet_pytorch took {:.6f} seconds".format(end - start))

            self.classIDs = (0 * self.scores + 0).astype('uint16')
        else:
            print('NoFaces')



    def ImVisu(self, imageIn):
        return imvisupredict_02(imageIn, self.bboxes, self.classIDs, self.scores, self.LABELS, self.COLORS_)

    def Struct_00(self):
        return bbox2struct_00(self.bboxes, self.scores, self.classIDs)

#### detector facenet_pytorch ###################################################
def detector_facenet_pytorch(thr_confidence, video_adapter = 1):
    COLORS_ = np.random.randint(100, 255, size=(1100, 3), dtype="uint8")
    LABELS = ['face0', 'face1']

    predictor_facenet_pytorch = Detector_Facenet_pytorch(thr_confidence, LABELS, COLORS_, video_adapter)

    return predictor_facenet_pytorch

### Визуализация предиктора.
def imvisupredict_02(image1, bboxes, classIDs, scores, LABELS, COLORS_):
    image = image1.copy()
    (H, W) = image1.shape[:2]
    if len(scores) > 0:
        # Смотрим индексы
        for i in range(len(scores)):
            if type(bboxes) != type(None):

                # Рисуем баундинг боксы и линию на имейдже.
                color = [int(c) for c in COLORS_[classIDs[i]]]
                cv2.rectangle(image, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2],bboxes[i][3]), color, 3)
                text = "{}: {:.1f}".format(LABELS[classIDs[i]], scores[i])
                cv2.putText(image, text, (bboxes[i][2], bboxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 1)
            else:
                print('NoFaces2')

        

    return image

### Описание баундинг бокса. Стороны, центр, точность
def bbox2struct_00(bboxes,scores,classIDs,ind_classes=None):
    if type(bboxes) != type(None):
        par_all = []

        for i in range(len(scores)):
            w = -bboxes[i][0] + bboxes[i][2]
            h = -bboxes[i][1] + bboxes[i][3]

            score = scores[i]
            label = classIDs[i]

            if (ind_classes is None) or (label in ind_classes):
                param_00 = {'bords': (bboxes[i][0], bboxes[i][1],bboxes[i][2],bboxes[i][3]), 'class': label,
                            'center': ((bboxes[i][0] + bboxes[i][2]) / 2, (bboxes[i][1] + bboxes[i][3]) / 2), 'w': w, 'h': h,
                        'confidences': score}

                par_all.append(param_00)
        return par_all
    else:
        print('NoFaces3')

# Выбор модели. Используем 11 - Facenet
def descr_745_init(flag_model):
    '''
    flag_model==10: # нулевая модель
    flag_model==1: ## модель TLOSS
    flag_model==3: ## модель TLOSS основная
    flag_model==2: # гисторамма
    '''
    path_=''
    config_p=path_+"descripror_car_model_TL_03.json"
    w_path=path_+"model_TPL_654_00.h5"
    dscr_846=ObjectDescriptor(config_p, w_path,flag_model)
    return  dscr_846 

# Выбор входных данных.
class Caption_Class_01:
    #Загрузка входных данных
    def __init__(self, path, flag,scale_):
        #path - путь для доступа к входным данным: для видео - путь к файлу, для папки с файлами - путь к папке, для rtsp/ftp - ссылка
        #flag - Тип входных данных: 0 - видео, 1 - папка с файлами, 2 - rtsp-поток, 3 - ftp-поток (не реализовано)
        #scale_ - флаг для read_01. В этой версии не используется.
        self.ROS_image=None
        if flag==2:
            self.cap = cv2.VideoCapture(path)
            #Пока не очень понял насколько это нужно
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
        else:
            if flag==1:
                self.files = os.listdir(path)
                self.files.sort()
                #shuffle(self.files)
            elif flag==0:
                self.cap = cv2.VideoCapture(path)
        self.path=path
        self.file=''
        self.input_type=flag
        self.count_file=0
        self.scale_=scale_

    def isOpened(self):
        if self.input_type==1:
            if (self.count_file<len(self.files)):
                return True
            else:
                return False
        elif self.input_type==4:
            print(self.ROS_image is not None)
            return self.ROS_image is not None
        else:
            return self.cap.isOpened()

    def updateImageROS(self,image):
        self.ROS_image=image
    #Проход по папке с фоторгафиями.
    def read(self):
        if self.input_type==1:
            file_=self.files[self.count_file]
            q_=-1
            open_cv_image=None
            if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                print(file_)
                self.file=file_
                pil_image = Image.open(self.path+file_).convert('RGB')
                open_cv_image = np.array(pil_image)
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                q_=1
            self.count_file+=1    
            return q_, open_cv_image
        elif self.input_type == 4:
            return 1, self.ROS_image
        else:
            return self.cap.read()
    def release(self):
        if self.input_type==1:
            return 0
        else:
            return self.cap.release()

# Стирает папку.  
def clear_dir_00(path_out):
        try:
            for root, dirs, files in os.walk(path_out, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                    
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except:
            pass

        
        