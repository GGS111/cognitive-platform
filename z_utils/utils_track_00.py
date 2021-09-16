import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import lsqr
import mediapipe as mp
import time
import torch
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from PIL import Image


from scipy.spatial.distance import cosine
from torchvision import transforms


class TrackObjects:

    def __init__(self, pts, flag_prediction, path_to_prediction, flag_classify_traj,
                 path_to_w_classify_traj, LABELS, COLORS,confidence_trashhold):
        # flag_classify_traj -  классификатор траекторий: 1 - lstm 0 - CNN
        # path_to_w_classify_traj - путь к весам классификатора траекторий
        # flag_prediction - предсказатель траектории: 1 - lstm 0 - зеркальное отражение
        # path_to_prediction - путь к весам предсказателя траекторий (для lstm)

        self.imageIn = None
        self.object_ = []  # объекты трекера
        self.count_id = 0  # сколько объектов уже было с уникальным id. Счетчик объектов
        self.obj_vectors = []
        self.tracker = None
        self.count = 1000
        try:
            for i in range(pts['coords'].shape[0]):
                kvant_state = kvant_state_init()
                kvant_state['id'] = self.count_id
                kvant_state['class'] = pts['class'][i]
                kvant_state['history'].append(pts['coords'][i])
                kvant_state['history_X'].append(pts['coords'][i])
                kvant_state['init_posit'] = kvant_state['history'][0]
                kvant_state['confidences'].append(pts['confidences'][i])
                kvant_state['w'] = pts['w'][i]
                kvant_state['h'] = pts['h'][i]
                self.object_.append(kvant_state)
                self.count_id = self.count_id + 1
        except:
            pass

        t_ = (np.array(range(60)) - 30) / 60
        u_ = np.array([pow(t_, 2), t_, 0.0 * t_ + 1])
        self.v_ = u_.T
        self.predict = []  # для каждого объекта из TrackObjects.object_ предсказание его позиции в следующий момент
        self.im_w = None  # размеры имеджей потока
        self.im_h = None
        self.N = 5
        self.l_predict = 1000
        self.count_positive_traject = 0  # число траекторий класса 1
        self.cannals_hist = 0  ### 0- гистограмма hue для идентификации объекта, 2- гистограмма яркости
        self.len_ts = 80  # условие на забывание объекта: из последних len_ts отсчетов число не подтверженных объектов не более l_predict
        self.flag_prediction = flag_prediction  # предсказание траектории по lstm или по зеркальному отражению
        self.LSTM_Predictor = LSTM_Predictor(self.l_predict, self.len_ts, path_to_prediction, self.flag_prediction)
        self.InOut_Classifier = InOut_Classifier(path_to_w_classify_traj, flag_classify_traj)
        self.dscr_TL_00 = None
        self.LABELS = LABELS
        self.COLORS = COLORS
        self.confidence_trashhold = confidence_trashhold
        self.id_negative_set = []
        self.percent_of_nrecognized_for_classity = 0.5

    # Работает, когда объект теряется (красный баундинг бокс)
    def update_one_ob_00_empty(self, i_object, pts, p_opt_flow, thr_measure_hue_optflow2predict, diffuse_map, predictor = None):

        object_k = self.object_[i_object]
        #try:
        if object_k['age'] > 5:
            X_ = ((1 - p_opt_flow) * np.array(object_k['history']) + p_opt_flow * np.array(
                pts['coords'][i_object]))[0]
        else:
            X_ = np.array(pts['coords'][i_object])
        object_k['history'].append(X_)
        a = object_k
        bbox = (
            max(0, min(int(a['history'][-1][1] + a['h'] // 2), self.im_h)),  # bottom
            max(0, min(int(a['history'][-1][0] + a['w'] // 2), self.im_w)),  # right
            max(0, min(int(a['history'][-1][1] - a['h'] // 2), self.im_h)),  # top
            max(0, min(int(a['history'][-1][0] - a['w'] // 2), self.im_w))  # left
        )
        w = abs(bbox[3] - bbox[1])
        h = abs(bbox[2] - bbox[0])
        #КОСТЫЛЬ
        if w < a['w']:
            w = a['w']
        if h < a['h']:
            h = a['h']
        object_k['3DV_lost'] = self.translate_2d_to_3D_00(object_k,w,h,int(X_[0]), int(X_[1]),1377)
        #except:
            #pass

        object_k['owned'].append(0)
        object_k['age_lost'] += 1

    # Работает, когда объект детектиться (синий баундинг бокс)
    def update_one_ob_00(self, object_k, i_best, pts_from_cnn, collect_color_featres_k,collect_confiedense):
        w = int(pts_from_cnn['l_'][i_best])
        h = int(pts_from_cnn['h_'][i_best])
        
        object_k['w'] = w
        object_k['h'] = h
        if collect_confiedense[i_best] > self.confidence_trashhold:
            object_k['descriptor_'].append(collect_color_featres_k[i_best])

        X_ = np.array(pts_from_cnn['coords'][i_best])
        object_k['history'].append(X_)
        object_k['history_X'].append(X_)
        object_k['owned'].append(1)
        object_k['age_lost'] = 0
        object_k['3DV'].append(self.translate_2d_to_3D_00(object_k,w,h,int(X_[0]), int(X_[1]),1377))
        object_k['3D_points'] = self.compinsation_for_one_points(np.array(object_k['3DV']))
        object_k['holistic_age'] += 1
        object_k['confidences'].append(collect_confiedense[i_best])
        
        




    # Функция апдейта. Прорабатывает каждый новый кадр
    def Update_04(self, pts, p_opt_flow, collect_featres_k,collect_confiedense, pts_from_cnn,
                  thr_color_features, thr_measure_hue_optflow2predict,
                  frame_counter, debug_count, manual_thr_radius, predictor = None):


        for i_object in range(len(self.object_)):
            object_k = self.object_[i_object] # Параметры объекта
            object_k['age'] += 1 # Длина памяти
            thr_color_features_0 = thr_color_features
            object_k['radius'] = manual_thr_radius  # Радиус
            if object_k['age'] > 5:

                q_0 = min(object_k['age'], 5)
                try:
                    object_k['N14'] = np.linalg.norm(object_k['history'][-1] - object_k['history'][-q_0]) / q_0
                except IndexError:
                    print('Index Error in Update')

        #Проверяет есть ли хоть один человек затреканый. Если нет то возвращает пустые списки. 
        if len(collect_featres_k) == 0: #collect_featres_k - список с векторами каждого человека
            for i_object in range(len(self.object_)):
                self.update_one_ob_00_empty(i_object, pts, p_opt_flow, thr_measure_hue_optflow2predict, self.tracker.diffuse_map, predictor)
            return [], []
        else:
        
            i_owned = np.array(range(len(collect_featres_k))).ravel() #Список в стиле [0,1,2]. Количество зависит от количества распознанных векторизованных лиц

            relation_ = np.zeros((len(self.object_), len(collect_featres_k))) - 0.0001 
            #Получаем матрицу. Количество строк = количество детекций. Количество столбцов = количество распознанных векторизованных лиц

            for i_object in range(len(self.object_)):
                
                object_k = self.object_[i_object] #Информация о каждом объекте
                if type(pts_from_cnn) != type(None):
                    if len(pts_from_cnn['coords']) > 0: #Проверка на наличие объектов. Если объекты детекции есть, то входим в цикл
                        i_UU = []
                        if 1:
                            class_compare = [pts_from_cnn['class'][i] == object_k['class'] for i in i_owned] #Получаем список с значениями True
                        else:
                            class_compare = [pts_from_cnn['class'][i] != -1 for i in i_owned]
                        i_class_owned = np.where(class_compare)[0]
                        i_class = i_owned[i_class_owned] #Такой же список как и i_owned. Просто показывает сколько лиц векторизованно на кадре.
                        if len(i_class) > 0: #Заходим в цикл, если есть векторизованные лица

                            diff_points = [x - pts['coords'][i_object] for x in pts_from_cnn['coords'][i_class]]
                            #По очереди вычитаем координаты каждого айдишника.
                            n_diff_opt_fl0 = np.linalg.norm(diff_points, axis=1)
                            #Получаем нормированное расстояние между полученными координатами
                            n_diff_opt_fl = fun_865(object_k['age_lost']) + n_diff_opt_fl0 #Не используется)



                            i_d_U1 = np.where(n_diff_opt_fl0 < object_k['radius']) #Проверка по радиусу. Отбирает только те, что находятся в зоне радиуса
                            #Показывает какие итерационные элементы пересекаются между собой

                            i_d_U2 = i_d_U1
                            ###  descriptor
                            cos_measure = n_diff_opt_fl0 * 0 + 0.4 #Просто берем равное 0.1. Косинусное расстояние для всех новых детекций

                            if object_k['age'] > 1 and len(object_k['descriptor_']) > 1:
                                feature_in_u = [collect_featres_k[i] for i in i_class]
                                feature_cur = object_k['descriptor_'][-2] #Вектор с которым мы сравниваем
                                cos_measure = cos_distance_multi(feature_cur, feature_in_u) #Тут получаем более корректное косинусное расстояние, исходя из сравнения векторов

                                i_d_U2 = np.where(cos_measure > thr_color_features_0) #Проверка к какому айдишнику относится данный вектор.

                            i_d_U = np.intersect1d(i_d_U1, i_d_U2) #Смотрит есть ли в изначальном списке выбранный айдишник

                            if len(i_d_U) > 0:
                                i_UU = np.array([i_class[i] for i in i_d_U])

                                i_UU = i_UU.ravel()

                            dist_in_u = n_diff_opt_fl[i_d_U]
                            cos_measure_in_u = cos_measure[i_d_U]

                            w234 = 0.001
                            relation_[i_object, i_UU] = 0.0001 + w234 * (1 / (1 + np.power(dist_in_u, 2))) + (
                                        1 - w234) * cos_measure_in_u 
                else:
                    print('NoFaces5')



            correspond_ = one_to_one_relation_01(relation_)


            if len(correspond_) == 0:
                for i_object in range(len(self.object_)):
                    self.update_one_ob_00_empty(i_object, pts, p_opt_flow, thr_measure_hue_optflow2predict, self.tracker.diffuse_map, predictor)
            else:
                all_ind = range(relation_.shape[0])
                i1_ = np.asarray(all_ind).ravel().tolist()
                index_objects_cnn = correspond_[:, 0].ravel().tolist()

                index_objects_no_cnn = np.array(list(set(i1_).difference(set(index_objects_cnn))))
                for i_89 in range(len(index_objects_cnn)):
                    i_best = correspond_[i_89, 1]
                    i_object = index_objects_cnn[i_89]
                    self.update_one_ob_00(self.object_[i_object], i_best, pts_from_cnn, collect_featres_k,collect_confiedense)

                for i_82 in range(len(index_objects_no_cnn)):
                    i_object = index_objects_no_cnn[i_82]
                    self.update_one_ob_00_empty(i_object, pts, p_opt_flow, thr_measure_hue_optflow2predict, self.tracker.diffuse_map, predictor)
                    pass
            return relation_, correspond_

    # Функция памяти 
    def Forget_horizont_vert(self, h_0):
        object_01 = []
        object_ = self.object_
        for iop in range(len(object_)):
            # h_0 - берет последние значения.
            object_[iop]['history'] = object_[iop]['history'][-h_0:]
            object_[iop]['history_X'] = object_[iop]['history_X'][-h_0:]
            object_[iop]['owned'] = object_[iop]['owned'][-150:] # Сколько помнит объект, когда потерял его.
            object_[iop]['descriptor_'] = object_[iop]['descriptor_'][-h_0:] 
            object_[iop]['class_trajectory'] = object_[iop]['class_trajectory'][-h_0:]
            object_[iop]['holistic'] = object_[iop]['holistic'][-7:]
            object_[iop]['depth'] = object_[iop]['depth'][-60:]
            object_[iop]['confidences'] = object_[iop]['confidences'][-h_0:]
            object_[iop]['3DV'] = object_[iop]['3DV'][-7:]


            if sum(object_[iop]['owned']) > 0.1 * len(object_[iop]['owned']) or len(object_[iop]['owned']) < 10:

                object_01.append(object_[iop])

        self.object_ = object_01

    # Добавление новых объектов
    def add_new_object(self, pts_01, classes_01, XXX, r_, border_edge_0, rotine_edge, ind_classes, frame1, known_vectors = None):
        image = self.imageIn
        border_edge = border_edge_0 * self.im_w / 500
        pts_00 = pts_01.copy()
        classes_00 = classes_01.copy()
        ### XXX from cnn
        ### pts_00 from object
        ###  classes_00 are classes of objects
        if type(XXX) != type(None):
            bords = [(x['bords'], x['class'], x['center'], x['w'], x['h']) for x in XXX]

            H_features_ = object_featre_descriptor(image, XXX, self.dscr_TL_00, 0, self.LABELS, known_vectors) #Получаем векторы для сравнения


            count = 0
            for borders_, class_, center_, w_, h_ in bords:
                #print(w_,h_)
                q_ = not (check_points_02(borders_, pts_00, r_, class_, classes_00))
                if rotine_edge == 1:
                    p_ = (center_[1] > border_edge) and \
                        (center_[1] < (self.im_h - border_edge)) and \
                        (center_[0] > border_edge) and \
                        (center_[0] < (self.im_w - border_edge))
                elif rotine_edge == -1:
                    p_ = (center_[1] < border_edge) or \
                        (center_[1] > (self.im_h - border_edge)) or \
                        (center_[0] < border_edge) or \
                        (center_[0] > (self.im_w - border_edge))
                if q_ and p_:
                    if class_ in ind_classes:
                        kvant_state = kvant_state_init()
                        kvant_state['descriptor_'].append(H_features_[count])
                        kvant_state['id'] = self.count_id + 1
                        self.count_id = self.count_id + 1
                        kvant_state['class'] = class_
                        kvant_state['history'].append(center_)
                        kvant_state['history_X'].append(center_)
                        kvant_state['init_posit'] = kvant_state['history'][0]
                        kvant_state['w'] = w_
                        kvant_state['h'] = h_
                        self.object_.append(kvant_state)
                        pts_00.append(center_)
                        classes_00.append(class_)
                else:
                    cv2.rectangle(frame1, (int(center_[0] - w_ / 2), int(center_[1] + h_ / 2)),
                                 (int(center_[0] + w_ / 2), int(center_[1] - h_ / 2)), (255, 255, 0), 5)
                count += 1
        else:
            print('NoFaces6')

    # Рисуем красные/синие баундинг боксы
    def Draw(self, frame1, frame_counter, labels=None):

        object_0 = self.object_
        count = 0
        for i in range(len(object_0)):
            couple = object_0[i]['history'][-1]
            color_id = (255, 255, 255)
            q_0 = object_0[i]['class_trajectory_general']

            if q_0 < 0:
                color_id = (255, 0, 10)
            elif q_0 > 0:
                color_id = (10, 0, 255)
            if labels is not None:
                cv2.putText(frame1, labels[i], (int(couple[0] - 20), int(couple[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, \
                            1, color_id, 2)
            else:
                cv2.putText(frame1, 'id: ' + str(object_0[i]['id']), (int(couple[0] - 20), int(couple[1]) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color_id, 2)
            color = [int(c) for c in self.COLORS[object_0[i]['id'] % 100]]
            owned = object_0[i]['owned'][-1]
            
            if owned < 0.5:

                a = object_0[i]
                bbox = [(
                    int(a['history'][-1][1] + a['h'] // 2),  # top
                    int(a['history'][-1][0] + a['w'] // 2),  # right
                    int(a['history'][-1][1] - a['h'] // 2),  # bottom
                    int(a['history'][-1][0] - a['w'] // 2)  # left
                )]
                cv2.rectangle(frame1, (bbox[0][3], bbox[0][0]), (bbox[0][1], bbox[0][2]), (255, 0, 10), 2) #Красный ББокс

            else:
                
                cv2.circle(frame1, (int(couple[0]), int(couple[1])), 5, color, 4)
                a = object_0[i]
                bbox = [(
                    int(a['history'][-1][1] + a['h'] // 2),  # top
                    int(a['history'][-1][0] + a['w'] // 2),  # right
                    int(a['history'][-1][1] - a['h'] // 2),  # bottom
                    int(a['history'][-1][0] - a['w'] // 2)  # left
                )]
                cv2.rectangle(frame1, (bbox[0][3], bbox[0][0]), (bbox[0][1], bbox[0][2]), (10, 0, 255), 2) #Синий Ббокс


            cv2.circle(frame1, (int(couple[0]), int(couple[1])), int(object_0[i]['radius']), color, 1)


            if object_0[i]['age'] > 1:
                count += 1
        plt.imshow(frame1)
        plt.show()
        print('Draw')



    
    # Скелетализация
    def holistic_model(self, frame1):
        start = time.time()
        object_0 = self.object_
        mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        result = []
        for i in range(len(object_0)):
            points = []
            owned = object_0[i]['owned'][-1]
            age = object_0[i]['holistic_age']
            if owned > 0.5 and age > 4 and object_0[i]['w'] > 50 and object_0[i]['h'] > 50:
                a = object_0[i]

                bbox = [
                    int(a['history'][-1][1] + a['h'] * 10),  # top
                    int(a['history'][-1][0] + a['w'] * 3),  # right
                    int(a['history'][-1][1] - a['h'] // 1.25),  # bottom
                    int(a['history'][-1][0] - a['w'] * 3)  # left
                ]

                img_height, img_width, img_channels = frame1.shape

                if bbox[0] > img_height:
                    bbox[0] = img_height - 10
                if bbox[1] > img_width:
                    bbox[1] = img_width - 10
                if bbox[2] < 0:
                    bbox[2] = 10
                if bbox[3] < 0:
                    bbox[3] = 10
                

                frame2 = frame1[bbox[2]:bbox[0], bbox[3]:bbox[1]]
                
                with mp_holistic.Holistic(static_image_mode=True,
                                        model_complexity=2, 
                                        min_detection_confidence=0.1, 
                                        min_tracking_confidence=0.4) as holistic:
                    image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False      

                    # Make Detections
                    results = holistic.process(image)


                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)          

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )


                    try:
                        a['holistic'].append(results.pose_landmarks)
                    except:
                        pass
                    ###Обнуление комонентов
                    image_ = image.copy()
                    del image
                    del results
            

                x_offset= bbox[3]
                y_offset= bbox[2]
                frame1[y_offset:y_offset+(bbox[0]-bbox[2]), x_offset:x_offset+(bbox[1]-bbox[3])] = image_
                end = time.time()
                #print("[INFO] Holistic took {:.6f} seconds".format(end - start))
                
    
    def translate_2d_to_3D_00(self,object_k,w_,h_,x_, y_,f):
        '''
        w_,h_  ширина высота ББокса
        x_, y_ центр ббокса
        f= фокус
        depth=  расстояние в метрах до головы
        '''
        model = LinearRegression()
        eps_=10**(-10)

        depth_w = self.polynomial_regression_for_w(w_)
        depth_h = self.polynomial_regression_for_h(h_)
        depth_ = (depth_w + depth_h)/200
        object_k['depth'].append(depth_)
        #Посторение линейное регрессии
        d = object_k['depth'][-5:] #Берем список 5 последних значений глубины. По другому это выходы/предикторы
        x = np.array(range(1,len(d)+1)).reshape((-1, 1)) #Входы/регрессоры
        model.fit(x, d) #Получаем веса
        model = LinearRegression().fit(x, d) #Получаем готовую модель. Из нее получаем предикшены на следующие 5 значений глубины
        depth = np.mean([model.predict(x),d]) #Усредняем полученные значения. Это и будет наша глубина
        Z_=np.array([self.im_w/2,self.im_h/2])#центр
        OX=np.array([x_, y_])-Z_# вектор из центра в точку
        V_3D=np.array([0,0,f])+np.array([OX[0],OX[1],0])
        V_3D_n=V_3D/(np.linalg.norm(V_3D)+eps_)
        V_3D_q=V_3D_n*depth

        return V_3D_q
    
    def transate_one_landmark_to_3d(self,frame1):
        res_path_out = "save_fig/"
        object_0 = self.object_
        fig = plt.figure(figsize=(32, 20))
        self.find_3d_one_points(object_0,fig) #Ищем 3Д точки от скелета
        ax3 = fig.add_subplot(133) 
        ax3.imshow(frame1)
        plt.savefig(res_path_out + str(self.count) + '.jpg')
        self.count += 1
        plt.close(fig)

    def find_3d_one_points(self,object_0,fig):
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(131, projection='3d')
        for i in range(len(object_0)):
            a = object_0[i]
            owned =a['owned'][-1]
            age = a['age']
            if owned > 0.5 and age > 1:
                self.drow_cub_1(ax1,a,a['3D_points'][0],a['3D_points'][1],a['3D_points'][2],'Вид спереди',3,270,270)
                self.drow_cub_1(ax2,a,a['3D_points'][0],a['3D_points'][1],a['3D_points'][2],'Вид сверху',2,0,270)

            elif owned < 0.5 and age > 1:
                try:
                    print('id',object_0[i]['id'])
                    print('lost',a['3DV_lost'])
                    print('3DV',a['3D_points'])
                    self.drow_cub_1(ax1,a,a['3DV_lost'][0],a['3DV_lost'][1],a['3DV_lost'][2],'Вид спереди',3,270,270)
                    self.drow_cub_1(ax2,a,a['3DV_lost'][0],a['3DV_lost'][1],a['3DV_lost'][2],'Вид сверху',2,0,270)
                except:
                    pass

    def drow_cub_1(self,ax,object,x,y,z,view,size,elev,azim):
        colors = ['coral', 'lime', 'blue', 'purple', 'green', 'black', 'silver', 'yellow', 'red', 'pink','black', 'silver', 'yellow', 'red', 'pink']
        ax.elev = elev #270 - вид спереди. 0 - вид сверху
        ax.azim = azim #270 ставим
        ax.scatter(0, 0, 0, c='r') #Точка камеры. Цвет красная
        ax.scatter(x,y,z, c = colors[object['id']], label = 'id:' + str(object['id'])) #Легенда
        ax.legend(shadow=True, fancybox=True) #Легенда
        ax.set_title(view, pad=30) #Название карты
        #ax.scatter(x[0], y[0], z[0],s = 40, c='black') #Точки из алгоритма Михаила.
        zz = size 
        #Рисуем Куб из зелных точек
        #Куб вид сверху
        if zz == 2:
            ax.scatter(zz, -zz, 0, c='g')
            ax.scatter(-zz, zz, 0, c='g')
            ax.scatter(-zz, -zz, zz*2, c='g')
            ax.scatter(zz, zz, 0, c='g')
            ax.scatter(zz, -zz, zz*2, c='g')
            ax.scatter(-zz, zz, zz*2, c='g')
            ax.scatter(zz, zz, zz*2, c='g')
        #Куб вид спереди
        else:
            ax.scatter(zz, -zz, -zz, c='g')
            ax.scatter(-zz, zz, -zz, c='g')
            ax.scatter(-zz, -zz, zz, c='g')
            ax.scatter(zz, zz, -zz, c='g')
            ax.scatter(zz, -zz, zz, c='g')
            ax.scatter(-zz, zz, zz, c='g')
            ax.scatter(zz, zz, zz, c='g')

    def transate_landmarks_to_3d(self,frame1,compensaition_index,collect_confiedense):
        res_path_out = "save_fig/"
        object_0 = self.object_
        fig = plt.figure(figsize=(32, 20))
        self.find_3d_points(object_0,fig,compensaition_index,collect_confiedense) #Ищем 3Д точки от скелета
        ax3 = fig.add_subplot(133) 
        ax3.imshow(frame1)
        plt.savefig(res_path_out + str(self.count) + '.jpg')
        self.count += 1
        plt.close(fig)
    
    def find_3d_points(self,object_0,fig,compensaition_index,collect_confiedense):
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(131, projection='3d')
        for i in range(len(object_0)):
            a = object_0[i]
            owned =a['owned'][-1]
            age = a['holistic_age']
            if owned > 0.5 and age > 4:
                #try:
                if a['confidences'][-1] > 0.99:
                    spis_x = []
                    spis_y = []
                    spis_z = []
                    mean_points = self.compinsation_for_points(a['holistic'],compensaition_index) #Получаем усредненные точки
                    for i in mean_points:
                        #Считаем нормы разности по каждой координате.
                        norma_x = mean_points[0][0] - i[0]
                        norma_y = mean_points[0][1] - i[1]
                        norma_z = (mean_points[0][2] - i[2]) * 0.3
                        #Получаем реальные координаты по х,y,z для тела и добавляем в списки
                        spis_x.append(a['3D_points'][0] - norma_x)
                        spis_y.append(a['3D_points'][1] - norma_y)
                        spis_z.append(a['3D_points'][2] - norma_z)
                    a['3D_holistic'].append([spis_x,spis_y,spis_z])
                    self.drow_cub_1(ax1,a,spis_x,spis_y,spis_z,'Вид спереди',3,270,270)
                    self.drow_cub_1(ax2,a,spis_x,spis_y,spis_z,'Вид сверху',2,0,270)
                        # ax.scatter(a['3DV'][0], a['3DV'][1], a['3DV'][2],s = 40, c='g') #Точки из алгоритма Михаила.
                    #except:
                        #pass
                #Костыль за случай конфиденса маленького
                else:
                    try:
                        if a['3DV_lost'] != []:
                            print('sit1')
                            raznica = a['3D_points'] - a['3DV_lost'] 
                            x = np.array(a['3D_holistic'][-2][0]) + raznica[0]
                            y = np.array(a['3D_holistic'][-2][1]) + raznica[1]
                            z = np.array(a['3D_holistic'][-2][2]) + raznica[2]
                            self.drow_cub_1(ax1,a,x,y,z,'Вид спереди',3,270,270)
                            self.drow_cub_1(ax2,a,x,y,z,'Вид сверху',2,0,270)
                        else:
                            print('sit2')
                            self.drow_cub_1(ax1,a,a['3D_holistic'][-2][0],a['3D_holistic'][-2][1],a['3D_holistic'][-2][2],'Вид спереди',3,270,270)
                            self.drow_cub_1(ax2,a,a['3D_holistic'][-2][0],a['3D_holistic'][-2][1],a['3D_holistic'][-2][2],'Вид сверху',2,0,270)
                    except:
                        pass

            elif owned < 0.5 and age > 4:
                try:
                    #a['holistic'] = []
                    print('id',object_0[i]['id'])
                    print('lost',a['3DV_lost'])
                    print('3DV',a['3D_points'])
                    raznica = a['3DV_lost'] - a['3D_points'] #3ДВ в этом случаи это последнйи раз когда был трекнут лицо и его точки. 3ДВ лост это красный ББокс и его координаты
                    x = np.array(a['3D_holistic'][-3][0]) + raznica[0]
                    y = np.array(a['3D_holistic'][-3][1]) + raznica[1]
                    z = np.array(a['3D_holistic'][-3][2]) + raznica[2]
                    self.drow_cub_1(ax1,a,x,y,z,'Вид спереди',3,270,270)
                    self.drow_cub_1(ax2,a,x,y,z,'Вид сверху',2,0,270)
                except:
                    pass
                
    # Не понял
    def draw_prediction(self, frame1, p_, q01, q00, r):
        for i in range(len(self.object_)):
            q_ = min(r, len(self.predict[i]) - 1)

            couple = self.predict[i][q_]
            q1 = (int(couple[0] - 6), int(couple[1] - 8))
            q2 = (int(couple[0] + 6), int(couple[1] + 8))
            color = [int(c) for c in self.COLORS[self.object_[i]['id']]]
            cv2.rectangle(frame1, q1, q2, color, 1)

            cv2.circle(frame1, (int(p_[i][0]), int(p_[i][1])), 2, [255, 0, 100], 2)

            # path=np.array(self.object_[i]['history'])
            line_ = np.array([q00[i], q01[i]])
            drawPolylineOnImg1(frame1, line_, color, 1)

    ### обновляем PCA of гистограммы цветовой компоненты HUE вдоль истории глубиной в 10 кадров
    def PCA_update(self):
        for i_ob in range(len(self.object_)):
            l_ = len(self.object_[i_ob]['descriptor_'])
            if l_ > 1:

                r_ = [random.randint(0, l_ - 1) for i in range(min(l_, 10))]

                UU = np.array(self.object_[i_ob]['descriptor_'])[r_, :]
                # UU=np.array(self.object_[i_ob]['descriptor_'] )
                if len(UU) > 0:
                    try:
                        SU = eigen_fast(np.dot(UU.T, UU), min(l_, 4))
                        self.object_[i_ob]['descriptor_PCA'] = SU
                    except:
                        R = np.array(self.object_[i_ob]['descriptor_'][-1])
                        self.object_[i_ob]['descriptor_PCA'] = R / (0.000001 + np.linalg.norm(R))

    ### предсказываем траекторию объектов на базе истории
    def predict_10(self, frame1):
        for i in range(len(self.object_)):
            path = np.array(self.object_[i]['history'][-7:])
            w_123 = self.v_[-path.shape[0] - 2:-2]
            we_1 = lsqr(w_123, path[:, 1], atol=1e-6, btol=1e-6, iter_lim=10)[0]
            predict_1 = np.dot(self.v_[-1], we_1)

            we_0 = lsqr(w_123, path[:, 0], atol=1e-6, btol=1e-6, iter_lim=10)[0]
            predict_0 = np.dot(self.v_[-1], we_0)
            drawPolylineOnImg0(frame1, np.array(self.object_[i]['history']))
            if len(path) > 5:
                drawPolylineOnImg1(frame1, np.int32(
                    np.round([path[-1], [min(self.im_w, predict_0), min(self.im_h, predict_1)]])), (124, 155, 0), 2)


    # Нормализация траектории
    def normalize_trajectory(self, path0):
        return path0 / [self.im_w, self.im_h] - [0.5, 0.5]

    def de_normalize_trajectory(self, path0):
        return [self.im_w, self.im_h] * (path0 + [0.5, 0.5])

    def delete_small_obj(self):
        spisok = []
        for i_object in range(len(self.object_)):
            object_k = self.object_[i_object] # Параметры объекта
            if object_k['w'] < 75 or object_k['h'] < 75:
                object_k = None
            if object_k != None:
                spisok.append(object_k)
        return spisok

    # Нужна ли?
    def predict_03(self):
        self.predict = []
        for i in range(len(self.object_)):
            path = np.array(self.object_[i]['history'])

            trajectory_0 = self.normalize_trajectory(path)

            Y_hat = self.LSTM_Predictor.predict(trajectory_0)

            predicted = self.de_normalize_trajectory(Y_hat)

            if path.shape[0] >= self.l_predict:
                pred_ = predicted[-self.l_predict:, :] - predicted[-self.l_predict, :] + path[-1, :]
            else:
                pred_ = predicted[path.shape[0]:, :]  # -predicted[0,:]+path[-1,:]
            self.predict.append(pred_)

    def conditions_for_classify(self, kj, flag, show):
        if flag == 1:
            X0 = self.object_[kj]['init_posit']
            X1 = self.object_[kj]['history'][-1]
            x1 = X0[1] - self.im_h / 2
            x2 = X1[1] - self.im_h / 2
            x3 = abs(X1[1] - X0[1])
            q1 = x3 > self.im_h / 3  # не зацикленная траектория
            q2 = np.sign(x1 * x2) < 0  # пересекают середину по вертикали
            q3 = self.object_[kj]['N14'] > 5  # объект движется
            q4 = X0[0] > 50 or X0[0] < self.im_w - 50
            return (q1 and q2 and q3 and q4)
        elif flag == 2:
            p0 = np.sum(self.object_[kj]['owned']) / self.object_[kj][
                'age']  ###  сколько процентов объект был распознаваем
            return (p0 > self.percent_of_nrecognized_for_classity)

    # Классифицируем траектории
    def classyfy_all_trajectory(self):

        for kj in range(len(self.object_)):
            object_black = self.object_[kj]
            if self.InOut_Classifier.flag_type == 0:  # cnn
                num_ticks = 6
            else:
                num_ticks = 15
            if object_black['age'] > (num_ticks - 1):
                if self.conditions_for_classify(kj, 1, 0):  # объект подлежит классификации
                    tr_ = self.normalize_trajectory(np.array(object_black['history'])[-num_ticks:, :])
                    self.object_[kj]['class_trajectory'].append(self.InOut_Classifier.predict_traject_model_00(tr_))
                else:  # объект не движется
                    self.object_[kj]['class_trajectory'].append(0.5)

    # Классифицируем траектории
    def class_tr_00(self):
        object_0 = self.object_
        for i in range(len(object_0)):
            q_0 = np.mean(object_0[i]['class_trajectory'])
            delta = 0.001
            if q_0 < 0.5 - delta:
                object_0[i]['class_trajectory_general'] = -1
            elif q_0 > 0.5 + delta:
                object_0[i]['class_trajectory_general'] = 1

    # Классифицируем траектории
    def counter_class_tr_00(self, qqq_0, qqq_1):
        self.class_tr_00()
        for i in range(len(self.object_)):
            if self.object_[i]['age'] > qqq_1:

                if (self.conditions_for_classify(i, 2, 0)) and (self.object_[i]['class_trajectory_general'] == qqq_0):
                    self.id_negative_set = np.union1d(self.id_negative_set, self.object_[i]['id'])
                if (self.conditions_for_classify(i, 2, 0)) and (self.object_[i]['class_trajectory_general'] == -qqq_0):
                    self.id_negative_set = list(set(self.id_negative_set).difference(set([self.object_[i]['id']])))
        self.count_positive_traject = len(self.id_negative_set)

    # Вырезаем объекты в ббоксах и сохраняем их.
    def cut_and_save_objects(self, image, frame_counter, path_out):
        ### вырезаем объекты
        for i in range(len(self.object_)):
            obj_id = self.object_[i]['id']
            obj_path_out = path_out + 'class_{:03d}_id_{:03d}\\'.format(self.object_[i]['class'], obj_id)
            if not os.path.exists(obj_path_out):
                os.makedirs(obj_path_out)
            object_ = self.object_[i]
            if object_['owned'][-1] == 1:
                w_ = object_['w']
                h_ = object_['h']
                center_ = np.array(object_['history'][-1])
                beg_x = max(0, int(center_[0] - w_ / 2))
                end_x = int(center_[0] + w_ / 2)
                beg_y = max(0, int(center_[1] - h_ / 2))
                end_y = int(center_[1] + h_ / 2)
                ob_ = image[max(0, beg_y):end_y, max(0, beg_x):end_x, :]

                pref = 'fr_{:03d}_id_{:03d}'.format(frame_counter, self.object_[i]['id'])
                #cv2.imwrite(obj_path_out + pref + '.png', cv2.cvtColor(ob_, cv2.COLOR_BGR2RGB))

    def polynomial_regression_for_w(self,w):
        x = np.array([387,352,286,261,230,213,195,172,163,155,140,100,69,58,46,39,33,27])
        y = np.array([50,60,70,80,90,100,110,120,130,140,150,200,300,400,500,610,710,800])

        x = x[:, np.newaxis]
        y = y[:, np.newaxis]

        polynomial_features= PolynomialFeatures(degree=5)
        x_poly = polynomial_features.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)

        w_ = model.predict(polynomial_features.fit_transform([[w]]))

        return w_[0]

    def polynomial_regression_for_h(self,h):
        x = np.array([525,477,389,349,304,283,255,234,217,204,182,134,87,72,57,48,40,33])
        y = np.array([50,60,70,80,90,100,110,120,130,140,150,200,300,400,500,610,710,800])

        x = x[:, np.newaxis]
        y = y[:, np.newaxis]

        polynomial_features= PolynomialFeatures(degree=5)
        x_poly = polynomial_features.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)

        h_ = model.predict(polynomial_features.fit_transform([[h]]))

        return h_[0]

    def compinsation_for_one_points(self,points):
        result = []
        spisok = []
        count = (points.shape[0] * points.shape[1]) // 3
        for i in range(3):
            spisok_ = []
            index = 0
            while index < count:
                spisok_.append(points[index][i])
                index += 1
            spisok.append(spisok_)
        # Собрали в списки по уникальным точкам. Всего списков 3 т.к уникальных 3 точек
        for y in np.array(spisok):
            x = np.array(range(1,len(y)+1)).reshape((-1, 1))
            model = LinearRegression()
            model.fit(x, y)
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)
            summa = [y_pred,y]
            result.append(np.mean(summa))
        #Усреднили каждую точку через линейную регрессию
        mean_points = np.array_split(np.array(result),1) 
        #Заспилитили 3 точек в списки по 1 точки. Получили 33 списка. Так как всего 33 ключевых точки по 3 координатам

        return mean_points[0]
    
    def compinsation_for_points(self,holistic_points,compensaition_index):
        try:
            result_ = []
            spisok = []
            result = [] 
            #Проверяем есть ли значения None. Убираем если есть
            holistic_ = [x for x in holistic_points[-compensaition_index:] if x is not None]
            #Открываем список медиапайп. Забираем все без компонента видимость
            for i in holistic_:
                for j in i.landmark:
                    result_.append(j.x)
                    result_.append(j.y)
                    result_.append(j.z)
            count = len(result_) // 99
            split_result = np.array_split(np.array(result_), count) #Сплитим большой список с шагом 99
            for i in range(99):
                spisok_ = []
                index = 0
                while index < count:
                    spisok_.append(split_result[index][i])
                    index += 1
                spisok.append(spisok_)
            # Собрали в списки по уникальным точкам. Всего списков 99 т.к уникальных 99 точек
            for y in np.array(spisok):
                x = np.array(range(1,len(y)+1)).reshape((-1, 1))
                model = LinearRegression()
                model.fit(x, y)
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                summa = [y_pred,y]
                result.append(np.mean(summa))
            #Усреднили каждую точку через линейную регрессию
            mean_points = np.array_split(np.array(result),33) 
            #Заспилитили 99 точек в списки по 3 точки. Получили 33 списка. Так как всего 33 ключевых точки по 3 координатам

            return mean_points
        except:
            print('Пустой скелет')
            return []

# Начальные данные
def kvant_state_init():
    return {'age' : 0,'holistic_age': 0,'age_lost' : 0,'id' :[] ,'class': [],  'history':[],'confidences':[],\
                               'history_X':[],'owned':[1],'holistic':[],'3DV':[],'3DV_lost':[],'3D_points':[],\
                               'descriptor_':[],'descriptor_PCA':[],'w':0,'h':0,'depth':[],'3D_holistic':[],\
            'class_trajectory': [0.5], 'class_trajectory_general': 0, 'radius': 10, 'N14': 10, 'init_posit':[]} 

# Предиктор. Вроде не используем.
class LSTM_Predictor:

    def __init__(self, l_predict, len_ts, path_to_w, flag_):

        self.l_predict = l_predict  ### 10 на сколько мы предсказываем траекторию
        self.len_ts = len_ts  ### 50 сколько минимум отсчетов траектории в истории необходимо для предсказания
        self.flag_ = flag_  ### доверяем lstm или нет
        self.cannals_hist = 0  ### 0- гистограмма hue для идентификации объекта, 2- гистограмма яркости
        hunits = 64
        if self.flag_:
            model_stateless, _ = define_model(
                hidden_neurons=hunits,
                len_ts=self.len_ts)
            model_stateless.summary()
            start = time.time()
            if path_to_w is not None:
                model_stateless.load_weights(path_to_w)
            end = time.time()
            print("Time Took :{:3.2f} min".format((end - start) / 60))
            self.model_stateless = model_stateless

    def predict(self, traject_D2):
        l_ = traject_D2.shape[0]
        if (l_ > self.len_ts) and self.flag_:
            Y_hat = self.model_stateless.predict(np.array([traject_D2[-self.len_ts:, :]]))[0]
            predicted = np.vstack([traject_D2, Y_hat[-self.l_predict:, :]])
        else:
            traject_D3 = traject_D2[-self.l_predict:, :]
            traject_D4=traject_D3.copy()
            for i in range(1,len(traject_D3)):
                traject_D4[i,:]=np.mean(traject_D3[:i,:],axis=0)
            predicted = np.vstack([traject_D2, traject_D4[-1, :] + traject_D3[-1, :] - traject_D4[::-1, :]])
        return predicted


def define_model(len_ts,
                 hidden_neurons=10,
                 nfeature=2,
                 batch_size=None,
                 stateful=False):
    
    in_out_neurons = 2

    inp = layers.Input(batch_shape=(batch_size, len_ts, nfeature),
                    name="input")

    rnn = layers.LSTM(hidden_neurons,
                    return_sequences=True,
                    stateful=stateful,
                    name="RNN")(inp)

    dens = layers.TimeDistributed(layers.Dense(in_out_neurons, name="dense"))(rnn)
    model = models.Model(inputs=[inp], outputs=[dens])
    #model = None


    model.compile(loss="mean_squared_error",
                sample_weight_mode="temporal",
                optimizer="rmsprop")
    return (model, (inp, rnn, dens))

 
# Классификатор объектов
class InOut_Classifier:

    def __init__(self, path_to_w, flag_type):
        #flag_type -  классификатор траекторий: 0 - CNN, 1 - lstm, 2- mll, 3 - simple classify
        #path_to_w - путь к весам классификатора траекторий
        if flag_type == 0: #CNN
            model = model_2conv_2Dence_level()
        elif flag_type == 1: #LSTM
            model = lstm_model()
        elif flag_type == 2: #MLL
            model = mll_model()        
        else:# simple
            model = []

        
        if (flag_type < 3) and (path_to_w is not None):
            model.compile(loss='mean_squared_error',
                      ####loss='logcosh','categorical_crossentropy',loss='binary_crossentropy'
                      optimizer='adam',
                      metrics=['mse'])  ### mse усредненная norm( ypred-ytrue);
            ### accuracy,  означает долю правильно предсказанных классов в общем количестве всех предсказаний
            model.load_weights(path_to_w)

        self.model_classif = model
        self.flag_type = flag_type

    def predict_traject_model_00(self, X0):
        X1 = np.expand_dims(X0, 2)
        if self.flag_type == 3:
            y1 = int(X0[-1,1]-X0[0,1]<0) # vertical up-down
            return y1
        elif self.flag_type == 4:
            y1 = int(X0[-1,0]-X0[0,0]<0) # horisontal left-right
            return y1
        elif self.flag_type ==0: #CNN
            X2 = X0.reshape((1, X1.shape[0], 2, 1))
        else:
            X2 = X0.reshape((1, X1.shape[0], 2))
        y1 = np.squeeze(self.model_classif.predict(X2), 0)
        return y1

    ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

########################## Вспомогательные функции.
def fun_865(x):
    x1=max(0,x-3)
    x2=np.power(x1,2)
    x3=min(x2,160)
    return x3

def one_to_one_relation_01(relation_):
    X_scaled = relation_ / (0.0001 + np.linalg.norm(relation_, axis=1, keepdims=True))
    correspond_ = []
    while (1):

        i_, j_, ma_ = amax_2D(X_scaled)
        if ma_ < 0:
            break
        correspond_.append([i_, j_])
        X_scaled[i_, :] = -0.0001
        X_scaled[:, j_] = -0.0001

    correspond_0 = np.array(correspond_)
    return correspond_0

# Рисует линию предикта.
def drawPolylineOnImg1(img, points, COL_, TH):

    points[:, 0] = (points[:, 0]).astype(int)
    points[:, 1] = (points[:, 1]).astype(int)
    points = np.array(points, dtype=np.int32)

    cv2.polylines(img, [points], 0, COL_, TH)

def drawPolylineOnImg0(img, points):

    points[:, 0] = (points[:, 0]).astype(int)
    points[:, 1] = (points[:, 1]).astype(int)
    points = np.array(points, dtype=np.int32)

    cv2.polylines(img, [points], 0, (0, 255, 255), 2)

def object_featre_descriptor(imageIn, XXX, dscr_TL_00, show_, LABELS, known_vectors = None):
    collect_color_featres = []
    for i_ob in range(len(XXX)):
        ob_ = imageIn[max(0, XXX[i_ob]['bords'][1]):XXX[i_ob]['bords'][3],
              max(0, XXX[i_ob]['bords'][0]):XXX[i_ob]['bords'][2]]
        if known_vectors is not None:
            hs = dscr_TL_00.descriptor_one_object_00(ob_, known_vectors[i_ob])

        collect_color_featres.append(hs)
    return collect_color_featres

def cos_distance_multi(hh_1d, hh_list):
        return np.array([1-cosine (hh_1d, x) for x in hh_list])


def check_points_02(bords, points, r_, cur_class, classes):
    # Проверка наличия точки(-ек) внутри бокса  или близко с ним
    zenter_box = [(bords[2] + bords[0]) / 2, (bords[3] + bords[1]) / 2]
    q = False
    for pt_, cl_ in zip(points, classes):
        x, y = pt_
        #Проверка на детект. Есть ли лицо.
        if (cl_ == cur_class):
            if x < bords[2] and x > bords[0] and y < bords[3] and y > bords[1]:
                q = True
                break
        if dist([x, y], zenter_box) < r_:
            q = True
            break
    return q

def dist(p1, p2):
    # Расчет геом расстояния между двумя точками
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def amax_2D(X_scaled):
    i_sort_relation_ = np.argmax(X_scaled.ravel(), axis=-1)
    i_amax = int(np.fix(i_sort_relation_ / X_scaled.shape[1]))
    j_amax = int(i_sort_relation_ - i_amax * X_scaled.shape[1])
    return i_amax, j_amax, X_scaled[i_amax, j_amax]
