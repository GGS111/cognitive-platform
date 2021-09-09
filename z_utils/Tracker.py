from z_utils.utils_015 import *
import cv2
import matplotlib.pyplot as plt

USE_NETWORK = 0
USE_CADET = 1

class Tracker:
    def __init__(self, _cap, _video_out, list_params):
        self.list_params = list_params
        self.cap = _cap
        self.video_out = _video_out
        self.pts = None
        self.q01 = None
        self.status = None
        self.error = None
        self.XXX = None
        self.q00 = None
        self.q02 = None
        self.q0 = None
        self.q03 = None
        self.collect_featres_k = None
        self.pts_from_cnn = None
        self.pts_00 = None
        self.classes_00 = None
        self.old_gray = None
        self.encodings = None

        #Загрузка всех параметров
        self.diffuse_map = list_params['diffuse_map']

        self.path_out = list_params['path_out']
        self.flag_debug = list_params['flag_debug']
        self.input_type_flag = list_params['input_type_flag']
        self.cannals_hist = list_params['cannals_hist']
        self.flag_classify_traj = list_params['flag_classify_traj']
        self.path_to_w_classify_traj = list_params['path_to_w_classify_traj']
        self.flag_prediction = list_params['flag_prediction']
        self.path_to_prediction = list_params['path_to_prediction']
        self.thr_color_features = list_params['thr_color_features']
        self.check_period = list_params['check_period']
        self.l_forget_history = list_params['l_forget_history']
        self.manual_thr_radius = list_params['manual_thr_radius']

        self.minsize = list_params['minsize']  # minimum size of face
        self.threshold = list_params['threshold']  # three steps's threshold
        self.factor = list_params['factor']  # scale factor 0.709
        self.margin = list_params['margin']

        self.thr_measure_hue_optflow2predict = list_params['thr_measure_hue_optflow2predict']
        self.thr_add_new = list_params['thr_add_new']
        self.thr_correct_opt_flow = list_params['thr_correct_opt_flow']
        self.prefix_ind_save_0 = list_params['prefix_ind_save_0']
        self.start_ind = list_params['start_ind']
        self.class_to_interest = list_params['class_to_interest']
        self.flag_cut_and_save_objects = list_params['flag_cut_and_save_objects']
        self.flag_save_tr_mat = list_params['flag_save_tr_mat']
        self.debug_count = list_params['debug_count']
        self.predictor_type = list_params[
            'predictor_type']  # 0 - yolo, 1 - detectron2 (cadet), 2 - detectron2 (pedet), 3 - facenet (?)
        self.detector_01 = list_params['detector_01']  # основной,   predictor_heads
        self.detector_02 = list_params['detector_02']  # второстепенный
        self.dscr_TL_00 = list_params['dscr_TL_00']  # дескриптор объекта
        self.double_detect_flag = list_params['double_detect_flag']
        self.p_opt_flow = list_params['p_opt_flow']
        self.border_edge = list_params['border_edge']
        self.r_edge = list_params['r_edge']
        self.DM_save = list_params['DM_save']
        self.path_out_for_DM_images = list_params['path_out_for_DM_images']
        self.show_diffusion = list_params['show_diffusion']
        self.show_3Dpoints = list_params['show_3Dpoints']
        self.video_adapter = list_params['video_adapter']
        self.add_holistic = list_params['add_holistic']
        self.quality_width = list_params['quality_width']
        self.resize_coef = list_params['resize_coef']
        self.compensaition_index = list_params['compensaition_index']
        try:
            self.percent_of_unrecognized_for_classity = list_params['percent_of_unrecognized_for_classity']
        except:
            self.percent_of_unrecognized_for_classity = 0.5

        try:
            self.delete_temp = list_params['delete_temp']
        except:
            self.delete_temp = 0
        self.lk_params = dict(winSize=(45, 45),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # YOLO configs

    ###Загружаем данные. Загружаем стрим через self.cap.isOpened()
    def InitTracker(self):
        print('start_ind   ', self.start_ind)
        self.frame_counter = 0


        print(self.cap.isOpened())
        assert self.cap.isOpened()

        while self.cap.isOpened(): 
            ret, frame = self.cap.read()
            print('был',frame.shape[1], frame.shape[0])
            #resize_coef = frame.shape[1]/frame.shape[0]
            
            quality_height = int(self.quality_width/self.resize_coef)

            frame = cv2.resize(frame, (self.quality_width,quality_height))
            print('стал',frame.shape[1], frame.shape[0])
            self.frame_counter += 1
            if self.frame_counter > self.start_ind:
                break
        # Инициализация параметров
        self.TrackObjects = init_000(self.flag_prediction, self.cap, self.detector_01, self.list_params,
                                     self.detector_01.LABELS,
                                     self.detector_01.COLORS_
                                     )

        #self.TrackObjects.p_opt_flow = self.p_opt_flow
        self.TrackObjects.dscr_TL_00 = self.dscr_TL_00
        self.TrackObjects.percent_of_nrecognized_for_classity = self.percent_of_unrecognized_for_classity
        self.TrackObjects.tracker = self


        if self.TrackObjects is not None:
            
            self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_counter = self.start_ind
            #####################################################################

            w = self.TrackObjects.im_w
            h = self.TrackObjects.im_h
            print(w, h)
            self.out = cv2.VideoWriter(self.video_out, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (w, h))
            frame_counter = 0

            self.pnet = self.rnet = self.onet = None

    #Трекаем Имейдж
    def TrackImage(self):
        self.frame_counter += 1
        ret, frame = self.cap.read() 
        #Считываем фреймы. Если Тру то прочитан
        if (not ret) or ret < 0:
            return None
        #resize_coef = frame.shape[1]/frame.shape[0]
        quality_height = int(self.quality_width/self.resize_coef)
        frame = cv2.resize(frame, (self.quality_width,quality_height))
            
        self.TrackObjects.imageIn = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #plt.imshow('image',self.TrackObjects.imageIn)
        #plt.show()
        #self.TrackObjects - Просто Картинка из фреймов выше

        try:
            gray_frame = cv2.cvtColor(self.TrackObjects.imageIn, cv2.COLOR_BGR2GRAY)
        except:
            return None


        ### центры объектов
        self.pts = object2pts(self.TrackObjects.object_)
        # Получаем центры каждого лица + Его класс + Его Айди
        ### двигаем центры объектов  по    OpticalFlow

        #Если трекнули имейдж. А именно имеем w,h, то:
        if len(self.pts['coords']) > 0:

            self.q01, self.status, self.error = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.pts['coords'], None, **self.lk_params)
            ### корректируем опт флоу  по аномальному смещению. разница с предсказанием по регрессии не должна сильно отличаться
            self.q00 = self.pts['coords'].copy()
            self.q02 = self.q01.copy()
            self.TrackObjects.predict_03()
            self.q0 = correct_opt_flow_02(self.q01, self.TrackObjects.object_, self.TrackObjects.predict, 2, self.thr_correct_opt_flow)
            self.q03 = self.q0.copy()
            self.pts['coords'] = self.q0

        #После куска выше более точно детектит центр лица.
        self.old_gray = gray_frame.copy()

        ### смотрим что дает нейросеть, какие коробки

        self.XXX = frame1 = None
        # Вводим доп. данные. Нам интересен frame1.
        if self.predictor_type == 0:  # YOLO
            self.XXX, frame1 = yolo_application_013(self.detector_01, self.TrackObjects.imageIn, self.class_to_interest)
            if self.double_detect_flag == 1:
                X12, frame1 = yolo_application_014(self.detector_02, self.TrackObjects.imageIn, frame1, np.array(range(200)))
        elif self.predictor_type == 1:  # detectron2 (cadet)
            self.XXX, frame1 = detectron_application(self.detector_01, self.TrackObjects.imageIn, ind_classes=self.class_to_interest)
        elif self.predictor_type == 2:  # detectron2 (pedet)
            self.XXX, frame1 = detectron_application(self.detector_01, self.TrackObjects.imageIn, ind_classes=self.class_to_interest)
        elif self.predictor_type == 3:  # face_detection
            LABELS = ['face0', 'face1', 'face2', 'face3', 'face4']
            np.random.seed(42)
            COLORS_ = np.random.randint(100, 215, size=(200, 3), dtype="uint8")
            if USE_FACE_DETECTION:
                self.XXX, frame1, self.encodings = face_detection_application(self.TrackObjects.imageIn, COLORS_, LABELS, self.minsize, self.threshold, self.factor, self.margin, self.pnet,
                                                                              self.rnet, self.onet)
                self.collect_featres_k = self.encodings
            else:
                self.XXX, frame1 = face_detection_application(self.TrackObjects.imageIn, COLORS_, LABELS, self.minsize, self.threshold, self.factor, self.margin, self.pnet,
                                                              self.rnet, self.onet)

                self.collect_featres_k = self.object_featre_descriptor(self.TrackObjects.imageIn, self.XXX, self.encodings, 0, self.detector_01.LABELS)
        # Тут происходит детект лица. Рисуется серый квадрат.
        elif self.predictor_type == 4:  # facenet_pytorch
            self.XXX, frame1, self.encodings = facenet_pytorch_application(self.detector_01, self.TrackObjects.imageIn)
            try:
                self.collect_featres_k = self.encodings
                self.collect_confiedense = []
                for x in self.XXX:
                    self.collect_confiedense.append(x['confidences'])
            except:
                pass

            #print((self.collect_featres_k)[0].shape) - На каждую голову свой вектор. Сайзинг 512
            #print(self.XXX[0]['confidences']) # Пулл данных для отрисовки квадрата. w,h + центр
            #print(frame1) - Имейдж в виде тензора
        else:
            print('Wrong predictor type')
            return

        # Получаем пулл данных о задетекченном человеке.
        self.pts_from_cnn = X2pts_01(self.XXX)


        # для каждого объекта ищем подходящую коробку. она или есть или ее нет
        # корректируем объекты исходя из коробок порожденных сетью. дополняем историю


        #ВАЖНЫЙ МОМЕНТ: Для всех объектов что есть в памяти. Проверяет по тому есть координаты или нет. Делает функцию апдейта.
        if len(self.pts['coords']) > 0:
            self.TrackObjects.Update_04(self.pts,self.p_opt_flow, self.collect_featres_k,self.collect_confiedense, self.pts_from_cnn,
                                        self.thr_color_features, self.thr_measure_hue_optflow2predict,
                                        self.frame_counter, self.debug_count, self.manual_thr_radius, self.detector_01)
            #self.TrackObjects.occlusion_movement()
            #self.TrackObjects.predict_10(frame1)
            if self.add_holistic == 1:
                self.TrackObjects.holistic_model(frame1)
                #self.TrackObjects.transate_landmarks_to_3d(frame1)



        ### рисуем центры объектов
        

        #Не очень понял, что вообще рисует. Если комитить ничего не меняется.
        if len(self.pts['coords']) > 0:
            self.TrackObjects.draw_prediction(frame1, self.q03, self.q02, self.q00, 2)

        ### обрезаем далекое прошлое
        if self.frame_counter % self.check_period == 0:
            self.TrackObjects.Forget_horizont_vert(self.l_forget_history)
        ### смотрим не появился ли новый объект
        if self.frame_counter % 1 == 0:
            self.pts_00 = [x_['history'][-1] for x_ in self.TrackObjects.object_]
            self.classes_00 = [x_['class'] for x_ in self.TrackObjects.object_]
            #print('self.pts_00',self.pts_00)

            self.r_edge = 1
            self.border_edge = 5
            self.TrackObjects.add_new_object(self.pts_00, self.classes_00, self.XXX, self.thr_add_new, self.border_edge, self.r_edge, self.class_to_interest,
                                           frame1, self.collect_featres_k)



        ### классифицируем траектории
        if self.frame_counter % 2 == 0:
            self.TrackObjects.classyfy_all_trajectory()
        if self.frame_counter % 5 == 0:
            self.TrackObjects.counter_class_tr_00(-1, 10)

        ######  рисуем id, ббоксы
        self.TrackObjects.Draw(frame1, self.frame_counter)


        ### обновляем PCA of гистограммы цветовой компоненты HUE вдоль истории глубиной в 10 кадров
        if self.frame_counter % 2000 == 0:
            self.TrackObjects.PCA_update()
        
        #self.TrackObjects.DrowLSTMpred_00(frame1, self.class_to_interest) #Рисует линию траектории.


        ### вырезаем объекты
        if self.flag_cut_and_save_objects:
            self.TrackObjects.cut_and_save_objects(self.TrackObjects000.imageIn, self.frame_counter, self.path_out)


        ### сохраняем результат
        pref = str(self.frame_counter + self.prefix_ind_save_0)
        if self.flag_debug:
            cv2.imwrite(self.path_out + pref + '.png', cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        if self.frame_counter == 1:
            s_ = self.video_out.split('.mp4')[0]
            cv2.imwrite(s_ + '.png', cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            print('first frame saved on: ', s_ + '.png')

        self.out.write(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        if self.frame_counter > 0:

            ### вывод результата на экран
            t_res = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            cv2.imshow("preview", cv2.resize(t_res, (1280, 960)))

        ### сохраняем траектории
        if (self.frame_counter % 2 == 0) and self.flag_save_tr_mat:
            for kj in range(len(self.TrackObjects.object_)):
                object_black = self.TrackObjects.object_[kj]
                if object_black['age'] > 5:
                    ob_name = 'k_' + str(self.frame_counter + 10000000) + '_id_' + str(
                        object_black['id'] + 100000) + '.mat'
                    tr_ = self.TrackObjects.normalize_trajectory_00(np.array(object_black['history']))
                    io.savemat(self.path_out + ob_name, {"trajectory": tr_})

        if self.show_diffusion:

            vectors = []
            labels = []
            for obj in self.TrackObjects.object_:
                for v in obj['descriptor_']:
                    vectors.append(v)
                    labels.append(obj['id'])

                pass
            self.diffuse_map.vizu_with_labels(vectors, list(labels), None, frame1, self.path_out_for_DM_images)
            
        if self.show_3Dpoints:
            self.TrackObjects.transate_landmarks_to_3d(frame1,self.compensaition_index,self.collect_confiedense)


        return self.TrackObjects.object_



    #Сохраняем видео и закрываем показ кадров
    def ShutdownTracker(self):
        print('gg')
        ##########################################################################3
        if self.delete_temp:
            os.remove(self.cap.path)
        self.cap.release()
        self.out.release()
        cv2.destroyWindow("preview")
