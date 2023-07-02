import os
import sys
import torch
import cv2
import time
import numpy as np
import json
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.experimental import attempt_load


def load_img(img_file, img_size, is_auto, stride):
    img0 = cv2.imread(img_file)
    if img0 is None:
        return None, None
    img = letterbox(img0, img_size, auto=is_auto, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img0, img


def convert_result(result_info_list):
    result_info_list_converted = []
    for box_info in result_info_list:
        box = box_info[0]
        conf = box_info[1]
        label = box_info[2]
        box_info_dict = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3], 'score': conf, 'value': '1', 'comment': label}
        result_info_list_converted.append(box_info_dict)
    return {'output': result_info_list_converted}


def write_json(data, json_file):
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


class Detector(object):
    def __init__(self, model_path, cfg_file, img_size, class_dict, device=None):
        self.class_dict = class_dict
        self.img_size = img_size
        # self.conf_thres = conf_thres
        # self.iou_thres = iou_thres
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        # self.model = attempt_load(model_path, map_location=self.device)
        self.model = Model(cfg=cfg_file)
        checkpoint = torch.load(model_path)
        # for key in checkpoint:
        #     print(key, checkpoint[key].shape)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.half = self.device.type != 'cpu'
        self.stride = max(int(self.model.stride.max()), 32)
        self.img_size = check_img_size(self.img_size, s=self.stride)
        if self.half:
            self.model.half()

    def detect(self, img0, img, conf_thres, iou_thres):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        result_info_list = []
        for box_info in det.cpu().tolist():
            box = [int(point) for point in box_info[:4]]
            conf = box_info[4]
            id = int(box_info[5])
            if id in self.class_dict:
                result_info_list.append([box, conf, self.class_dict[id]])
        return result_info_list

    def test(self, img_path, is_auto, conf_thres, iou_thres, result_file):
        start_time = time.time()
        result_list = []
        sum_time_load = 0
        sum_time_predict = 0
        img_num = 0
        predict_num = 0
        for file_name in os.listdir(img_path):
            if file_name.lower().endswith('jpg'):
                img_num += 1
                if img_num % 100 == 0:
                    print(img_num)
                    sys.stdout.flush()
                img_file = os.path.join(img_path, file_name)
                start_time_load = time.time()
                img0, img = load_img(img_file, self.img_size, is_auto, self.stride)
                end_time_load = time.time()
                load_time = end_time_load - start_time_load
                sum_time_load += load_time
                if img0 is None:
                    print(file_name)
                else:
                    predict_num += 1
                    result = self.detect(img0, img, conf_thres, iou_thres)
                    end_time_predict = time.time()
                    result_converted = convert_result(result)
                    result_converted['filename'] = file_name
                    predict_time = end_time_predict - end_time_load
                    sum_time_predict += predict_time
                    result_list.append(result_converted)
                    # print(load_time, predict_time, result_converted)
        end_time = time.time()
        all_time = end_time - start_time
        result_path = os.path.dirname(result_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        write_json(result_list, result_file)
        print(img_num, predict_num, all_time, sum_time_load, sum_time_predict)
        print(all_time / img_num, sum_time_load / img_num, sum_time_predict / predict_num)


if __name__ == '__main__':
    model_path = r'D:\model\yolov7\nr_train1_23\nr_train1_23_yolov7_best_288_7774.pt'
    cfg_file = r'D:\workspace\python\yolov7\cfg\training\yolov7_23.yaml'
    # class_dict = {0: 'bj_bpmh', 1: 'bj_bpps', 2: 'bj_wkps', 3: 'bjdsyc', 4:'jyz_pl', 5: 'sly_dmyw', 6: 'hxq_gjtps',
    #               7: 'hxq_gjbs', 8: 'ywzt_yfyc', 9: 'xmbhyc', 10: 'yw_gkxfw', 11: 'yw_nc', 12: 'gbps', 13: 'wcaqm',
    #               14: 'wcgz', 15: 'xy', 16: 'kgg_ybh'}
    class_dict = {0: 'bj_bpmh', 1: 'bj_bpps', 2: 'bj_wkps', 3: 'bj_zz', 4: 'jyz_pl', 5: 'sly_dmyw', 6: 'hxq_gjtps',
                  7: 'hxq_gjbs', 8: 'ywzt_yf', 9: 'xmbhyc', 10: 'yw_gkxfw', 11: 'yw_nc', 12: 'gbps', 13: 'wcaqm',
                  14: 'wcgz', 15: 'xy', 16: 'kgg_ybh', 17:'bj_sx', 18: 'hxq_gjzc', 19: 'xmbhzc', 20: 'aqmzc',
                  21: 'gzzc', 22: 'kgg_ybf'}
    img_size = 640
    is_auto = True
    conf_thres = 0.01
    iou_thres = 0.65
    start_time = time.time()
    detector = Detector(model_path, cfg_file, img_size, class_dict)
    end_time = time.time()
    init_time = end_time - start_time
    print(init_time)

    # img_path = r'D:\data\test\merge_0601\test50'
    img_path = r'D:\data\test\merge_0601\test_500\images'
    is_atuo = True
    out_file = r'D:\result\merge_0601\result_test_nr_23'
    detector.test(img_path, is_atuo, conf_thres, iou_thres, out_file)