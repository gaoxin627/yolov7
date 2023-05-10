import torch
import cv2
import time
import numpy as np
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


class Detector(object):
    def __init__(self, model_path, img_size, class_dict, conf_thres, iou_thres):
        self.class_dict = class_dict
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        self.model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
        self.half = self.device.type != 'cpu'
        self.stride = max(int(self.model.stride.max()), 32)
        self.img_size = check_img_size(self.img_size, s=self.stride)
        if self.half:
            self.model.half()

    def detect(self, img_path, is_auto):
        img0 = cv2.imread(img_path)
        img = letterbox(img0, self.img_size, auto=is_auto, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        result_info_list = []
        for box_info in det.cpu().tolist():
            box = [int(point) for point in box_info[:4]]
            conf = box_info[4]
            id = int(box_info[5])
            result_info_list.append([box, conf, self.class_dict[id]])
        return result_info_list


def parse_params(input_dict):
    # request_id = None
    # img_url = None
    label_set = None
    # if 'requestId' in params:
    #     request_id = params['requestId']
    if 'params' in input_dict:
        params_dict = input_dict['params']
        # if 'imageUrl' in params_dict:
        #     img_url = params_dict['imageUrl']
        if 'label' in params_dict:
            label_str = params_dict['label']
            label_set = set(label_str.split(','))
    # return request_id, img_url, label_set
    return label_set


def filt_result(result_info_list, label_set):
    result_info_list_filted = []
    for box_info in result_info_list:
        label = box_info[2]
        if label in label_set:
            result_info_list_filted.append(box_info)
    return result_info_list_filted


def convert_result(result_info_list):
    result_info_list_converted = []
    for box_info in result_info_list:
        box = box_info[0]
        conf = box_info[1]
        label = box_info[2]
        box_info_dict = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3], 'score': conf, 'value': '1', 'comment': label}
        result_info_list_converted.append(box_info_dict)
    return {'output': result_info_list_converted}


class DetectorJYZPL(object):
    def __init__(self, model_path = '/home/gaoxin/model/yolov7/jyz_pl_v2/jyz_pl_v2_yolov7_map05_400/best_152.pt'):
        img_size = 640
        class_dict = {0: 'jyz', 1: 'jyz_pl'}
        conf_thres = 0.5
        iou_thres = 0.65
        self.detector = Detector(model_path, img_size, class_dict, conf_thres, iou_thres)
        self.is_atuo = True

    def detect(self, img_path, params):
        try:
            label_set = parse_params(params)
            result_info_list = self.detector.detect(img_path, self.is_atuo)
            result_info_list_filted = filt_result(result_info_list, label_set)
            result_info_converted = convert_result(result_info_list_filted)
            return result_info_converted
        except Exception as e:
            return {'output': []}


if __name__ == '__main__':
    detector = DetectorJYZPL()
    img_path = '/home/gaoxin/data/train_val/jyz_pl_v2/test_images/DSC02018_JPG.rf.9099096b7e22348111f259693192647d.jpg'
    params = {'params': {'label': 'jyz_pl,jyz'}}
    start_time = time.time()
    result = detector.detect(img_path, params)
    end_time = time.time()
    print(result)
    print(end_time - start_time)

    img = cv2.imread(img_path)
    h, w, c = img.shape
    color_dict = {'jyz': (0, 255, 0), 'jyz_pl': (0, 0, 255)}
    resize_rate = min(1, 800 / h)
    img_resize = cv2.resize(img, None, fx=resize_rate, fy=resize_rate)
    for box_info in result['output']:
        x1 = int(box_info['x1'] * resize_rate)
        y1 = int(box_info['y1'] * resize_rate)
        x2 = int(box_info['x2'] * resize_rate)
        y2 = int(box_info['y2'] * resize_rate)
        label = box_info['comment']
        cv2.line(img_resize, (x1, y1), (x2, y1), color=color_dict[label], thickness=2)
        cv2.line(img_resize, (x2, y1), (x2, y2), color=color_dict[label], thickness=2)
        cv2.line(img_resize, (x2, y2), (x1, y2), color=color_dict[label], thickness=2)
        cv2.line(img_resize, (x1, y2), (x1, y1), color=color_dict[label], thickness=2)
    cv2.imshow('result', img_resize)
    cv2.waitKey()
