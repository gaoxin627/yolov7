import os
import multiprocessing
import time
from inference import load_img, Detector


file_list_queue_length = 20000
img_queue_length = 1000
load_img_process_num = 8
detect_process_num = 4

model_path = r'D:\model\yolov7\merge_0601\0.841.pt'
class_dict = {0: 'bj_bpmh', 1: 'bj_bpps', 2: 'bj_wkps', 3: 'bjdsyc', 4: 'jyz_pl', 5: 'sly_dmyw', 6: 'hxq_gjtps',
              7: 'hxq_gjbs', 8: 'ywzt_yfyc', 9: 'xmbhyc', 10: 'yw_gkxfw', 11: 'yw_nc', 12: 'gbps', 13: 'wcaqm',
              14: 'wcgz', 15: 'xy', 16: 'kgg_ybh'}
img_size = 640
is_auto = True
conf_thres = 0.01
iou_thres = 0.65
stride = 32


def load_data(file_list_queue, img_queue):
    while True:
        file_path = file_list_queue.get()
        img0, img = load_img(file_path, img_size, is_auto, stride)
        img_queue.put((os.path.basename(file_path), img0, img))


def detect(img_queue, result_queue):
    start_time = time.time()
    detector = Detector(model_path, img_size, class_dict, 'cpu')
    end_time = time.time()
    init_time = end_time - start_time
    print('init finished, init time:', init_time)

    while True:
        file_name, img0, img = img_queue.get()
        result = detector.detect(img0, img, conf_thres, iou_thres)
        result_queue.put((file_name, result))


if __name__ == '__main__':
    # img_path = r'D:\data\test\merge_0601\test50'
    img_path = r'D:\data\test\merge_0601\test_500\images'

    start_time = time.time()

    file_list_queue = multiprocessing.Queue(file_list_queue_length)
    img_queue = multiprocessing.Queue(img_queue_length)
    result_queue = multiprocessing.Queue(file_list_queue_length)

    process_list = []
    for i in range(load_img_process_num):
        process = multiprocessing.Process(target=load_data, args=(file_list_queue, img_queue))
        process.start()
        process_list.append(process)
    for i in range(detect_process_num):
        process = multiprocessing.Process(target=detect, args=(img_queue, result_queue))
        process.start()
        process_list.append(process)

    file_num = 0
    for file_name in os.listdir(img_path):
        if file_name.endswith('jpg'):
            file_list_queue.put(os.path.join(img_path, file_name))
            file_num += 1
    print('file num:', file_num)

    while result_queue.qsize() < file_num:
        time.sleep(0.1)

    for process in process_list:
        process.terminate()

    end_time = time.time()
    test_time = end_time - start_time
    print(test_time, test_time / file_num)