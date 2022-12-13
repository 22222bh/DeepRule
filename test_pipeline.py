#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse

import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBar
from RuleGroup.LineQuiry import GroupQuiry
from RuleGroup.LIneMatch import GroupLine
from RuleGroup.Pie import GroupPie
import math
from PIL import Image, ImageDraw, ImageFont
torch.backends.cudnn.benchmark = False
import requests
import time
import re
from PIL import Image
import pytesseract
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", dest="cfg_file", help="config file", default="CornerNetLine", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=50000, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument('--cache_path', dest="cache_path", type=str)
    parser.add_argument('--result_path', dest="result_path", type=str)
    parser.add_argument('--tar_data_path', dest="tar_data_path", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_dir", dest="data_dir", default="data/linedata", type=str)
    parser.add_argument("--image_dir", dest="image_dir", default="C:/work/linedata/line/images/test2019/f4b5dac780890c2ca9f43c3fe4cc991a_d3d3LmVwc2lsb24uaW5zZWUuZnIJMTk1LjEwMS4yNTEuMTM2.xls-3-0.png", type=str)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
def load_net(testiter, cfg_name, data_dir, cache_dir, result_dir, cuda_id=0):

    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = result_dir
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def Pre_load_nets():
    methods = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", "annotation", "data/clsdata(1031)/cache",
                                "clsdata/result")

    from testfile.test_line_cls_pure_real import testing
    path = 'testfile.test_%s' % "CornerNetCls"
    testing_cls = importlib.import_module(path).testing
    methods['Cls'] = [db_cls, nnet_cls, testing_cls]
    db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", "annotation", "data/bardata(1031)/cache",
                                "bardata/result")
    path = 'testfile.test_%s' % "CornerNetPureBar"
    testing_bar = importlib.import_module(path).testing
    methods['Bar'] = [db_bar, nnet_bar, testing_bar]
    db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", "annotation", "data/piedata(1008)/cache",
                                "piedata/result")
    path = 'testfile.test_%s' % "CornerNetPurePie"
    testing_pie = importlib.import_module(path).testing
    methods['Pie'] = [db_pie, nnet_pie, testing_pie]
    db_line, nnet_line = load_net(50000, "CornerNetLine", "annotation", "data/linedata(1028)/cache",
                                  "linedata/result")
    path = 'testfile.test_%s' % "CornerNetLine"
    testing_line = importlib.import_module(path).testing
    methods['Line'] = [db_line, nnet_line, testing_line]
    db_line_cls, nnet_line_cls = load_net(20000, "CornerNetLineClsReal", "annotation",
                                          "data/linedata(1028)/cache",
                                          "linedata/result")
    path = 'testfile.test_%s' % "CornerNetLineCls"
    testing_line_cls = importlib.import_module(path).testing
    methods['LineCls'] = [db_line_cls, nnet_line_cls, testing_line_cls]
    return methods
methods = Pre_load_nets()


def ocr_result(image_path):
    '''
    The key output of this function is the bounding box of the words and the str version of the words. 
    E.g., word_info["text"]='Hello', word_info["boundingBox"] = [1, 2, 67, 78] 
    The boudningBox is the topleft_x, topleft_y, bottomleft_x, bottomlef_y.
    '''

    ocr_res = pytesseract.image_to_boxes(Image.open(image_path), lang='kor').split('\n')
    word_info = []
    for res in ocr_res:
        word_info.append(res.split(' '))
    return word_info

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return 0

def try_math(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    word_infos = ocr_result(image_path)[:-1]
    for id in title_list:
        if id in cls_info.keys():
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [int(word_info[1]), int(word_info[2]), int(word_info[3]), int(word_info[4])]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    words.append([word_info[0], word_bbox[1], word_bbox[2]]) 
            words.sort(key=lambda x: x[1]+10*x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + ' '
            title2string[id] = word_string
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [int(word_info[1]), int(word_info[2]), int(word_info[3]), int(word_info[4])]
            word_text = word_info[0]
            word_text = re.sub('[^-+0123456789.]', '',  word_text)
            word_text_num = re.sub('[^0123456789]', '', word_text)
            word_text_pure = re.sub('[^0123456789.]', '', word_text)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board+10:
                dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    (word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                y_mid = (word_bbox[1]+word_bbox[3])/2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == '-':
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == '-':
                        min_value = -min_value
        delta_min_max = max_value-min_value
        delta_mark = min_y - max_y + 1e-5
        delta_plot_y = y_min - y_max
        delta = delta_min_max/delta_mark
        if abs(min_y-y_min)/delta_plot_y > 0.1:
            min_value = int(min_value + (min_y-y_min)*delta)

    return title2string, round(min_value, 2), round(max_value, 2), word_infos

def make_words(valid_label):
    labels = []
    ranges = []
    _label = ""
    _range = [0, 0]
    num_letters = len(valid_label)
    cnt = 0
    tau = 20    
    MAX_DIST_INIT = tau*2
    max_dist = MAX_DIST_INIT
    if num_letters == 1:
        labels = [valid_label[0][0]]
        ranges = [[valid_label[0][1], valid_label[0][3]]]
        return labels, ranges

    for i in range(num_letters):
        dist = valid_label[i+1][1] - valid_label[i][3] if i != num_letters -1 else valid_label[i][1] - valid_label[i-1][3]
        if dist < max_dist + tau and dist >= 0: 
            _label += valid_label[i][0]
            if max_dist == MAX_DIST_INIT: 
                cnt = 1
                max_dist = 2 * dist 
                _range[0] = valid_label[i][1]
            else:
                max_dist = (max_dist*cnt + dist) / (cnt+1)
                cnt += 1
                _range[1] = valid_label[i][3]

        else: 
            _label += valid_label[i][0]
            _range[1] = valid_label[i][3]
            if _range[0] == 0: 
                _range[0] = valid_label[i][1]
            labels.append(_label)
            ranges.append(_range)

            _label = ""
            _range = [0, 0]
            max_dist = MAX_DIST_INIT
            cnt = 0
    if _label != "":
        labels.append(_label)
        ranges.append(_range)   
    return labels, ranges

def in_bbox(box, letter):
    """
    box = [tl.x, tl.y, br.x, br.y]  
    letter = [text, tl.x, tl.y br.x, br.y] 
    """
    if box[0] < letter[1] and box[1] < letter[2] and box[2] > letter[3] and box[3] > letter[4]:
        return True
    return False

def ocr_post_process(ocr_res, ret, h):
    # filtering. Brute force..
    valid_str = []
    for word in ret:
        for lt in word:
            if lt != ' ' and lt != '' and lt != '\x0c':
                valid_str.append(lt)
    filtered_ocr_res = []
    for word in ocr_res:
        if word[0] in valid_str:
            filtered_ocr_res.append(word)

    cvrt_ocr_res = []
    for res in filtered_ocr_res:
        _spl = res.split(' ')[:-1]
        _spl[1] = eval(_spl[1])
        _spl[2] = h - eval(_spl[2]) if eval(_spl[2]) > 0 else 0
        _spl[3] = eval(_spl[3])
        _spl[4] = h - eval(_spl[4]) if eval(_spl[4]) > 0 else 0
        cvrt_ocr_res.append(_spl)
    return cvrt_ocr_res

def chart_process(image_path):
    image_cls = Image.open(image_path)
    image = cv2.imread(image_path)
    with torch.no_grad():
        results = methods['Cls'][2](image, methods['Cls'][0], methods['Cls'][1], debug=False)
        info = results[0]
        tls = results[1]
        brs = results[2]
        plot_area = []
        image_painted, cls_info = GroupCls(image_cls, tls, brs)
        title2string, min_value, max_value, word_infos = try_math(image_path, cls_info)
        chartinfo = [info['data_type'], cls_info, title2string, min_value, max_value]

        if info['data_type'] == 0:
            # Assume vertical bar
            print("Predicted as BarChart")
            results = methods['Bar'][2](image, methods['Bar'][0], methods['Bar'][1], debug=False)
            tls = results[0]
            brs = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            image_painted, bar_data, bbox = GroupBar(image_painted, tls, brs, plot_area, min_value, max_value)
            img = Image.open(image_path)
            ret = pytesseract.image_to_string(image_painted, lang='kor').split('\n')
            ocr_res = pytesseract.image_to_boxes(image_painted, lang='kor').split('\n')[:-1]  
            w,h = img.size 

            cvrt_ocr_res = ocr_post_process(ocr_res, ret, h)

            legend_bbox = cls_info[0] if 0 in cls_info.keys() else None
            if legend_bbox != None:
                # Check the location of legend. In the chart, left, right, bottom, top
                # tl.x tl.y br.x br.y
                leg_loc = "in"
                if legend_bbox[2] < plot_area[0]:
                    leg_loc = "left"    
                elif legend_bbox[0] > plot_area[2]:
                    leg_loc = "right"
                elif legend_bbox[1] > plot_area[3]:
                    leg_loc = "bottom"
                elif legend_bbox[3] < plot_area[1]:
                    leg_loc = "top"

                legend_valid_label = []
                for letter in cvrt_ocr_res:
                    if (in_bbox(legend_bbox, letter)):
                        legend_valid_label.append(letter)
                legend_labels, legend_ranges = make_words(legend_valid_label)

            else:
                leg_loc = None
                legend_labels = []
            
            leg_lim = legend_bbox[1] +5 if leg_loc=="bottom" else 1e5 
            bar_bottom_y = max([b[3] for b in bbox[0]])
            valid_label = []
            for label in cvrt_ocr_res:
                if label[4] > bar_bottom_y and label[4] < leg_lim: 
                    valid_label.append(label)

            labels, ranges = make_words(valid_label)

            num_label = len(labels)
            del_list = []
            for i in range(num_label):
                for j in range(num_label):
                    if i >= j:
                        continue
                    up_l, up_r = ranges[i]; down_l, down_r = ranges[j]
                    if (up_l >= down_l and up_r <= down_r) or (up_l <= down_l and up_r >= down_r) or (up_r >= down_l and up_l <= down_r) or (up_l <= down_r and up_r >= down_l):
                        labels[i] += labels[j]
                        del_list.append(j)
                        continue

            del_list.sort(reverse=True)
            for d in del_list:
                del ranges[d]
                del labels[d]
                
            min_dist = 1e5
            output = {}
            _key = ""
            def list_chunk(lst, n):
                return [lst[i:i+n] for i in range(0, len(lst), n)]
            num_data = int(len(bbox[0]) / len(bar_data))
            bbox = list_chunk(bbox[0], num_data) 

            legend_omit = len(bar_data) - len(legend_labels)   
            for i in range(legend_omit):
                legend_labels.append(chr(65 + i))

            for _data, _legend in zip(bar_data, legend_labels):
                intermd_output = {}
                for data, label in zip(_data, labels):
                    intermd_output[label] = data
                output[_legend] = intermd_output
            result = pd.DataFrame.from_dict(output)
            # else:
            #     for data, label in zip(bar_data[0], labels):
            #         output[label] = data
            #     result = pd.DataFrame.from_dict([output])
            # import pdb; pdb.set_trace()

            return result
            
        if info['data_type'] == 2:
            print("Predicted as PieChart")
            results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
            cens = results[0]
            keys = results[1]
            image_painted, pie_data = GroupPie(image_painted, cens, keys)

            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]

            img = Image.open(image_path)
            ret = pytesseract.image_to_string(image_painted, lang='kor').split('\n')
            ocr_res = pytesseract.image_to_boxes(image_painted, lang='kor').split('\n')[:-1] 
            w,h = img.size 

            cvrt_ocr_res = ocr_post_process(ocr_res, ret, h)

            legend_bbox = cls_info[0] if 0 in cls_info.keys() else None
            if legend_bbox != None:
                # Check the location of legend. In the chart, left, right, bottom, top
                # tl.x tl.y br.x br.y
                leg_loc = "in"
                if legend_bbox[2] < plot_area[0]:
                    leg_loc = "left"    
                elif legend_bbox[0] > plot_area[2]:
                    leg_loc = "right"
                elif legend_bbox[1] > plot_area[3]:
                    leg_loc = "bottom"
                elif legend_bbox[3] < plot_area[1]:
                    leg_loc = "top"

                legend_valid_label = []
                for letter in cvrt_ocr_res:
                    if (in_bbox(legend_bbox, letter)):
                        legend_valid_label.append(letter)
                legend_labels, legend_ranges = make_words(legend_valid_label)
            else:
                leg_loc = None
                legend_labels = []

            legend_omit = len(pie_data) - len(legend_labels)
            for i in range(legend_omit):
                legend_labels.append(chr(65 + i))
            
            output = {}
            for data, label in zip(pie_data, legend_labels):
                output[label] = data

            result = pd.DataFrame.from_dict([output]).T
            return result
        
        if info['data_type'] == 1:
            print("Predicted as LineChart")
            results = methods['Line'][2](image, methods['Line'][0], methods['Line'][1], debug=False, cuda_id=1)
            keys = results[0]
            hybrids = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            image_painted, quiry, keys, hybrids = GroupQuiry(image_painted, keys, hybrids, plot_area, min_value, max_value)
            results = methods['LineCls'][2](image, methods['LineCls'][0], quiry, methods['LineCls'][1], debug=False, cuda_id=1)
            line_data = GroupLine(image_painted, keys, hybrids, plot_area, results, min_value, max_value)

            img = Image.open(image_path)
            ret = pytesseract.image_to_string(image_painted, lang='kor').split('\n')
            ocr_res = pytesseract.image_to_boxes(image_painted, lang='kor').split('\n')[:-1]  
            w,h = img.size 

            cvrt_ocr_res = ocr_post_process(ocr_res, ret, h)

            legend_bbox = cls_info[0] if 0 in cls_info.keys() else None
            if legend_bbox != None:
                # Check the location of legend. In the chart, left, right, bottom, top
                # tl.x tl.y br.x br.y
                leg_loc = "in"
                if legend_bbox[2] < plot_area[0]:
                    leg_loc = "left"   
                elif legend_bbox[0] > plot_area[2]:
                    leg_loc = "right"
                elif legend_bbox[1] > plot_area[3]:
                    leg_loc = "bottom"
                elif legend_bbox[3] < plot_area[1]:
                    leg_loc = "top"

               
                legend_valid_label = []
                for letter in cvrt_ocr_res:
                    if (in_bbox(legend_bbox, letter)):
                        legend_valid_label.append(letter)
                legend_labels, legend_ranges = make_words(legend_valid_label)
            else:
                leg_loc = None
                legend_labels = []

            # 5 = plot area, 4 = whole space
            
            bot_bnd = cls_info[4][3] if 4 in cls_info.keys() else 1e5
            top_bnd = cls_info[5][3] if 5 in cls_info.keys() else 0
            valid_label = []
            for label in cvrt_ocr_res:
                if label[4] > top_bnd and label[4] < bot_bnd:
                    valid_label.append(label)

            labels, ranges = make_words(valid_label)
            num_label = len(labels)
            del_list = []
            for i in range(num_label):
                for j in range(num_label):
                    if i >= j:
                        continue
                    up_l, up_r = ranges[i]; down_l, down_r = ranges[j]
                    if (up_l >= down_l and up_r <= down_r) or (up_l <= down_l and up_r >= down_r) or (up_r >= down_l and up_l <= down_r) or (up_l <= down_r and up_r >= down_l):
                        labels[i] += labels[j]
                        del_list.append(j)
                        continue

            del_list.sort(reverse=True)
            for d in del_list:
                del ranges[d]
                del labels[d]

            legend_omit = len(line_data) - len(legend_labels)
            for i in range(legend_omit):
                legend_labels.append(chr(65 + i))

            output = {}
            for line, legend in zip(line_data, legend_labels):
                intermd_output = {}
                for point, label in zip(line, labels):
                    intermd_output[label] = point
                output[legend] = intermd_output

            result = pd.DataFrame.from_dict(output)
            return result




if __name__ == "__main__":

    # tar_path = 'PATH/TO/DATASET'
    # images = os.listdir(tar_path)
    # from random import shuffle
    # shuffle(images)
    images = ["/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/796930_figure_2.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/1111_bar_AGC.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/pie_ex.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/160124_figure_1.png"]
    for image in tqdm(images):
        res = chart_process(images)
        print(res)
