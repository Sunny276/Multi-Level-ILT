import os
import sys
import time
import torch
import logging
from argparse import ArgumentParser


def make_logger(log_name):
    savedir = "./img/{}".format(log_name)
    if(not os.path.exists(savedir)):
        os.makedirs(savedir)
        
    log_format = "%(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                        format=log_format)
    fh = logging.FileHandler(os.path.join(savedir, '{}.log'.format(log_name)), mode='w+')
    logging.getLogger().addHandler(fh)
    
    return logging, savedir

def make_parser():
    parser = ArgumentParser("Multi-Level ILT")
    parser.add_argument("--name", default="center_fast_test", help="name")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--dataset", default="neural_test", help="dataset name")
    parser.add_argument("--exact", action="store_true", help="whether use exact mode")
    
    args = parser.parse_args()
    return args

def get_restrict_map(img, device='cpu', mode=0):
    if mode==0:
        x1, y1, x2, y2 = img.getbbox()
        max_width, max_height = img.size
        w, h = x2 - x1, y2 - y1
        max_len = max(w, h)
        x_new = int(x1 + w / 2) - int(max_len / 2)
        y_new = int(y1 + h / 2) - int(max_len / 2)
        margin = 256
        new_cord = [
            x_new - margin,
            y_new - margin,
            x_new + max_len + margin,
            y_new + max_len + margin,
        ]
        if (
            (x_new - margin) <= 0
            or (y_new - margin) <= 0
            or (x_new + max_len + margin) >= max_width
            or (y_new + max_len + margin) >= max_height
        ):
            new_cord = [0, 0, max_width, max_height]

        restrict_map = torch.zeros([2048, 2048], dtype=float).to(device)
        restrict_map[new_cord[1]:new_cord[3], new_cord[0]:new_cord[2]] = 1
    
    return restrict_map

if __name__ == '__main__':
    savedir = make_logger("test")
    