import os
import cv2
import time
import torch
import numpy as np
import argparse

import torch.nn.functional as F
import torchvision
from PIL import Image

from litho.lithosimer import Lithosimer
from utils import make_parser, make_logger, get_restrict_map
from epe_checker import get_epe_checkpoints, report_epe_violations


def l2_loss_torch(litho_out, target_img):
    return torch.square(litho_out - target_img)

def post_process(img):
    try:
        img, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    cv_contours = []
    img = np.zeros_like(img, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 1000) and (area < 5000):
            x, y, w, h = cv2.boundingRect(contour)  
            img[y:y+h, x:x+w] = 255  
        elif area >= 5000:  # area > 1000
            cv_contours.append(contour)
            
    cv2.fillPoly(img, cv_contours, 255)
    return img

def prepare_mask(mask_img):
    s = int(img_size / mask_img.size(0))
    upsample_func = torch.nn.Upsample(scale_factor=s, mode='nearest')
    add_thr = 0.1
    bmask = torch.sigmoid(beta * (mask_img - mthr + add_thr))
    bmask[bmask>=0.5] = 1
    bmask[bmask<0.5] = 0
    bmask = upsample_func(bmask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    bmask = post_process(bmask.detach().cpu().numpy().astype(np.uint8) * 255)
    
    return bmask

def check_result(bmask, cnt=0):
    litho_out = lithosimer.litho_show(bmask, dose=1.0, flag_defocus=False, ilt_option=False)
        
    l2 = torch.logical_xor(litho_out, target_img).sum()
    cv2.imwrite(os.path.join(savedir, "mask_%d.png"%cnt), bmask.detach().cpu().numpy() * 255)
    cv2.imwrite(os.path.join(savedir, "litho_%d.png"%cnt), litho_out.detach().cpu().numpy() * 255)
    
    checkpoints = get_epe_checkpoints((target_img.detach().cpu().numpy() * 255).astype(np.uint8))
    epe_violation = report_epe_violations((litho_out.detach().cpu().numpy()*255).astype(np.uint8), checkpoints)
    
    r_out = lithosimer.litho_show(bmask, dose=max_dose, flag_defocus=False, ilt_option=False)
    r_in = lithosimer.litho_show(bmask, dose=min_dose, flag_defocus=True, ilt_option=False)
    pvb = torch.logical_xor(r_out, r_in).sum()

    return l2, pvb, epe_violation

def low_resolution_ilt(target_img, mask_img, restrict_map, s, n, lr=1):
    target_img = F.avg_pool2d(target_img.unsqueeze(0).unsqueeze(0), s, stride=s, padding=0).squeeze(0).squeeze(0)
    mask_img = F.avg_pool2d(mask_img.unsqueeze(0).unsqueeze(0), s, stride=s, padding=0).squeeze(0).squeeze(0)
    mask_img = mask_img.detach().clone()
    restrict_map = restrict_map[::s, ::s]
    mask_img.requires_grad = True
    upsample_func = torch.nn.Upsample(scale_factor=s, mode='nearest')
    for i in range(1, n+1):
        mask_img = torch.clamp(mask_img, -2, 2)
        mask_img = F.avg_pool2d(mask_img.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze(0).squeeze(0)
        bmask = torch.sigmoid(beta * (mask_img - mthr)) 
        r_out = lithosimer.litho_show(bmask, dose=max_dose, flag_defocus=False)
        r_in = lithosimer.litho_show(bmask, dose=min_dose, flag_defocus=True)
        err = l2_loss_torch(r_in, r_out) + l2_loss_torch(r_out, target_img)
        grad_m = torch.autograd.grad(err, mask_img, grad_outputs=torch.ones_like(mask_img))[0]
                     
        mask_img = mask_img - lr * grad_m * restrict_map
    mask_img = upsample_func(mask_img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    return mask_img

def high_resolution_ilt(target_img, mask_img, restrict_map, s, n, lr=1):
    mask_img = F.avg_pool2d(mask_img.unsqueeze(0).unsqueeze(0), s, stride=s, padding=0).squeeze(0).squeeze(0)
    mask_img = mask_img.detach().clone()
    restrict_map = restrict_map[::s, ::s]
    mask_img.requires_grad = True
    r = int(target_img.size(0) / mask_img.size(0))
    upsample_func = torch.nn.Upsample(scale_factor=r, mode='nearest') 
    for i in range(1, n+1):
        bmask = torch.sigmoid(beta * (mask_img - mthr))
        bmask = upsample_func(bmask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        r_out = lithosimer.litho_show(bmask, dose=max_dose, flag_defocus=False)
        r_in = lithosimer.litho_show(bmask, dose=min_dose, flag_defocus=True)
        err = l2_loss_torch(r_in, r_out) + l2_loss_torch(r_out, target_img)
        err = F.avg_pool2d(err.unsqueeze(0).unsqueeze(0), r, stride=r, padding=0).squeeze(0).squeeze(0)
        grad_m = torch.autograd.grad(err, mask_img, grad_outputs=torch.ones_like(mask_img))[0]
        mask_img = mask_img - lr * grad_m * restrict_map
        mask_img = torch.clamp(mask_img, -2, 2)
            
    upsample_func = torch.nn.Upsample(scale_factor=s, mode='nearest')
    mask_img = upsample_func(mask_img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    return mask_img

if __name__ == '__main__': 
    args = make_parser()
    logger, savedir = make_logger(args.name)
    
    beta = 4
    mthr = 0.5
    add_thr = 0.1
    max_dose = 1.02
    min_dose = 0.98
    img_size = 2048
    
    device = "cuda:%s" % args.gpu if torch.cuda.is_available() else "cpu"
    
    data_root = "./dataset/{}".format(args.dataset)
    
    gray_scale_img_loader = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ])
    
    lithosimer = Lithosimer(device=device)
    
    if args.dataset == "develset_test":
        restrict_map = cv2.imread(os.path.join(data_root, "filter.png"), 0)
        restrict_map = restrict_map / 255
        restrict_map = torch.tensor(restrict_map).to(device)
    
    for i in range(1, 10+1):
        if args.dataset == "neural_test":
            filename = "t%d_0_mask.png"%i
            input_layout_path = os.path.join(data_root, filename)
            target = Image.open(input_layout_path)
            restrict_map = get_restrict_map(target, device=device, mode=0)
        else:
            filename = "%d.png"%i
            input_layout_path = os.path.join(data_root, filename)
            target = Image.open(input_layout_path)
            
        t0 = time.time()
        target_img = gray_scale_img_loader(target).to(device).squeeze(0)
        mask_img = target_img.detach().clone()
        mask_img.requires_grad = True
        
        if args.exact:
            mask_img = low_resolution_ilt(target_img, mask_img, restrict_map, s=4, n=80, lr=1)
            mask_img = high_resolution_ilt(target_img, mask_img, restrict_map, s=8, n=10, lr=1)
            
        else:
            mask_img = low_resolution_ilt(target_img, mask_img, restrict_map, s=4, n=35, lr=1)
            mask_img = high_resolution_ilt(target_img, mask_img, restrict_map, s=8, n=5, lr=1)
            
        bmask = prepare_mask(mask_img)
        t1 = time.time()
        
        bmask = torch.tensor(bmask/255).to(device)
        restrict_map = restrict_map.to(dtype=bool)
        bmask[~restrict_map] = 0
        l2, pvb, epe_violation = check_result(bmask, cnt=i)
        logger.info("ID=%d\t L2=%d\t PVB=%d\t EPE=%d\t TAT=%.3f"%(i, l2, pvb, epe_violation, (t1-t0)))
            