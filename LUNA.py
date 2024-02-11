from aim.screen_inf import grab_screen_mss, grab_screen_win32, get_parameters
from aim.cs_model import load_model
import pywintypes
import cv2
import win32gui
import win32con
import torch
import numpy as np
import configparser
import os
import sys
import aim.ghub_mouse as ghub
from math import *

from aim.verify_args import verify_args
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import pynput
import argparse
import time
import os
from simple_pid import PID

def read_config():
    config_dir = '.\\config.ini'
    cf = configparser.ConfigParser()
    cf.read(config_dir, encoding="utf-8")
    return cf

print(
    """\033[0;34m
LOGOhere
\033[0m""")
cf = read_config()
smooth = cf.getfloat('CONFIG', 'smooth')
ADS = cf.getfloat('CONFIG', 'ADS')
button = cf.get('CONFIG', 'button')
head = cf.get('CONFIG', 'head')
mode = cf.get('CONFIG', 'mode')
window = cf.get('CONFIG', 'window')
xP = cf.getfloat('CONFIG', 'xP')
xI = cf.getfloat('CONFIG', 'xI')
xD = cf.getfloat('CONFIG', 'xD')
yP = cf.getfloat('CONFIG', 'yP')
yI = cf.getfloat('CONFIG', 'yI')
yD = cf.getfloat('CONFIG', 'yD')
confidence = cf.getfloat('CONFIG', 'confidence')
decrange = cf.getfloat('CONFIG', 'decrange')
pos = cf.getfloat('CONFIG', 'pos')
deca=cf.getfloat('CONFIG', 'deca')
decb=cf.getfloat('CONFIG', 'decb')

def lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy):
    mouse_pos_x, mouse_pos_y = mouse.position
    aims_copy = aims.copy()
    detect_arange =decrange
    aims_copy = [x for x in aims_copy if x[0] in args.lock_choice and (len_x * float(x[1]) + top_x - mouse_pos_x) ** 2 + (len_y * float(x[2]) + top_y - mouse_pos_y) ** 2 < detect_arange]
    k = 4.07 * (1 / args.lock_smooth)
    if len(aims_copy):
        dist_list = []
        for det in aims_copy:
            _, x_c, y_c, _, _ = det
            dist = (len_x * float(x_c) + top_x - mouse_pos_x) ** 2 + (len_y * float(y_c) + top_y - mouse_pos_y) ** 2
            dist_list.append(dist)

        if dist_list:
            det = aims_copy[dist_list.index(min(dist_list))]
            tag, x_center, y_center, width, height = det
            x_center, width = len_x * float(x_center) + top_x, len_x * float(width)
            y_center, height = len_y * float(y_center) + top_y, len_y * float(height)
            rel_x = int(k / args.lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)
            rel_y = int(k / args.lock_sen * atan((mouse_pos_y - y_center + pos * height) / 640) * 640)#瞄準高度可自行調整(建議為1/4)
            pid_movex = pidx(rel_x)
            pid_movey = pidy(rel_y)
            ghub.mouse_xy(round(pid_movex), round(pid_movey))


#if os.path.exists('C:\\Windows\\twaicn_32.dll') !=True:
#    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('--model-path',type=str, default='.\\NC.dll',help='权重')
parser.add_argument('--imgsz', type=int, default=640, help='训练模型时的分辨率')
parser.add_argument('--conf-thres', type=float, default=confidence, help='信心分数')
parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU分数')
parser.add_argument('--use-cuda', type=bool, default=True, help='启用cuda核心') 
parser.add_argument('--show-window', type=bool, default=window, help='是否显示实时检测窗口')
parser.add_argument('--top-most', type=bool, default=True, help='是否窗口置頂')
parser.add_argument('--resize-window', type=float, default=0.5, help='窗口大小调整')
parser.add_argument('--thickness', type=int, default=2, help='方框标记粗细')
parser.add_argument('--show-fps', type=bool, default=True, help='显示帧率')
parser.add_argument('--show-label', type=bool, default=False, help='是否显示label')
parser.add_argument('--use_mss', type=str, default=True, help='是否使用mss截屏；为False时使用win32截屏')
parser.add_argument('--region', type=tuple, default=(deca, decb), help='检测范围(占全屏分辨率比例)')
parser.add_argument('--hold-lock', type=bool, default=mode, help='自瞄模式；True为按住，False为切换')
parser.add_argument('--lock-sen', type=float, default= ADS, help='自瞄幅度系數,游戏中灵敏度(建议不要调整)')
parser.add_argument('--lock-smooth', type=float, default=smooth, help='自瞄平滑度 大则平滑')
parser.add_argument('--lock-button', type=str, default=button, help='自瞄按键(鼠标)；left(左鍵)、right(右鍵)、x1(侧键下)、x2(侧键上)')
parser.add_argument('--head-first', type=bool, default=head, help='是否优先锁定头部')
parser.add_argument('--lock-tag', type=list, default=[0], help='对应标签；person(根据模型修改)')
parser.add_argument('--lock-choice', type=list, default=[0], help='目标选择；决定锁定的目标，从自己的标签中选择')

args = parser.parse_args()

'------------------------------------------------------------------------------------'

verify_args(args)

cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'

args.weights = cur_dir + args.model_path
args.lock_tag = [str(i) for i in args.lock_tag]
args.lock_choice = [str(i) for i in args.lock_choice]

device = 'cuda' if args.use_cuda else 'cpu'
half = device != 'cpu'
imgsz = args.imgsz

conf_thres = args.conf_thres
iou_thres = args.iou_thres

top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))

monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

model = load_model(args)
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

lock_mode = False
team_mode = True
lock_button = eval('pynput.mouse.Button.' + args.lock_button)

mouse = pynput.mouse.Controller()

#PID系数可调整
#PID(P, I, D)
#P: 加快系统反映。输出值较快，但越大越不稳定
#I: 积分。用于稳定误差
#D: 微分。提高系统的动态性能
#以下为个人使用参数可供参考
pidx = PID(xP, xI, xD, setpoint=0, sample_time=0.001,)
pidy = PID(yP, yI, yD, setpoint=0, sample_time=0.001,)
pidx.output_limits = (-4000 ,4000)
pidy.output_limits = (-3000 ,3000)

if args.show_window:
    #cv2.namedWindow('LUNA', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('LUNA', int(len_x * args.resize_window), int(len_y * args.resize_window))
    cv2.namedWindow('NC', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('NC', int(len_x * args.resize_window), int(len_y * args.resize_window))

def on_click(x, y, button, pressed):
    global lock_mode
    if button == lock_button:
        if args.hold_lock:
            if pressed:
                lock_mode = True
                print('开始自瞄')
            else:
                lock_mode = False
                print('停止自瞄')
        else:
            if pressed:
                lock_mode = not lock_mode
                print('自瞄：', 'on' if lock_mode else 'off')

listener = pynput.mouse.Listener(on_click=on_click)
listener.start()

print('自瞄已部署')
t0 = time.time()
cnt = 0

while True:

    if cnt % 20 == 0:
        top_x, top_y, x, y = get_parameters()
        len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
        top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
        monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}
        cnt = 0

    if args.use_mss:
        img0 = grab_screen_mss(monitor)
        img0 = cv2.resize(img0, (len_x, len_y))
    else:
        img0 = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
        img0 = cv2.resize(img0, (len_x, len_y))

    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    #print(pred[0])  #预测到的信心分数
    aims = []
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # bbox:(tag, x_center, y_center, x_width, y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                aims.append(aim)

        if len(aims):
            if lock_mode:
                lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy)

        if args.show_window:
            for i, det in enumerate(aims):
                tag, x_center, y_center, width, height = det
                x_center, width = len_x * float(x_center), len_x * float(width)
                #print("width:" , width)
                #print("x_center:", x_center)
                y_center, height = len_y * float(y_center), len_y * float(height)
                top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                #print("top_left:", top_left)
                bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                #print("bottom_right:", bottom_right)
                cv2.rectangle(img0, top_left, bottom_right, (0, 255, 0), thickness=args.thickness)
                if args.show_label:
                    cv2.putText(img0, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 0, 0), 4)

    if args.show_window:
        if args.show_fps:
            cv2.putText(img0,"FPS:{:.1f}".format(1. / (time.time() - t0)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (210, 240,193), 4)
            if lock_mode:
                cv2.putText(img0, "lock on", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(197, 205, 0), 4)
            else:
                cv2.putText(img0, "lock off", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (197, 205, 0), 4)
            #print(1. / (time.time() - t0))
            t0 = time.time()

        #cv2.imshow('LUNA', img0)
        cv2.imshow('NC', img0)

        if args.top_most:
            #hwnd = win32gui.FindWindow(None, 'LUNA')
            #CVRECT = cv2.getWindowImageRect('LUNA')
            hwnd = win32gui.FindWindow(None, 'NC')
            CVRECT = cv2.getWindowImageRect('NC')
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        cv2.waitKey(1)
    pidx(0)
    pidy(0)
    cnt += 1
