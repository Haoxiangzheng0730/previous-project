from filters import KalmanFilter
from estimator import RansacMotionEstimator
from stabilzer import OptimalPathStabilizer, OptimalPathStabilizerXYA
from trajectory import Trajectory
import cv2
import numpy as np
import time
import datetime
import math


class FrameInfo:
    def __init__(self):
        self.features = []
        self.number = 0
        self.trajectory = Trajectory()
        self.shape = ()

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def width_height(self):
        return self.shape[::-1]


def read_VideoGray(video):
    ret, frame = video.read()
    if ret:
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frameGray = None
    return ret, frameGray


def get_PSNR(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0 ** 2
    return 10 * np.log10(PIXEL_MAX / mse)


def get_ITF(videoPath):
    video = cv2.VideoCapture(videoPath)
    N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ITF = 0

    _, currFrame = read_VideoGray(video)
    for i in range(N_FRAMES - 1):
        _, nextFrame = read_VideoGray(video)
        ITF += get_PSNR(currFrame, nextFrame)
        currFrame = nextFrame

    ITF = 1.0 / (N_FRAMES - 1) * ITF
    video.release()
    return ITF


def crop_image(img):
    return img[limits[1]:frame.height - limits[1], limits[0]:frame.width - limits[0]]


def get_x(mat):
    return mat[0, 2]


def get_y(mat):
    return mat[1, 2]


def get_rad_angle(mat):
    return math.atan2(mat[1, 0], mat[0, 0])


def fill_mat(mat, dx, dy, angle):
    mat[0, 0] = math.cos(angle)
    mat[0, 1] = -math.sin(angle)
    mat[1, 0] = math.sin(angle)
    mat[1, 1] = math.cos(angle)

    mat[0, 2] = dx
    mat[1, 2] = dy


def transform(mat, point):
    return (
        mat[0, 0] * point[0] + mat[0, 1] * point[1] + mat[0, 2],
        mat[1, 0] * point[0] + mat[1, 1] * point[1] + mat[1, 2]
    )


time_begin = datetime.datetime.now()
video_path = "fang.wmv"
video = cv2.VideoCapture(video_path)

N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 按帧读取视频
FPS = video.get(cv2.CAP_PROP_FPS)  # fps
VID_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
VID_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height
print("N_FRAMES: " + str(N_FRAMES))
print("FPS: " + str(FPS))
print("Ready")
time_pre = datetime.datetime.now()

frame = None
prev_frame = None

org_trajectories = []
org_transformations = []
frames = []

prev_trans = None
prev_frame_img = None
frame_number = 0

crop = 40
crop_rate = crop / 20

# Kalman滤波
filter = KalmanFilter(Trajectory(4e-2, 4e-2, 4e-2), Trajectory(crop_rate, crop_rate, crop_rate), Trajectory(1, 1, 1))

frame_width = int(1336.0 / 2)
frame_height = int(768.0 / 2)


def resize(img):
    return cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)


lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
fast_params = dict(threshold=25,
                   nonmaxSuppression=True)

# fast特征点检测
fast = cv2.FastFeatureDetector_create(**fast_params)

# 采用ransac，具体参见estimator.py
motion_estimator = RansacMotionEstimator(20, 1.5, remember_inlier_indices=True)

crop_rate = 0.9
limits = [int(frame_width * (1 - crop_rate)), int(frame_height * (1 - crop_rate)), 0.05]

# 参数and阈值
feature_cont = 0
flow_cont = 0
ransac_cont = 0
kalman_cont = 0
features_quant = []
percent = 0
inliers_quant = []
ouliers_quant = []
compensate_count = 0

print("开始优化")
for k in range(N_FRAMES - 1):
    ret, frame_img = video.read()

    if frame_img is None:
        break
    if frame is not None:
        prev_frame = frame

    frame_number += 1
    frame = FrameInfo()
    frame.number = frame_number
    frame_img = resize(frame_img)
    frame_img_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    feature_time_ini = time.time()
    frame.features = fast.detect(frame_img_gray, None)
    frame.shape = frame_img_gray.shape
    frames.append(frame)

    features_quant.append(len(frame.features))
    # im = cv2.drawKeypoints(frame_img, frame.features, None, color=(255, 0, 0),
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('point.jpg', im)

    if prev_frame is not None:
        # begin
        feature_time_fim = time.time()
        feature_cont += feature_time_fim - feature_time_ini
        feature_time_ini = 0
        feature_time_fim = 0

        flow_time_ini = time.time()
        LK_pts = np.array([[[f.pt[0], f.pt[1]]] for f in frame.features], np.float32)

        new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame_img, frame_img, LK_pts, None, **lk_params)
        new_features_for_validation, _, _ = cv2.calcOpticalFlowPyrLK(frame_img, prev_frame_img, new_features, None,
                                                                     **lk_params)

        flow_time_fim = time.time()
        flow_cont += flow_time_fim - flow_time_ini
        flow_time_ini = 0
        flow_time_fim = 0

        d = abs(LK_pts - new_features_for_validation).reshape(-1, 2).max(-1)
        good_features = d < 1

        # 挑选feature_points
        good_new = np.array([x for x, s in zip(new_features, good_features) if s], dtype=np.float32)
        good_old = np.array([x for x, s in zip(LK_pts, good_features) if s], dtype=np.float32)
        ransac_time_ini = time.time()
        trans, inliers_indices, outliers_indices = motion_estimator.estimate(good_old, good_new)

        ransac_time_fim = time.time()
        ransac_cont += ransac_time_fim - ransac_time_ini
        ransac_time_ini = 0
        ransac_time_fim = 0

        inliers_quant.append(len(inliers_indices))
        ouliers_quant.append(len(outliers_indices))

        if trans is None and prev_trans is None:
            continue

        if trans is None:
            trans = prev_trans

        org_transformations.append(trans)
        prev_trans = trans.copy()
    prev_frame_img = frame_img

kalman_time_ini = time.time()
stabilizer = OptimalPathStabilizerXYA(
    [get_x(trans) for trans in org_transformations],
    [get_y(trans) for trans in org_transformations],
    [get_rad_angle(trans) for trans in org_transformations]
    , [limits[0] * 0.5, limits[1] * 0.5, limits[2]])

new_trans = stabilizer.stabilize()

filter.put(new_trans)
delta = filter.get()

kalman_time_fim = time.time()
kalman_cont += kalman_time_fim - kalman_time_ini
kalman_time_ini = 0
kalman_time_fim = 0

# 优化完成， 开始视频转化
time_pre_end = datetime.datetime.now()
time_pro = datetime.datetime.now()

for _ in range(1):
    video.release()
    video = cv2.VideoCapture(video_path)

    videoOutPath = 'sab111.avi'

    frame_number = 0
    pressed_q = False
    output_video = cv2.VideoWriter(videoOutPath,
                                   cv2.VideoWriter_fourcc(*'XVID'), FPS,
                                   (frame_width - 2 * limits[0], frame_height - 2 * limits[1]))
    trans = np.zeros((2, 3), dtype=np.float32)

    for t, frame in enumerate(frames):
        if t + 1 >= len(org_transformations):
            break

        print("\r已处理 %00.2f%% (%04d / %04d)" % (t / N_FRAMES * 100, t, N_FRAMES))

        fill_mat(trans, delta[0][t - 1], delta[1][t - 1], delta[2][t - 1])

        _, frame_img = video.read()
        frame_img = resize(frame_img)

        compensate_time_ini = time.time()
        out = cv2.warpAffine(frame_img, trans, frame.width_height, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)

        compensate_time_fim = time.time()
        compensate_count += compensate_time_fim - compensate_time_ini
        compensate_time_ini = 0
        compensate_time_fim = 0

        for t, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = transform(trans, new.ravel())
            c, d = transform(trans, old.ravel())
            color = [0, 255, 255]
            color_bad = [255, 0, 0]
            is_inlier = t in inliers_indices
            # cv2.line(out, (int(a), int(b)), (int(c), int(d)), color if is_inlier else color_bad, 2)
            # cv2.circle(out, (int(a), int(b)), 3, color if is_inlier else color_bad, -1)
        output_video.write(crop_image(out))
        cv2.imshow('stab', crop_image(out))
        cv2.imshow('org', crop_image(frame_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pressed_q = True
            break

    video.release()
    if pressed_q:
        break

    video.release()
    output_video.release()

cv2.destroyAllWindows()
time_pro_end = datetime.datetime.now()

soma = sum(features_quant)
percent = soma / len(features_quant)

soma2 = sum(inliers_quant)
percent2 = soma / len(inliers_quant)

soma3 = sum(outliers_indices)
if len(outliers_indices) != 0:
    percent3 = soma3 / len(outliers_indices)
else:
    percent3 = 0

time_end = datetime.datetime.now()
print("-------------statics-------------")
print("检测到的每帧平均特征点：" + str(percent))
print("检测到的每帧平均内线数：", str(percent2))
print("产生的总误匹配（异常值）：", str(percent3))
print("roi：", str(feature_cont))
print("ransanc counts剔除：", str(ransac_cont))
print("kalman counts：", str(kalman_cont))
print("帧补偿: ", str(compensate_count))
print("原视频峰PSNR: " + str(get_ITF(video_path)))
print("稳像视频PSNR: " + str(get_ITF(videoOutPath)))
print("预处理时间  ：" + str((time_end - time_begin).seconds - (time_pre_end - time_pre).seconds) + "s")
print("视频处理时间: " + str((time_pre_end - time_pre).seconds) + "s")
print("总处理时间  : " + str((time_end - time_begin).seconds) + "s")
print("处理每帧的时间为：" + str((time_pro_end - time_pro).seconds / N_FRAMES) + "s")
