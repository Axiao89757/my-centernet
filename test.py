import datetime
import time
import os
import shutil
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from nets.centernet_training import focal_loss, reg_l1_loss
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import decode_bbox, postprocess
from utils.utils_map import get_map, get_coco_map
from summary import net_summary
from nets.net import Net


# <editor-folder desc="参数">
plan = 11  # 计划
model_path = 'logs/dense_connection_11/best_epoch_weights.pth'  # 模型权重 .pth
# model_path = 'logs/dense_connection/ep700-loss0.668-val_loss1.090.pth'
use_cuda = True
save_dir = 'logs/test'

test_txt_path = 'dataset/SSDD/ImageSets/Exp1/test.txt'  # 测试集txt文件路径
batch_size = 16
num_workers = 8

nms = True
nms_iou = 0.5
confidence = 0.05
max_boxes = 100
MINOVERLAP = 0.5

input_shape = [512, 512]  # 输入图片尺寸（别改）
class_names = ['ship']  # 类别名字
# </editor-fold>

# <editor-folder desc="模型纵览">
flops, params = net_summary(plan=plan)
net_summary_str = 'Total GFLOPS: %s, Total params: %s' % (flops, params)
print('=======================\n')
# </editor-fold>

# <editor-folder desc="Net">
net = Net(num_classes=1, backbone_pretrained=False, plan=plan)  # 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Loading {} model, and classes.'.format(model_path))
# 载入权重
model_dict = net.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
net.load_state_dict(model_dict)
# 显示没有匹配上的Key
print("\nSuccessful Load Key:", str(load_key)[:500], "\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[:500], "\nFail To Load Key num:", len(no_load_key))
print('\n')
if use_cuda:
    net = torch.nn.DataParallel(net)
    net = net.cuda()
# </editor-fold>

# <editor-folder desc="DataLoader">
with open(test_txt_path) as f:
    test_lines = f.readlines()
val_dataset = CenternetDataset(test_lines, input_shape, 1, train=False)
gen_test = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      drop_last=True, collate_fn=centernet_dataset_collate, sampler=None)
# </editor-fold>

# <editor-folder desc="Loss">
total_time = 0
loss = 0
wh_loss = 0
offset_loss = 0
steps = len(test_lines) // batch_size
net = net.eval()
pbar = tqdm(gen_test, postfix=dict, ncols=100, mininterval=0.3)
for iteration, batch in enumerate(pbar):
    if iteration >= steps:
        break
    with torch.no_grad():
        if use_cuda:
            batch = [ann.cuda() for ann in batch]
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        # inference
        torch.cuda.synchronize()
        start = time.time()
        hm, wh, offset = net(batch_images)
        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        # 计算loss
        c_loss = focal_loss(hm, batch_hms)
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        offset_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        # 累计loss
        loss = c_loss + wh_loss + offset_loss
        loss += loss.item()
        wh_loss += wh_loss.item()
        offset_loss += offset_loss.item()
        # 展示loss
        pbar.set_postfix(
            **{'offset_loss': offset_loss.item() / (iteration + 1), 'wh_loss': wh_loss.item() / (iteration + 1),
               'loss': loss.item() / (iteration + 1)}
        )
pbar.close()

# </editor-fold>

# <editor-folder desc="mAP">
map_out_path = ".temp_map_out"
if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)
if not os.path.exists(os.path.join(map_out_path, "ground-truth")):
    os.makedirs(os.path.join(map_out_path, "ground-truth"))
if not os.path.exists(os.path.join(map_out_path, "detection-results")):
    os.makedirs(os.path.join(map_out_path, "detection-results"))


def get_map_txt(image_id, image):
    """
    将一个图片的预测结果保存到临时文件 map_out_path/detection-results/image_id.txt
    保存的结果为：类别 置信度 左 上 右 下
    :param image_id: 图像id
    :param image: 图像对象
    :param class_names: 所有类别名字
    :param map_out_path: 文件保存路径
    :return: 无
    """
    letterbox_image = True
    f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")

    #   计算输入图片的高和宽
    image_shape = np.array(np.shape(image)[0:2])

    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    image = cvtColor(image)

    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    image_data = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)

    #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        if use_cuda:
            images = images.cuda()

        #   将图像输入网络当中进行预测！
        outputs = net(images)

        #   利用预测结果进行解码
        outputs = decode_bbox(outputs[0], outputs[1], outputs[2], confidence, use_cuda)

        # -------------------------------------------------------#
        #   对于centernet网络来讲，确立中心非常重要。
        #   对于大目标而言，会存在许多的局部信息。
        #   此时对于同一个大目标，中心点比较难以确定。
        #   使用最大池化的非极大抑制方法无法去除局部框
        #   所以我还是写了另外一段对框进行非极大抑制的代码
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        # -------------------------------------------------------#
        results = postprocess(outputs, nms, image_shape, input_shape, letterbox_image, nms_iou)

        #   如果没有检测到物体，则返回原图
        if results[0] is None:
            return

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    # todo 我觉得这里应该按照置信度来sort，直接按照类别来排序是算什么回事，得看看 decode_bbox() 做了什么
    top_100 = np.argsort(top_label)[::-1][:max_boxes]  # 前一百的下标
    top_boxes = top_boxes[top_100]
    top_conf = top_conf[top_100]
    top_label = top_label[top_100]

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = str(top_conf[i])

        top, left, bottom, right = box

        if predicted_class not in class_names:
            continue

        f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

    f.close()
    return


print("\nGet mAP.")
for annotation_line in tqdm(test_lines):
    line = annotation_line.split()
    image_id = os.path.basename(line[0]).split('.')[0]
    #   读取图像并转换成RGB图像
    image = Image.open(line[0])
    #   获得预测框
    gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    #   获得预测txt
    get_map_txt(image_id, image)

    #   获得真实框txt
    with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
        for box in gt_boxes:
            left, top, right, bottom, obj = box
            obj_name = class_names[obj]
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Calculate mAP.")
try:
    mAP = get_coco_map(class_names=class_names, path=map_out_path)[1]
except:
    mAP = get_map(MINOVERLAP, False, path=map_out_path)
print("Get mAP done.")
shutil.rmtree(map_out_path)
# </editor-fold>

# <editor-folder desc="保存结果">
avr_loss = loss / steps
wh_loss = wh_loss / steps
offset_loss = offset_loss / steps
print('\n')
loss_str = 'loss: %.3f, wh loss: %.3f, offset loss: %.3f ' % (avr_loss, wh_loss, offset_loss)
mAP_str = 'mAP: %.4f ' % mAP
single_time = total_time / len(test_lines)
inference_time_str = 'time: %.2f ms, fps: %.2f' % (single_time * 1000, 1/single_time)
print(loss_str)
print(mAP_str)
print(inference_time_str)
print(net_summary_str)
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
save_dir = os.path.join(save_dir, 'test_' + str(time_str))
with open(save_dir + '.txt', 'a') as f:
    f.write(loss_str)
    f.write('\n')
    f.write(mAP_str)
    f.write('\n')
    f.write(inference_time_str)
    f.write('\n')
    f.write(net_summary_str)
# </editor-fold>
