import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.centernet_resnet50 import CenterNetResnet50
from nets.centernet_hourglassnet import CenterNetHourglassNet
from nets.centernet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from opts import Opts
from utils.utils import download_weights, get_classes, show_opts
from utils.utils_fit import fit_one_epoch

from nets.centernet_dense_connection import CenterNetDenseConnection

if __name__ == "__main__":
    # <editor-fold desc="参数设置">
    opts = Opts()
    show_opts(vars(opts))
    # </editor-fold>

    # <editor-fold desc="训练准备">
    # <editor-folder desc="显卡">
    available_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # </editor-fold>

    # <editor-folder desc="p16配置">
    if opts.use_fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    # </editor-fold>

    # <editor-folder desc="下载预训练权重">
    if opts.use_pretrained_backbone:
        download_weights(opts.backbone, model_dir="./model_data")
    # </editor-fold>

    # <editor-folder desc="获取classes">
    class_names, num_classes = get_classes(opts.classes_path)
    # </editor-fold>

    # <editor-folder desc="创建 Network">
    if opts.backbone == "resnet50":
        model = CenterNetResnet50(num_classes, backbone_pretrained=opts.use_pretrained_backbone)
        # model = CenterNetDenseConnection(num_classes, backbone_pretrained=opts.use_pretrained_backbone)
    else:
        model = CenterNetHourglassNet({'hm': num_classes, 'wh': 2, 'reg': 2}, pretrained=opts.use_pretrained_backbone)
    if opts.pretrained_model_path != '':
        print('Load weights {}.'.format(opts.pretrained_model_path))
        # 根据预训练权重的Key和模型的Key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opts.pretrained_model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    model_train = model.train()
    if opts.use_cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # </editor-fold>

    # <editor-folder desc="记录Loss">
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(opts.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=opts.input_shape)
    # </editor-fold>

    # <editor-folder desc="读取数据集对应的txt">
    with open(opts.train_annotation_path) as f:
        train_lines = f.readlines()
    with open(opts.val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    # </editor-fold>

    # <editor-folder desc="打印epoch、batch_size设置建议">
    wanted_step = 5e4 if opts.optimizer_type == "sgd" else 1.5e4
    total_step = num_train // opts.batch_size * opts.epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // opts.batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (opts.optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, opts.batch_size, opts.epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    UnFreeze_flag = False
    batch_size = opts.batch_size
    if opts.freeze_backbone:
        # 冻结backbone训练
        model.freeze_backbone()
        batch_size = opts.freeze_batch_size
    # </editor-fold>

    # <editor-folder desc="配置 Optimizer">
    nbs = 64
    lr_limit_max = 5e-4 if opts.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 2.5e-4 if opts.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * opts.init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * opts.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    # 根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(opts.momentum, 0.999),
                           weight_decay=opts.weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=opts.momentum, nesterov=True,
                         weight_decay=opts.weight_decay)
    }[opts.optimizer_type]
    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(opts.lr_decay_type, Init_lr_fit, Min_lr_fit, opts.epoch)
    # </editor-fold>

    # <editor-folder desc="配置 Data Loader">
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    train_dataset = CenternetDataset(train_lines, opts.input_shape, num_classes, train=True)
    val_dataset = CenternetDataset(val_lines, opts.input_shape, num_classes, train=False)
    train_sampler = None
    val_sampler = None
    shuffle = True
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opts.num_workers,
                     pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opts.num_workers,
                         pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler)
    # </editor-fold>

    # <editor-folder desc="记录eval的map曲线">
    eval_callback = EvalCallback(model, opts.backbone, opts.input_shape, class_names, num_classes, val_lines, log_dir,
                                 opts.use_cuda, eval_flag=opts.use_eval, period=opts.eval_period)
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="训练">
    for epoch in range(opts.init_epoch, opts.epoch):
        # 满足要求则解冻
        if epoch >= opts.freeze_epoch and not UnFreeze_flag and opts.freeze_backbone:
            batch_size = opts.batch_size
            # 判断当前batch_size，自适应调整学习率
            nbs = 64
            lr_limit_max = 5e-4 if opts.optimizer_type == 'adam' else 5e-2
            lr_limit_min = 2.5e-4 if opts.optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * opts.init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * opts.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            # 获得学习率下降的公式
            lr_scheduler_func = get_lr_scheduler(opts.lr_decay_type, Init_lr_fit, Min_lr_fit, opts.epoch)
            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opts.num_workers,
                             pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate,
                             sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opts.num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate,
                                 sampler=val_sampler)
            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, opts.epoch, opts.use_cuda, opts.use_fp16, scaler, opts.backbone, opts.save_period,
                      opts.save_dir)

    loss_history.writer.close()
    # </editor-fold>
