"""
训练参数设置注意事项说明

# 1. 预训练
a. 权值文件的下载请看README
b. 预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
c. 如果想要让模型从主干的预训练权值开始训练，pretrained_model_path = ''，use_pretrained_backbone = True，此时仅加载主干。
d. 如果想要让模型从0开始训练，则设置pretrained_model_path = ''，use_pretrained_backbone = Fasle，freeze_train = Fasle，此时从0开始训练，且没有冻结主干的过程。
一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。

# 2. 中断续练
1. 将pretrained_model_path设置成logs文件夹下指定epoch的pth文件
2. 设置init_epoch=指定epoch-1，设置epoch=你想要的总epoch
3. 设置log的保存路径，应该与之前的pth文件处于同一个文件夹
4. ！！！！设置train.py的模型创建，将模型改为你想要的模型！！！

# 3. 冻结

"""


class Opts(object):
    def __init__(self,
                 use_cuda=True,
                 use_fp16=True,
                 classes_path='model_data/ssdd_classes.txt',
                 pretrained_model_path='',
                 input_shape=None,

                 backbone="resnet50",
                 use_pretrained_backbone=True,

                 init_epoch=0,
                 freeze_epoch=200,
                 freeze_batch_size=4,
                 epoch=1000,
                 batch_size=32,
                 freeze_backbone=True,

                 init_lr=5e-4,
                 min_lr=5e-6,
                 lr_decay_type='cos',

                 optimizer_type="adam",
                 momentum=0.9,
                 weight_decay=0,
                 save_period=50,
                 eval_period=1,
                 save_dir='logs/test',
                 use_eval=True,
                 num_workers=8,
                 train_annotation_path='dataset/SSDD/ImageSets/Exp1/train.txt',
                 val_annotation_path='dataset/SSDD/ImageSets/Exp1/val.txt'):
        """
        :param use_cuda: 使用Cuda，没有GPU可以设置成False
        :param use_fp16: 是否使用混合精度训练，可减少约一半的显存、需要pytorch1.7.1以上
        :param classes_path: 自己训练的数据集的类别描述文件.txt路径
        :param pretrained_model_path: 模型的预训练权重文件.pth路径，当model_path = ''的时候不加载整个模型的权值。
        :param input_shape: 输入的shape大小，32的倍数
        :param backbone: 主干特征提取网络的选择，resnet50和hourglass
        :param use_pretrained_backbone: 使用主干网络的预训练权重，设置了pretrained_model_path，则主干的权值无需加载，use_pretrained_backbone的值无意义。

        :param init_epoch: 模型当前开始的训练epoch
        :param freeze_epoch: 模型冻结训练的epoch
        :param freeze_batch_size: 模型冻结训练的batch_size
        :param epoch: 模型总共训练的epoch
        :param batch_size: 模型在默认的batch_size
        :param freeze_backbone: 训练时，冻结backbone

        :param init_lr: 模型的最大（初始）学习率
        :param min_lr: 模型的最小学习率
        :param lr_decay_type: 使用到的学习率下降方式，可选的有'step'、'cos'
        :param optimizer_type: 使用到的优化器种类，可选的有adam、sgd
        :param momentum: 优化器内部使用到的momentum参数
        :param weight_decay: 权值衰减，可防止过拟合

        :param save_period: 多少个epoch保存一次权值
        :param eval_period: 多少个epoch评估一次
        :param save_dir: 权值与日志文件保存的文件夹
        :param use_eval: 训练时进行评估，评估对象为验证集
        :param num_workers: 用于设置是否使用多线程读取数据，1代表关闭多线程
        :param train_annotation_path: 训练图片和标签.txt文件路径
        :param val_annotation_path: 验证图片和标签.txt文件路径
        """
        self.use_cuda = use_cuda
        self.use_fp16 = use_fp16
        self.classes_path = classes_path
        self.pretrained_model_path = pretrained_model_path
        self.input_shape = input_shape if input_shape is not None else [512, 512]
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.init_epoch = init_epoch
        self.freeze_epoch = freeze_epoch
        self.freeze_batch_size = freeze_batch_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_type = lr_decay_type
        self.save_period = save_period
        self.eval_period = eval_period
        self.save_dir = save_dir
        self.use_eval = use_eval
        self.num_workers = num_workers
        self.train_annotation_path = train_annotation_path
        self.val_annotation_path = val_annotation_path
