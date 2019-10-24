import os
import yaml
from datetime import datetime
import collections

import torch
import torch.nn as nn

from opts import get_parser


class Processor():
    def __init__(self, args):
        self.args = args
        self._save_arg()

        if self.args.phase == 'train' or self.args.phase == 'visualize':
            self._load_logger()

        self.data_loader = {}
        if self.args.phase == 'train':
            self._load_train_data()
            self._load_test_data()
        elif self.args.phase == 'test':
            self._load_test_data()
        elif self.args.phase == 'visualize':
            self._load_visualize_data()

        self._load_model()
        self._load_evaluator()

        if self.args.phase == 'train' or self.args.phase == 'lr_finder':
            self._load_criterion()
            self._load_optimizer()

        if self.args.phase == 'train':
            self._load_trainer()
        elif self.args.phase == 'lr_finder':
            self._load_lr_finder()
        elif self.args.phase == 'test':
            self._load_tester()
        elif self.args.phase == 'visualize':
            self._load_visualizer()

    def _load_logger(self):
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(
            'runs', self.args.model, datetime.now().isoformat()))

    def _load_train_data(self):
        print('loading train data...')

        import torchvision.transforms as transforms

        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        import torchvision
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform)
        self.data_loader['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,
                                                                shuffle=True, num_workers=self.args.workers,
                                                                pin_memory=True)

        print('train data load finished!')

    def _load_test_data(self):
        print('loading test data...')

        import torchvision.transforms as transforms

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        import torchvision
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform)
        self.data_loader['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size,
                                                               shuffle=False, num_workers=self.args.workers,
                                                               pin_memory=True)

        print('test data load finished!')

    def _load_visualize_data(self):
        print('loading visualize data...')

        import torchvision.transforms as transforms

        visualize_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        import torchvision
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=visualize_transform)
        self.data_loader['visualize'] = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size,
                                                               shuffle=False, num_workers=self.args.workers,
                                                               pin_memory=True)

        print('visualize data load finished!')

    def _load_model(self):
        print('loading model...')

        if self.args.model == 'resnet':
            from models.resnet import resnet
            self.model = resnet(**self.args.model_args)
        elif self.args.model == 'densenet':
            from models.densenet import densenet
            self.model = densenet(**self.args.model_args)

        self.policies = self.model.parameters()

        #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus).cuda()

        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(("=> loading checkpoint '{}'".format(self.args.resume)))
                checkpoint = torch.load(self.args.resume)
                d = collections.OrderedDict()
                for key, value in checkpoint['state_dict'].items():
                    tmp = key[7:]
                    d[tmp] = value
                self.args.start_epoch = checkpoint['epoch']
                #self.model.load_state_dict(checkpoint['state_dict'])
                self.model.load_state_dict(d)
                print(("=> loaded checkpoint '{}' (epoch {})".format(self.args.phase, checkpoint['epoch'])))
            else:
                print(("=> no checkpoint found at '{}'".format(self.args.resume)))

        print('model load finished!')

    def _load_evaluator(self):
        print('loading evaluator...')

        from utils.evaluator import Evaluator
        self.evaluator = Evaluator()

        print('evaluator load finished!')

    def _load_criterion(self):
        print('loading criterion...')

        self.criterion = nn.CrossEntropyLoss()

        print('criterion load finished!')

    def _load_optimizer(self):
        print('loading optimizer...')

        import torch.optim as optim

        if self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.policies, lr=self.args.lr,
                                            weight_decay=self.args.wd)
        elif self.args.optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.policies, lr=self.args.lr,
                                           weight_decay=self.args.wd)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.policies, lr=self.args.lr,
                                        weight_decay=self.args.wd)
        elif self.args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policies, lr=self.args.lr,
                                           momentum=self.args.momentum,
                                           weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.policies, lr=self.args.lr,
                                       momentum=self.args.momentum, dampening=0,
                                       nesterov=self.args.nesterov, weight_decay=self.args.wd)
        elif self.args.optimizer == 'adabound':
            import adabound
            self.optimizer = adabound.AdaBound(self.policies, lr=self.args.lr, final_lr=self.args.final_lr)

        if self.args.scheduler == 'step_lr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.2,
                                                       last_epoch=-1)
        elif self.args.scheduler == 'multi_step_lr':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=0.2,
                                                            last_epoch=-1)

        print('optimizer load finished!')

    def _load_trainer(self):
        print('loading trainer...')

        from trainer import Trainer
        self.trainer = Trainer(self.args, self.writer, self.data_loader, self.model, \
                               self.evaluator, self.criterion, self.scheduler, self.optimizer)

        print('trainer load finished!')

    def _load_lr_finder(self):
        print('loading lr finder...')

        from utils.lr_finder import LRFinder
        self.lr_finder = LRFinder(self.args, self.data_loader, self.model, self.criterion, self.optimizer)

        print('lr finder load finished!')

    def _load_tester(self):
        print('loading tester...')

        from tester import Tester
        self.tester = Tester(self.args, self.writer, self.data_loader, self.model, self.evaluator)

        print('tester load finished!')

    def _load_visualizer(self):
        print('loading visualizer...')

        from utils.visualizer import Visualizer
        self.visualizer = Visualizer(self.writer, self.model, self.args.mode)

        print('visualizer load finished!')

    def start(self):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self._print_log('Parameters:\n{}\n'.format(str(vars(self.args))))
        if self.args.phase == 'train':
            self.trainer.start()
        elif self.args.phase == 'lr_finder':
            self.lr_finder.start()
        elif self.args.phase == 'test':
            self.tester.start()
        elif self.args.phase == 'visualize':
            for batch_idx, (image, label) in enumerate(self.data_loader['visualize'], 0):
                if batch_idx == self.args.sample_idx:
                    #image = image.cuda()
                    self.visualizer.start(image)
        else:
            raise ValueError

    def _print_log(self, s):
        print(s)
        if self.args.print_log:
            with open('{}/log.txt'.format(self.args.work_dir), 'a') as f:
                print(s, file=f)

    def _save_arg(self):
        arg_dict = vars(self.args)
        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)
        with open('{}/config.yaml'.format(self.args.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


def main():
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Wrong Arg: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()
    processor = Processor(args)
    processor.start()


if __name__ == '__main__':
    main()