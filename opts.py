import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="UCF101")
    parser.add_argument('--work-dir', default='./work_dir/', \
                        type=str, help='the work folder for storing results')
    parser.add_argument('--config', default='./work_dir/config.yaml', \
                        type=str, help='path to the configure file')

    # ==================== Dataset Configs ======================
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)

    # =================== Model Configs =====================
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--model-args', type=dict, default=dict(), \
                        metavar='MA', help='the arguments of model')

    # =================== Optimizer Configs ===================
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
    parser.add_argument('--lr-decay', default=1e-4, type=float)
    parser.add_argument('--final-lr', default=0.1, type=float)

    # =================== Learning Configs ====================
    parser.add_argument('--scheduler', default='step_lr', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step-size', default=10, type=int)
    parser.add_argument('--milestones', default=[30, 50], type=int, nargs='+')

    # =================== Monitor Configs =====================
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--eval-freq', '-ef', default=5, type=int)
    parser.add_argument('--print-log', default=True, type=str2bool)

    # =================== Visualize Configs =====================
    parser.add_argument('--mode', default='feature_map', type=str)
    parser.add_argument('--sample_idx', default=5, type=int)

    # =================== Runtime Configs =====================
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--snapshot_pref', type=str, default='sign')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default='')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')