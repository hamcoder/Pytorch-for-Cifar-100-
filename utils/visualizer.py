import torch.nn.functional as F
import torchvision.utils as utils


class Visualizer(object):
    def __init__(self, writer, model, mode='feature_map'):
        self.writer = writer
        self.model = model
        self.mode = mode

    def start(self, x):
        if self.mode == 'feature_map':
            self.visual_feature_map(x)
        elif self.mode == 'parameters':
            self.visual_parameters()
        elif self.mode == 'net_architecture':
            self.visual_net_architecture(x)

    def visual_feature_map(self, img):
        x = img
        img_grid = utils.make_grid(x, normalize=True, scale_each=True, nrow=2)
        self.writer.add_image('raw img', img_grid, global_step=666)
        print(x.size())

        self.model.eval()
        for name, layer in self.model._modules.items():
            x = x.view(x.size(0), -1) if "fc" in name else x
            print(x.size())

            x = layer(x)
            print(f'{name}')

            x = F.relu(x) if 'conv' in name else x
            if 'layer' in name or 'conv' in name:
                x1 = x.transpose(0, 1)
                img_grid = utils.make_grid(x1, normalize=True, scale_each=True, nrow=4)
                self.writer.add_image(f'{name}_feature_map)', img_grid, global_step=0)

    def visual_parameters(self):
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if 'bn' not in name:
                self.writer.add_histogram(name, param, 0)
            if 'conv' in name and 'weight' in name:
                in_channels = param.size()[1]
                k_w, k_h = param.size()[3], param.size()[2]
                kernel_all = param.view(-1, 1, k_w, k_h)
                kernel_grid = utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
                writer.add_image(f'{name}_all', kernel_grid, global_step=0)

    def visual_net_architecture(self, x):
        dummy_input = torch.autograd.Variable(x)
        with SummaryWriter(comment='ResNet18') as w:
            w.add_graph(model, (dummy_input,))