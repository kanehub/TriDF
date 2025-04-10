import torch
import os
from loguru import logger

from internal.modeling.image_encoder import ImageEncoder
from internal.tridf import TriDF


class TriDFModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_net = ImageEncoder().to(device)
        self.net = TriDF(
            ref_feat_dim = self.feature_net.output_dim
        ).to(device)


        self.device = device
        self.max_steps = args.n_iters


        self.optimizer = self.get_optimizer()

        if args.multi_step_lr:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    self.max_steps // 2,
                    self.max_steps * 3 // 4,
                    self.max_steps * 5 // 6,
                    self.max_steps * 9 // 10,
                ],
                gamma=0.6,

            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.lr_decay_step,
                                                             gamma=args.lr_decay_gamma, last_epoch=-1)

        self.out_folder = os.path.join(args.ckpt_dir, args.expname)
        self.start_step = self.load_from_ckpt(self.out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

    def get_optimizer(self):
        params_list = []
        lr = self.args.lr
        params_list.append(
            dict(
                params=self.net.encoding.parameters(),
                lr=lr * self.args.triplane_lr_scale,
            )
        )
        params_list.append(
            dict(params=self.net.direction_encoding.parameters(), lr=lr)
        )

        params_list.append(dict(params=self.net.mlp_density.parameters(), lr=lr))
        params_list.append(dict(params=self.net.mlp_base.parameters(), lr=lr))
        params_list.append(dict(params=self.net.mlp_head.parameters(), lr=lr))

        if self.args.finetune_encoder:
            params_list.append(dict(params=self.feature_net.parameters(), lr=lr))
        else:
            self.feature_net.eval()

        optim = torch.optim.AdamW(
            params_list,
            weight_decay=1e-5,
            eps=1e-8,
        )
        return optim

    def save_model(self, filename):
        to_save = {
            'optimizer': self.optimizer.state_dict(),
            'network': self.net.state_dict(),
            'scheduler':self.scheduler.state_dict()
        }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt and 'optimizer' in to_load and to_load['optimizer'] is not None:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler and 'scheduler' in to_load and to_load['scheduler'] is not None:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net.load_state_dict(to_load['network'])


    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if out_folder is not None and os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step

    def load_spc_ckpt(self):
        filename = os.path.join(self.out_folder,'spc','model_init.pth')
        if os.path.exists(filename):
            print(f"loading initialization spc ckpt from {filename}")
            logger.info(f"loading initialization spc ckpt from {filename}")
            self.load_model(filename, load_opt=False, load_scheduler=False)
        else:
            print("No spc ckpt found!")
            logger.info("No spc ckpt found!")
            raise ValueError("No spc ckpt found!")

    def switch_to_eval(self):
        self.net.eval()
        if self.args.finetune_encoder:
            self.feature_net.eval()

    def switch_to_train(self):
        self.net.train()
        if self.args.finetune_encoder:
            self.feature_net.eval()

