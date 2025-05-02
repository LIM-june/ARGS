"""PIOWave training and testing
"""
from collections import defaultdict
import os
import argparse
from shutil import copyfile
import yaml
from tqdm import tqdm
from datetime import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets_loader import WaveLoader
import wandb

from model import AVRModel, AVRModel_complex
from renderer import AVRRender
from utils.metric import metric_cal
from utils.logger import logger_config, log_inference_figure, plot_and_save_figure
from utils.criterion import Criterion


class AVR_Runner():
    def __init__(self, mode, dataset_dir, ckpt, log, **kwargs) -> None:
        # Seperate each settings
        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['model']
        kwargs_train = kwargs['train']
        self.kwargs_train = kwargs['train']

        # Path settings
        self.expname = kwargs_path['expname']
        self.dataset = kwargs_path['dataset']
        self.logdir = kwargs_path['logdir']
        self.time = datetime.now().strftime("%m%d-%H%M%S")
        self.runname = os.path.join(self.dataset, self.expname, self.time)
        self.savedir = os.path.join(self.logdir, self.runname)

        # Logger
        self.logger = logger_config(
            log_savepath=os.path.join(self.logdir, "logger.log"),
            logging_name=self.runname
        )
        self.logger.info(
            "\n⭐️ LOGDIR:%s, DATASET:%s, EXPNAME:%s, TIME:%s ⭐️",
            self.logdir, self.dataset, self.expname, self.time)

        # tensorboard writer
        self.log = True if log.lower() == 'on' else False
        if self.log and mode == 'train':
            tensorboard_dir = os.path.join(
                'tensorboard_logs', self.runname)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)

        # wandb logger
            self.wandb = wandb.init(
                name=self.runname,
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="LIM-june",
                # Set the wandb project where this run will be logged.
                project=f"ARGS",
                # Track hyperparameters and run metadata.
                config=kwargs,
            )

        # Image save path
            os.makedirs(os.path.join(self.savedir, 'img_train'), exist_ok=True)
            os.makedirs(os.path.join(self.savedir, 'img_test'), exist_ok=True)
            os.makedirs(os.path.join(
                self.savedir, 'img_test_energy'), exist_ok=True)

        # network and renderer
        self.fs = kwargs['render']['fs']

        if self.dataset == 'MeshRIR' or self.dataset == 'Simu':
            audionerf = AVRModel(kwargs_network)  # network
        elif self.dataset == 'RAF':
            audionerf = AVRModel_complex(kwargs_network)  # network

        self.renderer = AVRRender(
            networks_fn=audionerf, **kwargs_render)  # renderer

        # multi gpu
        self.devices = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            self.renderer = nn.DataParallel(self.renderer)
        self.renderer = self.renderer.cuda()

        # Optimization
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=float(self.kwargs_train['lr']),
                                          weight_decay=float(
                                              self.kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))

        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                     T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                     last_epoch=-1)

        # Print total number of parameters
        params = list(self.renderer.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        # Train Settings
        self.current_iteration = 1
        self.load_checkpoints(ckpt)
        self.batch_size = kwargs_train['batch_size']
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']
        self.val_freq = kwargs_train['val_freq']

        # dataloader
        self.train_set = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset,
                                    eval=False, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])
        self.test_set = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset,
                                   eval=True, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])
        self.train_set_show = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset,
                                         eval=False, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])

        self.train_iter = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_iter = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.train_iter_show = DataLoader(
            self.train_set_show, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.logger.info("Train set size:%d, Test set size:%d",
                         len(self.train_set), len(self.test_set))

        # loss settings
        self.criterion = Criterion(kwargs_train)

    def load_checkpoints(self, ckpt):
        """load previous checkpoints
        """

        # Skip loading
        if ckpt is None:
            self.logger.info('No checkpoint given. Train from scratch')
            return

        # Load from checkpoint path
        if os.path.exists(ckpt) and ckpt.endswith('.tar'):
            ckpt_path = ckpt
        elif os.path.exists(ckpt+'.tar'):
            ckpt_path = ckpt+'.tar'

        # Load the latest checkpoint
        else:
            # Default
            dir = os.path.join(self.logdir, self.dataset, self.expname)
            target = '.tar'

            # Specify model name
            if os.path.exists(os.path.join(self.logdir, self.dataset, ckpt)):
                dir = os.path.join(self.logdir, self.dataset, ckpt)
            # Specify runname
            elif os.path.exists(os.path.join(dir, ckpt)):
                dir = os.path.join(dir, ckpt)
            # Specify iteration number
            elif ckpt.isdigit() or ckpt.endswith('.tar'):
                target = f'{int(ckpt.split(".")[0]):06d}.tar'
            # Specify latest
            elif ckpt != '-1':
                self.logger.info('No checkpoint directory found, so skipping')
                return

            # Recursive search for ckpt
            ckpts = []
            for dir, _, files in os.walk(dir):
                for file in files:
                    if file.endswith(target):
                        ckpts.append(os.path.join(dir, file))

            if len(ckpts) == 0:
                self.logger.info('No checkpoints found, so skipping')
                return

            self.logger.info('Found ckpts:')
            ckpts = sorted(ckpts)
            for ckpt in ckpts:
                self.logger.info(ckpt)
            ckpt_path = ckpts[-1]

        self.logger.info('Loading ckpt %s', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.devices)

        try:
            self.renderer.load_state_dict(
                ckpt['audionerf_network_state_dict'])
        except:
            self.renderer.module.load_state_dict(
                ckpt["audionerf_network_state_dict"])

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.renderer.parameters()), lr=float(self.kwargs_train['lr']),
                                          weight_decay=float(
            self.kwargs_train['weight_decay']),
            betas=(0.9, 0.999))
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=float(
            self.kwargs_train['T_max']), eta_min=float(self.kwargs_train['eta_min']))
        self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_iteration = ckpt['current_iteration']

    def save_checkpoint(self):
        """save model checkpoint

        Returns
        -------
        checkpoint name
        """
        ckptsdir = os.path.join(self.savedir, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)

        ckptname = os.path.join(
            ckptsdir, '{:06d}.tar'.format(self.current_iteration))

        if torch.cuda.device_count() > 1:
            state_dict = self.renderer.module.state_dict()
        else:
            state_dict = self.renderer.state_dict()

        torch.save({
            'current_iteration': self.current_iteration,
            'audionerf_network_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname

    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d",
                         self.current_iteration)

        while self.current_iteration <= self.total_iterations:
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_batch in self.train_iter:
                    if self.dataset == "RAF":
                        ori_sig, position_rx, position_tx, direction_tx = train_batch
                        pred_sig = self.renderer(
                            position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                    else:
                        ori_sig, position_rx, position_tx = train_batch
                        pred_sig = self.renderer(
                            position_rx.cuda(), position_tx.cuda())

                    pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
                    ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                    losses, *_ = self.calculate_metrics(
                        pred_sig, ori_sig, self.fs, False)

                    if torch.isnan(losses['energy']).item():
                        print("Nan loss detected")
                        continue

                    total_loss = losses['total']

                    self.optimizer.zero_grad()
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.renderer.parameters(), max_norm=1)
                    for param in self.renderer.parameters():
                        if param.grad is not None:
                            with torch.no_grad():
                                param.grad[param.grad != param.grad] = 0
                                param.grad[torch.isinf(param.grad)] = 0

                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1

                    if self.log and self.current_iteration % 20 == 0:
                        self.writer.add_scalar(
                            f'train_loss', total_loss.detach(), self.current_iteration)
                        self.wandb.log(
                            {"train_loss": total_loss.detach()}, step=self.current_iteration)

                        for param_group in self.optimizer.param_groups:
                            current_lr = param_group['lr']
                            self.writer.add_scalar(
                                f'learning rate', current_lr, self.current_iteration)
                            self.wandb.log(
                                {"learning rate": current_lr}, step=self.current_iteration)

                    pbar.update(1)
                    pbar.set_description(
                        f"{self.expname} Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str(''.join([f'{key} = {val:.3f}, ' for key, val in losses.items(
                    )]) + f'lr = {self.optimizer.param_groups[0]["lr"]:.6f}')

                    if self.current_iteration % self.save_freq == 0:
                        ckptname = self.save_checkpoint()
                        pbar.write('Saved checkpoints at {}'.format(ckptname))

                    if self.current_iteration % self.val_freq == 0:
                        self.logger.info("Start evaluation")
                        self.renderer.eval()

                        avg_losses = defaultdict(float)
                        avg_metrics = defaultdict(float)

                        for check_idx, test_batch in enumerate(self.test_iter):
                            with torch.no_grad():
                                if self.dataset == "RAF":
                                    ori_sig, position_rx, position_tx, direction_tx = test_batch
                                    pred_sig = self.renderer(
                                        position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx = test_batch
                                    pred_sig = self.renderer(
                                        position_rx.cuda(), position_tx.cuda())

                                pred_sig = pred_sig[..., 0] + \
                                    1j * pred_sig[..., 1]
                                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                                losses, metrics, ori_time, pred_time = self.calculate_metrics(
                                    pred_sig, ori_sig, self.fs, True)

                            for key in losses:
                                avg_losses[key] += losses[key].detach()

                            for key in metrics:
                                avg_metrics[key] += metrics[key]

                            if check_idx < self.kwargs_train['n_imgs']:
                                save_dir = os.path.join(
                                    self.savedir, f'img_test/{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')
                                plot_and_save_figure(pred_sig[0, :], ori_sig[0, :], pred_time[0, :], ori_time[0, :],
                                                     position_rx[0, :], position_tx[0, :], mode_set='test', save_path=save_dir)

                                save_dir = os.path.join(
                                    self.savedir, f'img_test_energy/{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')
                                log_inference_figure(ori_time.detach().cpu().numpy()[0, :], pred_time.detach(
                                ).cpu().numpy()[0, :], metrics=metrics, save_dir=save_dir)

                        num_batches = len(self.test_iter)
                        avg_losses = {
                            key: val / num_batches for key, val in avg_losses.items()}
                        avg_metrics = {
                            key: val / num_batches for key, val in avg_metrics.items()}

                        if self.log:
                            self.log_all(
                                losses=avg_losses, metrics=avg_metrics, cur_iter=self.current_iteration, mode_set="test")
                        self.logger.info(
                            "Evaluations. Current Iteration:%d", self.current_iteration)

                        self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format(
                            avg_metrics['Angle'], avg_metrics['Amplitude'], avg_metrics['Envelope'], avg_metrics['T60'], avg_metrics['C50'], avg_metrics['EDT'], avg_metrics['multi_stft']))

                        avg_losses = defaultdict(float)
                        avg_metrics = defaultdict(float)

                        for check_idx, train_iter_batch in enumerate(self.train_iter_show):
                            with torch.no_grad():
                                if self.dataset == "RAF":
                                    ori_sig, position_rx, position_tx, direction_tx = train_iter_batch
                                    pred_sig = self.renderer(
                                        position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx = train_iter_batch
                                    pred_sig = self.renderer(
                                        position_rx.cuda(), position_tx.cuda())

                                pred_sig = pred_sig[..., 0] + \
                                    1j * pred_sig[..., 1]
                                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                                losses, metrics, ori_time, pred_time = self.calculate_metrics(
                                    pred_sig, ori_sig, self.fs, True)

                            for key in losses:
                                avg_losses[key] += losses[key].detach()

                            for key in metrics:
                                avg_metrics[key] += metrics[key]

                            if check_idx < self.kwargs_train['n_imgs']:
                                save_dir = os.path.join(
                                    self.savedir, f'img_train/{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')
                                plot_and_save_figure(pred_sig[0, :], ori_sig[0, :], pred_time[0, :], ori_time[0, :],
                                                     position_rx[0, :], position_tx[0, :], mode_set='train', save_path=save_dir)

                            if check_idx > 3000 or check_idx == len(self.train_iter_show) - 1:

                                num_batches = check_idx + 1
                                avg_losses = {
                                    key: val / num_batches for key, val in avg_losses.items()}
                                avg_metrics = {
                                    key: val / num_batches for key, val in avg_metrics.items()}

                                if self.log:
                                    self.log_all(
                                        losses=avg_losses, metrics=avg_metrics, cur_iter=self.current_iteration, mode_set="train")

                                self.logger.info("Evaluations on training set")
                                self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format(
                                    avg_metrics['Angle'], avg_metrics['Amplitude'], avg_metrics['Envelope'], avg_metrics['T60'], avg_metrics['C50'], avg_metrics['EDT'], avg_metrics['multi_stft']))

                                break

                        self.renderer.train()

    def calculate_metrics(self, pred_sig, ori_sig, fs, return_metrics=False):
        """ calculate the metrics and losses
        """
        # Loss calculation
        spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time = self.criterion(
            pred_sig, ori_sig)

        total_loss = spec_loss + amplitude_loss + angle_loss + \
            time_loss + energy_loss + multi_stft_loss

        losses = {
            'total': total_loss,
            'spec': spec_loss,
            'fft': amplitude_loss + angle_loss,
            'time': time_loss,
            'energy': energy_loss,
            'multi_stft': multi_stft_loss
        }

        # Metrics calculation
        if return_metrics:
            angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft, _, _ = metric_cal(
                ori_time.detach().cpu().numpy(),
                pred_time.detach().cpu().numpy(),
                fs=fs
            )

            metrics = {
                'Angle': angle_error,
                'Amplitude': amp_error,
                'Envelope': env_error,
                'T60': t60_error,
                'C50': C50_error,
                'EDT': edt_error,
                'multi_stft': multi_stft
            }

            return losses, metrics, ori_time, pred_time

        else:
            return losses, ori_time, pred_time

    def log_all(self, losses=None, metrics=None, cur_iter=None, mode_set="train"):
        for loss_name, value in losses.items():
            self.writer.add_scalar(
                f'{mode_set}_loss/{loss_name}', value, cur_iter)
            self.wandb.log(
                {f'{mode_set}_loss/{loss_name}': value}, step=cur_iter)

        for metric_name, value in metrics.items():
            self.writer.add_scalar(
                f'{mode_set}_metric/{metric_name}', value, cur_iter)
            self.wandb.log(
                {f'{mode_set}_metric/{metric_name}': value}, step=cur_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='on')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str,
                        default='avr.yml', help='config file path')
    parser.add_argument('--dataset_dir', type=str, default='S1-M3969_npy')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--idx', type=string_to_float_list)
    parser.add_argument('--pos-rx', type=string_to_float_list)
    parser.add_argument('--pos-tx', type=string_to_float_list)
    parser.add_argument('--dir-tx', type=string_to_float_list)
    parser.add_argument('--viz', type=str)
    args = parser.parse_args()

    if args.mode == 'train':  # specify the config yaml
        with open(args.config, 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
    elif args.mode == 'test':  # specify the dict of the config yaml
        with open(os.path.join(args.config, 'avr_conf.yml'), 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

    # backup config file
    logdir = kwargs['path']['logdir']
    os.makedirs(logdir, exist_ok=True)

    # Log the command to the file
    logfile_path = "command_log.txt"
    command = " ".join(sys.argv)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(logdir, logfile_path), "a") as logfile:
        logfile.write(f"[{current_time}] : {command}\n")

    # Construct the destination file path
    dest_file_path = os.path.join(logdir, 'avr_conf.yml')

    # Check if the source and destination paths are the same
    if os.path.abspath(args.config) != os.path.abspath(dest_file_path):
        copyfile(args.config, dest_file_path)
    else:
        print("Source and destination are the same, skipping copy.")

    print()
    if args.mode == 'train':
        print("Training mode selected.")
    elif args.mode == 'infer':
        print("Inference mode selected.")
        if args.ckpt is None:
            args.ckpt = '-1'
            print("Please specify a checkpoint to load for inference.")
            print("Defaulting to the latest checkpoint.")
    else:
        print("Invalid mode. Please choose 'train' or 'infer'.")
        sys.exit(1)

    worker = AVR_Runner(
        mode=args.mode, dataset_dir=args.dataset_dir, ckpt=args.ckpt, log=args.log, **kwargs)

    worker.train()
