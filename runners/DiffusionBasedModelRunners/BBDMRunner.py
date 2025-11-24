import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        print(f'optimizer = {str(config.model.BB.optimizer.optimizer)}')
        if config.model.BB.optimizer.optimizer=='AdamW':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   **vars(config.model.BB.lr_scheduler))
        
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                mode='min',
                                                                verbose=True,
                                                                threshold_mode='rel',
                                                                **vars(config.model.BB.lr_scheduler)
)
            print(f'actual scheduler optimizer = {str(scheduler.optimizer)}')
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            # unpack robustly
            x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name = self._unpack_batch(batch)
            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            if x_cond_herringbone is not None:
                x_cond_herringbone = x_cond_herringbone.to(self.config.training.device[0])
            else:
                # fallback zeros if missing
                x_cond_herringbone = torch.zeros_like(x_cond_sar)

            # encode
            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_sar_latent = self.net.encode(x_cond_sar, cond=True, normalize=False)
            x_cond_herringbone_latent = self.net.encode(x_cond_herringbone, cond=True, normalize=False)

            # concat to build 6-channel cond latent
            x_cond_latent = torch.cat([x_cond_sar_latent, x_cond_herringbone_latent], dim=1)

            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name = self._unpack_batch(batch)
            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            if x_cond_herringbone is not None:
                x_cond_herringbone = x_cond_herringbone.to(self.config.training.device[0])
            else:
                x_cond_herringbone = torch.zeros_like(x_cond_sar)

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_sar_latent = self.net.encode(x_cond_sar, cond=True, normalize=False)
            x_cond_herringbone_latent = self.net.encode(x_cond_herringbone, cond=True, normalize=False)
            x_cond_latent = torch.cat([x_cond_sar_latent, x_cond_herringbone_latent], dim=1)

            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def _unpack_batch(self, batch):
        """
        SỬA LỖI: Unpack batch từ dataloader (đã sửa)
        Format mong đợi: [ (x, x_name), (x_cond_sar, x_cond_sar_name), (x_cond_hb, x_cond_hb_name) ]
        """
        # default returns
        x = x_name = x_cond_sar = x_cond_sar_name = x_cond_herringbone = x_cond_herringbone_name = None

        # Kiểm tra format mới (3 mục)
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 3):
            # Nếu không phải 3 mục, thử logic cũ (fallback)
            print("Cảnh báo: _unpack_batch đang chạy logic cũ.")
            if isinstance(batch, (list, tuple)):
                first = batch[0]
                second = batch[1] if len(batch) > 1 else None
            else:
                raise ValueError("Batch has unexpected type: {}".format(type(batch)))
            # unpack first
            if isinstance(first, (list, tuple)):
                x = first[0]
                x_name = first[1] if len(first) > 1 else None
            else:
                x = first; x_name = None
            # unpack second
            if isinstance(second, (list, tuple)):
                x_cond_sar = second[0]
                x_cond_name = second[1] if len(second) > 1 else None
            else:
                x_cond_sar = second; x_cond_name = None
            
            # Trả về theo signature cũ, 'herringbone' sẽ là None
            return x, x_name, x_cond_sar, None, x_cond_name


        # ==========================================================
        # Logic mới cho 3 mục (ảnh, tên)
        # ==========================================================

        # Unpack first item (x, x_name)
        if isinstance(batch[0], (list, tuple)):
            x = batch[0][0]
            x_name = batch[0][1] if len(batch[0]) > 1 else None
        else:
            x = batch[0] # Fallback

        # Unpack second item (x_cond_sar, x_cond_sar_name)
        if isinstance(batch[1], (list, tuple)):
            x_cond_sar = batch[1][0]
            x_cond_sar_name = batch[1][1] if len(batch[1]) > 1 else None
        else:
            x_cond_sar = batch[1] # Fallback

        # Unpack third item (x_cond_herringbone, x_cond_herringbone_name)
        if isinstance(batch[2], (list, tuple)):
            x_cond_herringbone = batch[2][0]
            x_cond_herringbone_name = batch[2][1] if len(batch[2]) > 1 else None
        else:
            x_cond_herringbone = batch[2] # Fallback
            
        # Hàm loss_fn mong đợi: x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name
        # (x_cond_name cũ có lẽ là tên của SAR)
        return x, x_name, x_cond_sar, x_cond_herringbone, x_cond_sar_name
    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        # robust unpack
        x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name = self._unpack_batch(batch)

        x = x.to(self.config.training.device[0])
        x_cond_sar = x_cond_sar.to(self.config.training.device[0])
        if x_cond_herringbone is None:
            # create zeros as fallback so model call remains consistent
            x_cond_herringbone = torch.zeros_like(x_cond_sar)
        else:
            x_cond_herringbone = x_cond_herringbone.to(self.config.training.device[0])

        # call model: LatentBrownianBridgeModel.forward expects (x, x_cond_sar, x_cond_herringbone)
        loss, additional_info = net(x, x_cond_sar, x_cond_herringbone)

        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if isinstance(additional_info, dict):
                if additional_info.__contains__('recloss_noise'):
                    self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
                if additional_info.__contains__('recloss_xy'):
                    self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        # unpack robustly
        x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name = self._unpack_batch(batch)

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond_sar = x_cond_sar[0:batch_size].to(self.config.training.device[0])
        if x_cond_herringbone is None:
            x_cond_herringbone = torch.zeros_like(x_cond_sar)
        else:
            x_cond_herringbone = x_cond_herringbone[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        if self.config.testing.sample_num > 1:
            samples, one_step_samples = net.sample(x_cond_sar, x_cond_herringbone, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=True)
            # save mid-step samples
            self.save_images(samples, reverse_sample_path, grid_size, save_interval=200,
                             writer_tag=f'{stage}_sample' if stage != 'test' else None)
            self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200,
                             writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
            sample = samples[-1]
        else:
            sample = net.sample(x_cond_sar, x_cond_herringbone, clip_denoised=self.config.testing.clip_denoised)
            sample = sample.to('cpu')

        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        # save both condition images (sar and herringbone) separately
        image_grid_sar = get_image_grid(x_cond_sar.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid_sar)
        im.save(os.path.join(sample_path, 'condition_sar.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition_sar', image_grid_sar, self.global_step, dataformats='HWC')

        image_grid_hb = get_image_grid(x_cond_herringbone.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid_hb)
        im.save(os.path.join(sample_path, 'condition_herringbone.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition_herringbone', image_grid_hb, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        # SỬA ĐỔI: Đổi tên 'condition' thành 'condition_sar' cho rõ ràng
        condition_sar_path = make_dir(os.path.join(sample_path, f'condition_sar'))
        # THÊM MỚI: Thêm đường dẫn cho ảnh điều kiện "xương cá"
        condition_herringbone_path = make_dir(os.path.join(sample_path, f'condition_herringbone'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            # SỬA ĐỔI: Giải nén 3 mục thay vì 2
            x, x_name, x_cond_sar, x_cond_sar_name, x_cond_herringbone, x_cond_herringbone_name = None, None, None, None, None, None
            # Support both styles: ((x, x_name), (x_cond_sar, x_cond_herringbone, x_cond_name)) OR ((x,x_name),(x_cond_sar,x_cond_sar_name),(x_cond_hb,...))
            if isinstance(test_batch, (list, tuple)) and len(test_batch) == 3:
                # case: ((x,x_name), (x_cond_sar, x_cond_sar_name), (x_cond_herringbone, x_cond_herringbone_name))
                (x, x_name), (x_cond_sar, x_cond_sar_name), (x_cond_herringbone, x_cond_herringbone_name) = test_batch
            else:
                # fallback to using _unpack_batch then adjust names
                x, x_name, x_cond_sar, x_cond_herringbone, x_cond_name = self._unpack_batch(test_batch)
                x_cond_sar_name = x_cond_name
                x_cond_herringbone_name = None

            # SỬA ĐỔI: Chuyển cả 3 ảnh lên device
            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            if x_cond_herringbone is None:
                x_cond_herringbone = torch.zeros_like(x_cond_sar)
            else:
                x_cond_herringbone = x_cond_herringbone.to(self.config.training.device[0])

            for j in range(sample_num):
                # SỬA ĐỔI: Truyền cả 2 ảnh điều kiện vào hàm sample
                sample = net.sample(x_cond_sar, x_cond_herringbone, clip_denoised=False)
                # sample = net.sample_vqgan(x)
                for i in range(batch_size):
                    # SỬA ĐỔI: Lấy cả 2 ảnh điều kiện
                    condition_sar = x_cond_sar[i].detach().clone()
                    condition_herringbone = x_cond_herringbone[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    if j == 0:
                        # SỬA ĐỔI: Lưu ảnh SAR
                        name_sar = x_cond_sar_name[i] if (x_cond_sar_name is not None and len(x_cond_sar_name)>i) else f"sample_{i}"
                        save_single_image(condition_sar, condition_sar_path, f'{name_sar}.png', to_normal=to_normal)
                        # THÊM MỚI: Lưu ảnh "xương cá"
                        name_hb = x_cond_herringbone_name[i] if (x_cond_herringbone_name is not None and len(x_cond_herringbone_name)>i) else f"hb_{i}"
                        save_single_image(condition_herringbone, condition_herringbone_path, f'{name_hb}.png', to_normal=to_normal)
                        
                        gt_name = x_name[i] if (x_name is not None and len(x_name)>i) else f"gt_{i}"
                        save_single_image(gt, gt_path, f'{gt_name}.png', to_normal=to_normal)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.png', to_normal=to_normal)
