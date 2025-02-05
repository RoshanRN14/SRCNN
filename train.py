import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

import torch.nn.functional as F


def apply_gaussian_high_pass_spatial_torch(image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Applies a Gaussian High-Pass Filter in the spatial domain using PyTorch.
    
    Parameters:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation for the Gaussian kernel
    
    Returns:
        torch.Tensor: High-pass filtered image of the same shape as input
    """
    # Input validation
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Handle different input dimensions
    original_dim = image.dim()
    if original_dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif original_dim == 3:
        image = image.unsqueeze(0)  # Add batch dim
    
    # Calculate Gaussian kernel
    coords = torch.arange(kernel_size, device=image.device) - (kernel_size - 1) / 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
    # Expand kernel for each input channel
    kernel = kernel.expand(image.size(1), -1, -1, -1)
    
    # Apply padding
    padding = kernel_size // 2
    
    # Apply Gaussian blur using conv2d
    blurred = F.conv2d(
        image,
        kernel.to(image.device),
        padding=padding,
        groups=image.size(1)  # Apply separately to each channel
    )
    
    # Subtract blurred image from original to get high-pass filtered image
    high_pass_image = image - blurred
    
    # Restore original dimensions
    if original_dim == 2:
        high_pass_image = high_pass_image.squeeze()
    elif original_dim == 3:
        high_pass_image = high_pass_image.squeeze(0)
        
    return high_pass_image

def spatial_high_pass_loss(y_true, y_pred, kernel_size=5, sigma=1.0):
    """
    Computes the spatial high-pass loss as the mean absolute error between
    the high-pass filtered ground truth and predicted images in the spatial domain.

    Parameters:
        y_true (Tensor): Ground truth image tensor with shape [B, 1, H, W].
        y_pred (Tensor): Predicted image tensor with shape [B, 1, H, W].
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        loss (Tensor): Scalar tensor representing the mean absolute error.
    """
    # Apply spatial high-pass filter to ground truth and predicted images
    y_true_hp = apply_gaussian_high_pass_spatial_torch(y_true, kernel_size, sigma)
    y_pred_hp = apply_gaussian_high_pass_spatial_torch(y_pred, kernel_size, sigma)
    
    # Compute the loss (Mean Absolute Error)
    loss = torch.mean(torch.abs(y_true_hp - y_pred_hp))
    return loss


import torch
import torch.nn as nn

class FrequencyHighPassLoss(nn.Module):
    def __init__(self, cutoff=5):
        super(FrequencyHighPassLoss, self).__init__()
        self.cutoff = cutoff

    def gaussian_high_pass_filter(self, shape, device):
        rows, cols = shape[-2], shape[-1]
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device))
        distance = torch.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        filter = 1 - torch.exp(-(distance ** 2) / (2 * (self.cutoff ** 2)))
        return filter

    def apply_hann_window(self, x):
        B, C, H, W = x.shape
        window_x = torch.hann_window(W, periodic=True, dtype=x.dtype, device=x.device)
        window_y = torch.hann_window(H, periodic=True, dtype=x.dtype, device=x.device)
        window_2d = window_y.unsqueeze(1) * window_x.unsqueeze(0)
        window_2d = window_2d.expand(B, C, H, W)
        return x * window_2d

    def get_filtered_fourier_components(self, x):
        # Apply Hann windowing to reduce spectral leakage
        x_windowed = self.apply_hann_window(x)

        # Compute FFT
        fft = torch.fft.fft2(x_windowed)

        # Apply High-Pass Filter
        high_pass_filter = self.gaussian_high_pass_filter(x.shape, x.device)
        fft_filtered = fft * high_pass_filter

        # Extract Amplitude & Phase
        amplitude = torch.abs(fft_filtered)
        phase = torch.angle(fft_filtered)

        return amplitude, phase

    def forward(self, pred, target):
        # Compute High-Pass Filtered Amplitude & Phase
        pred_amp, pred_phase = self.get_filtered_fourier_components(pred)
        target_amp, target_phase = self.get_filtered_fourier_components(target)

        # Compute Amplitude & Phase Loss
        B, C, H, W = pred.shape
        U, V = H, W

        amplitude_loss = (2.0 / (U * V)) * torch.sum(
            torch.abs(pred_amp[:, :, :U//2, :V-1] - target_amp[:, :, :U//2, :V-1])
        )

        phase_loss = (2.0 / (U * V)) * torch.sum(
            torch.abs(pred_phase[:, :, :U//2, :V-1] - target_phase[:, :, :U//2, :V-1])
        )

        return 0.5 * amplitude_loss + 0.5 * phase_loss

mse_loss = nn.MSELoss()
#spatial_loss = spatial_high_pass_loss
freq_loss=FrequencyHighPassLoss(cutoff=5)
def combined_loss(pred, target, alpha=0.5):
    """
    Computes a weighted sum of MSE loss and frequency domain loss.
    
    Args:
    - pred (torch.Tensor): Predicted high-resolution image.
    - target (torch.Tensor): Ground truth high-resolution image.
    - alpha (float): Weight factor for balancing MSE and Fourier losses.
    
    Returns:
    - torch.Tensor: Combined loss value.
    """
    loss_mse = mse_loss(pred, target)
    loss_freq=freq_loss(pred,target)
    return alpha * loss_mse + (1 - alpha) * loss_freq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    #criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = combined_loss(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        if epoch%100==0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
