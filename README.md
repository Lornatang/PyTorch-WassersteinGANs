### PyTorch-WGAN
 
Please refer to https://zhuanlan.zhihu.com/p/25071913

#### Args

```text
Namespace(batch_size=64, beta1=0.5, beta2=0.999, checkpoints_dir='./checkpoints', clip_value=0.01, cuda=True, dataroot='/input/pytorch_datasets/', img_size=32, lr=5e-05, manualSeed=None, n_critic=5, n_epochs=200, nc=3, netD='', netG='', ngpu=1, nz=100, out_images='./imgs', phase='eval', sample_size=1000)
```

#### Model struct

```text
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 256, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): Tanh()
  )
)
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```

#### Usage

```text
git clone https://github.com/Lornatang/PyTorch-WGAN.git
cd PyTorch-WGAN/ 
python3 main.py --dataroot ${Datasets path} --cuda
```

You can use my pre-training model for quick testing.

Model file in `checkpoints` directory.

just run

```text
python3 main.py --cuda --netG checkpoints/netG_epoch_200.pth --netD checkpoints/netD_epoch_200.pth  
```

#### LICENSE

Apache License Version 2.0, January 2004
http://www.apache.org/licenses/
