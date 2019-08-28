# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of convolution network neural"""

import torch.nn as nn


def weights_init(m):
  """ custom weights initialization called on netG and netD
  """
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  """ generate model
  """

  def __init__(self, opt):
    super(Generator, self).__init__()

    self.nz = opt.nz
    self.ngpu = opt.gpus

    self.main = nn.Sequential(
      # inputs is Z, going into a convolution
      nn.ConvTranspose2d(self.nz, 256, 4, 1, 0, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      # state size. 256 x 4 x 4
      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      # state size. 128 x 8 x 8
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # state size. 64 x 16 x 16
      nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. 3 x 32 x 32
    )

  def forward(self, x):
    """ forward layer
    Args:
      x: input tensor data.
    Returns:
      forwarded data.
    """
    if self.ngpu > 1:
      x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      x = self.main(x)
    return x


class Discriminator(nn.Module):
  """ discriminate model
  """

  def __init__(self, opt):
    super(Discriminator, self).__init__()

    self.nc = opt.nc
    self.ngpu = opt.gpus

    self.main = nn.Sequential(
      # inputs is 3 x 32 x 32
      nn.Conv2d(self.nc, 64, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. 64 x 16 x 16
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. 128 x 8 x 8
      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. 256 x 4 x 4
      nn.Conv2d(256, 1, 4, 1, 0, bias=False),
    )

  def forward(self, x):
    """ forward layer
    Args:
      x: input tensor data.
    Returns:
      forwarded data.
    """
    if self.ngpu > 1:
      x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      x = self.main(x)
    return x