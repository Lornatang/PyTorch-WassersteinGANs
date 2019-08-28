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

"""Implementation of multilayer perceptron"""

import torch.nn as nn


class MLP_Generate(nn.Module):

  def __init__(self, opt):
    super(MLP_Generate, self).__init__()

    self.nz = opt.nz
    self.nc = opt.nc
    self.img_size = opt.img_size
    self.ngpu = opt.ngpu

    def block(in_features, out_features, bias=False, bn=True):
      layers = [nn.Linear(in_features, out_features, bias)]
      if bn:
        layers.append(nn.BatchNorm1d(out_features, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))

    self.main = nn.Sequential(
      *block(self.nz, 128, bn=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, self.nc * self.img_size * self.img_size),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.main(x)
    return x.view(x.size(0), self.nc, self.img_size, self.img_size)


class MLP_Discriminator(nn.Module):

  def __init__(self, opt):
    super(MLP_Discriminator, self).__init__()

    self.nc = opt.nc
    self.img_size = opt.img_size
    self.ngpu = opt.ngpu

    self.main = nn.Sequential(
      nn.Linear(self.nc * self.img_size * self.img_size, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 1),
    )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.main(x)
    return x