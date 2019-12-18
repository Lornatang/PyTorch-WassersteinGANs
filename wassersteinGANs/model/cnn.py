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

import torch
import torch.nn as nn


class Generator(nn.Module):
  """ generate model
  """

  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.channels = 1
    self.ngpu = ngpu

    def block(in_channels, out_channels, normalize=True):
      """ simple layer struct.
      Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        normalize (bool, optional): If ``True``, accelerating Deep Network Training
          by Reducing Internal Covariate Shift`. Default: ``True``

      Returns:
        A list layer.
      """
      layers = [nn.Linear(in_channels, out_channels)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_channels, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.main = nn.Sequential(
      *block(100, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, self.channels * 28 * 28),
      nn.Tanh()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if torch.cuda.is_available() and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs.view(outputs.size(0), *(self.channels, 28, 28))


class Discriminator(nn.Module):
  """ discriminate model
  """

  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.channels = 1
    self.ngpu = ngpu

    self.main = nn.Sequential(
      nn.Linear(self.channels * 28 * 28, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, self.channels),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    inputs = inputs.view(inputs.size(0), -1)
    if torch.cuda.is_available() and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs
