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

ngf = 64
ndf = 64


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self, ngpu=1):
    super(Generator, self).__init__()
    self.channels = 3
    self.ngpu = ngpu

    def block(in_channels, out_channels, kernel_size, stride, padding,
              normalize=True):
      """ simple layer struct.
      Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution.
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input.
        normalize (bool, optional): If ``True``, accelerating Deep Network Training
          by Reducing Internal Covariate Shift`. Default: ``True``

      Examples:
        >>> print(*block(100, ngf * 8, 4, 1, 0))

      Returns:
        A list layer.
      """
      layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                   padding, bias=False)]
      if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.ReLU(inplace=True))
      return layers

    self.main = nn.Sequential(
      *block(100, ngf * 8, 4, 1, 0),
      *block(ngf * 8, ngf * 4, 4, 2, 1),
      *block(ngf * 4, ngf * 2, 4, 2, 1),
      *block(ngf * 2, ngf, 4, 2, 1),
      nn.ConvTranspose2d(ngf, self.channels, 4, 2, 1),
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
    return outputs


class Discriminator(nn.Module):
  def __init__(self, ngpu=1):
    super(Discriminator, self).__init__()
    self.channels = 3
    self.ngpu = ngpu

    def block(in_channels, out_channels, kernel_size, stride, padding,
              normalize=True):
      """ simple layer struct.
      Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution.
        padding (int or tuple, optional): Zero-padding added to both sides of the input.
        normalize (bool, optional): If ``True``, accelerating Deep Network Training
          by Reducing Internal Covariate Shift`. Default: ``True``

      Examples:
        >>> print(*block(self.channels, ndf, 4, 2, 1))
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups,
                      bias, padding_mode)]
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

      Returns:
        A list layer.
      """
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, bias=False)]
      if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    # create network sequential.
    self.main = nn.Sequential(
      *block(self.channels, ndf, 4, 2, 1, normalize=False),
      *block(ndf, ndf * 2, 4, 2, 1),
      *block(ndf * 2, ndf * 4, 4, 2, 1),
      *block(ndf * 4, ndf * 8, 4, 2, 1),
      nn.Conv2d(ndf * 8, 1, 4, 1, 0),
      nn.Sigmoid()
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
    return outputs.view(-1, 1).squeeze(1)
