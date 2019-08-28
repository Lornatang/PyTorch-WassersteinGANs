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

"""Only used to load the Cifar dataset"""

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_datasets(opt):
  dataset = datasets.CIFAR10(root=opt.dataroot,
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)
  return dataloader
