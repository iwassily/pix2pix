import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class generator_encoder_layer(nn.Module):
  def __init__(self, input_channels, output_channels, pos='middle'):
    super().__init__()
    if (pos == 'first'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                       nn.LeakyReLU(negative_slope=0.2))
    elif (pos == 'bottleneck'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                       nn.LeakyReLU(negative_slope=0.2))

    else:
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                       nn.BatchNorm2d(output_channels),
                                       nn.LeakyReLU(negative_slope=0.2))

  def forward(self, x):
    return self.layer(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.ups = nn.Upsample(scale_factor = 4, mode='bilinear')
        self.refl = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, inp):
        inp = self.ups(inp)
        inp = self.refl(inp)
        return self.conv(inp)


class generator_decoder_layer(nn.Module):
  def __init__(self, input_channels, output_channels, pos='middle', use_ConvTranspose=False):
    super().__init__()
    if (use_ConvTranspose == False):
          if (pos == 'first3'):
              self.layer = nn.Sequential(Upsample(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.ReLU(),
                                     nn.Dropout(0.5)
                                     )
          elif (pos == 'middle'):
              self.layer = nn.Sequential(Upsample(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.ReLU(),
                                     )
          elif (pos == 'final'):
              self.layer = nn.Sequential(Upsample(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.Tanh(),
                                     )


    else:
          if (pos == 'first3'):
              self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.ReLU(),
                                     nn.Dropout(0.5)
                                     )
          elif (pos == 'middle'):
              self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.ReLU(),
                                     )
          elif (pos == 'final'):
              self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1),
                                     nn.BatchNorm2d(output_channels),
                                     nn.Tanh(),
                                     )

  def forward(self, x):
    return self.layer(x)

class generator(nn.Module):
  def __init__(self, use_ConvTranspose):
    super().__init__()
    # input B * 3 * 256 * 256, output B * 3 * 256 * 256
    self.enc1 = generator_encoder_layer(input_channels=3, output_channels=64, pos='first')                   #input B * 3 * 256 * 256, output B * 64 * 128 * 128
    self.enc2 = generator_encoder_layer(input_channels=64, output_channels=128)                              #input B * 64 * 128 * 128, output B * 128 * 64 * 64
    self.enc3 = generator_encoder_layer(input_channels=128, output_channels=256)                             #input B * 128 * 64 * 64, output B * 256 * 32 * 32
    self.enc4 = generator_encoder_layer(input_channels=256, output_channels=512)                             #input B * 256 * 32 * 32, output B * 512 * 16 * 16
    self.enc5 = generator_encoder_layer(input_channels=512, output_channels=512)                             #input B * 512 * 16 * 16, output B * 512 * 8 * 8
    self.enc6 = generator_encoder_layer(input_channels=512, output_channels=512)                             #input B * 512 * 8 * 8, output B * 512 * 4 * 4
    self.enc7 = generator_encoder_layer(input_channels=512, output_channels=512)                             #input B * 512 * 4 * 4, output B * 512 * 2 * 2
    self.enc8 = generator_encoder_layer(input_channels=512, output_channels=512, pos='bottleneck')           #input B * 512 * 2 * 2, output B * 512 * 1 * 1

    self.dec1 = generator_decoder_layer(input_channels=512, output_channels=512, pos='first3', use_ConvTranspose=use_ConvTranspose)               #input B * 512 * 1 * 1, output B * 512 * 2 * 2
    self.dec2 = generator_decoder_layer(input_channels=1024, output_channels=512, pos='first3', use_ConvTranspose=use_ConvTranspose)              #input B * 1024 * 2 * 2, output B * 512 * 4 * 4
    self.dec3 = generator_decoder_layer(input_channels=1024, output_channels=512, pos='first3', use_ConvTranspose=use_ConvTranspose)              #input B * 1024 * 4 * 4, output B * 512 * 8 * 8
    self.dec4= generator_decoder_layer(input_channels=1024, output_channels=512, use_ConvTranspose=use_ConvTranspose)                             #input B * 1024 * 8 * 8, output B * 512 * 16 * 16
    self.dec5 = generator_decoder_layer(input_channels=1024, output_channels=256, use_ConvTranspose=use_ConvTranspose)                            #input B * 1024 * 16 * 16, output B * 512 * 32 * 32
    self.dec6 = generator_decoder_layer(input_channels=512, output_channels=128, use_ConvTranspose=use_ConvTranspose)                             #input B * 512 * 32 * 32, output B * 256 * 64* 64
    self.dec7 = generator_decoder_layer(input_channels=256, output_channels=64, use_ConvTranspose=use_ConvTranspose)                              #input B * 256 * 64 * 64, output B * 64 * 128 * 128
    self.dec8 = generator_decoder_layer(input_channels=128, output_channels=3, pos='final', use_ConvTranspose=use_ConvTranspose)                  #input B * 128 * 128 * 128, output B * 3 * 256 * 256

  

  def forward (self, x):
    e1 = self.enc1(x)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)
    e5 = self.enc5(e4)
    e6 = self.enc6(e5)
    e7 = self.enc7(e6)
    e8 = self.enc8(e7)

    d1 = self.dec1(e8)
    d2 = self.dec2(torch.cat([d1, e7], 1))
    d3 = self.dec3(torch.cat([d2, e6], 1))
    d4 = self.dec4(torch.cat([d3, e5], 1))
    d5 = self.dec5(torch.cat([d4, e4], 1))
    d6 = self.dec6(torch.cat([d5, e3], 1))
    d7 = self.dec7(torch.cat([d6, e2], 1))
    d8 = self.dec8(torch.cat([d7, e1], 1))
    return d8

class discriminator_encoder_layer(nn.Module):
  def __init__(self, input_channels, output_channels, pos='middle'):
    super().__init__()
    if (pos == 'first'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1, bias=True),
                                       nn.LeakyReLU(negative_slope=0.2))
    elif (pos == 'middle'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(output_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
    elif (pos == 'before_last'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(output_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
    elif (pos == 'final'):
          self.layer = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(4,4), stride=1, padding=1, bias=False))
          # no nn.Sigmoid() here, but nn.BCEWithLogitsLoss() for disc.

  def forward(self, x):
    return self.layer(x)

class discriminator(nn.Module):
  # input is annotation + photo(real or fake)
  # input B * 6 * 256 * 256, output B * 1 * 30 * 30
  def __init__(self):
    super().__init__()
    self.enc1 = discriminator_encoder_layer(input_channels=6, output_channels=64, pos='first')
    self.enc2 = discriminator_encoder_layer(input_channels=64, output_channels=128)
    self.enc3 = discriminator_encoder_layer(input_channels=128, output_channels=256)
    self.enc4 = discriminator_encoder_layer(input_channels=256, output_channels=512, pos='before_last')
    self.enc5 = discriminator_encoder_layer(input_channels=512, output_channels=1, pos='final')

  def forward(self, x, y): # annotations + photos
    x = self.enc1(torch.cat([x,y],1))
    x = self.enc2(x)
    x = self.enc3(x)
    x = self.enc4(x)
    x = self.enc5(x)

    return x



        