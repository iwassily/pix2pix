import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import generator, discriminator
import pickle

class pix2pix_trainer(object):
    def __init__(self, tr_loader, val_loader, checkpoint_path, device='cuda', use_ConvTranspose=False):
        super().__init__()
        self.generator = generator(use_ConvTranspose=use_ConvTranspose).to(device)
        self.discriminator = discriminator().to(device)

        self.disc_loss = nn.BCEWithLogitsLoss()
        self.gen_loss = nn.L1Loss()

        self.train_loader = tr_loader
        self.val_loader = val_loader
        self.epochs_trained = 0
        self.gen_losses = []
        self.disc_losses = []
        self.device = 'cuda'
        self.checkpoint_path = checkpoint_path

    def save_checkpoint(self):
        torch.save(self.generator.state_dict(), self.checkpoint_path+'/generator.pt')
        torch.save(self.discriminator.state_dict(), self.checkpoint_path+'/discriminator.pt')

        history = (self.gen_losses, self.disc_losses)
        file_to_store = open(self.checkpoint_path+'/history.pickle', "wb")
        pickle.dump(history, file_to_store)
        file_to_store.close()

    def load_checkpoint(self):
        self.generator.load_state_dict(torch.load(self.checkpoint_path+'/generator.pt', map_location=self.device))
        self.discriminator.load_state_dict(torch.load(self.checkpoint_path+'/discriminator.pt', map_location=self.device))
        file_to_read = open(self.checkpoint_path+'/history.pickle', "rb")
        history = pickle.load(file_to_read)
        self.gen_losses = history[0]
        self.disc_losses = history[1]
        self.epochs_trained = len(self.gen_losses)
        file_to_read.close()

    def plot_losses(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.gen_losses, '-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Generator loss');
        plt.show()

        plt.figure(figsize=(15, 6))
        plt.plot(self.disc_losses, '-', color='g')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Discriminator loss');
        plt.show()
         
    def show_examples(self, num_examples=3, figsize=(18, 18), mode='train'):
        plt.figure(figsize=figsize)
        loader = self.train_loader if (mode=='train') else self.val_loader
        annotations, photos = next(iter(loader))
        output = self.generator(annotations.to(self.device))
        output = output.permute(0, 2, 3, 1).detach().cpu().numpy()
        # detech etc
        for k in range(num_examples):
            plt.subplot(3, num_examples, k+1)
            plt.imshow(annotations[k].permute(1, 2, 0))
            plt.title('Input')
            plt.axis('off')
        
            plt.subplot(3, num_examples, k+num_examples+1)
            plt.imshow(output[k])
            plt.title('Output')
            plt.axis('off')
        
            plt.subplot(3, num_examples, k+1+(num_examples*2))
            plt.imshow(photos[k].permute(1, 2, 0))
            plt.title('Ground Truth')
            plt.axis('off')
        plt.show();

    def fit_one_epoch(self, lr=0.0002, lambd=100):
        total_epoch_loss_gen = 0
        total_epoch_loss_disc = 0
          
        for annotations, photos in self.train_loader:
          self.discriminator.train()
          self.generator.train()

          annotations = annotations.to(self.device)
          photos = photos.to(self.device)
          # train discriminator
          self.gen_opt.zero_grad()
          disc_preds_real = self.discriminator(annotations, photos)
          real_targets = torch.ones(disc_preds_real.shape, device=self.device)
          real_loss = self.disc_loss(disc_preds_real, real_targets)
        
          fake_photos = self.generator(annotations)
          disc_preds_fake = self.discriminator(annotations, fake_photos)
          fake_targets = torch.zeros(disc_preds_fake.shape, device=self.device)
          fake_loss = self.disc_loss(disc_preds_fake, fake_targets)

          loss_d = (real_loss + fake_loss)/2
          loss_d.backward()
          self.disc_opt.step()

          total_epoch_loss_disc += loss_d.item()

          #train generator
          self.gen_opt.zero_grad()
          fake_photos = self.generator(annotations)
          disc_preds_fake = self.discriminator(annotations, fake_photos)
          fake_targets = torch.zeros(disc_preds_fake.shape, device=self.device)

          fake_loss = self.disc_loss(disc_preds_fake, fake_targets)
          l1_loss = self.gen_loss(photos, fake_photos)
        
          loss_g = fake_loss + lambd *l1_loss
          loss_g.backward()
          self.gen_opt.step()

          total_epoch_loss_gen += loss_g.item()

        self.epochs_trained += 1
        return total_epoch_loss_gen/len(self.train_loader), total_epoch_loss_disc/len(self.train_loader)

    def fit(self, epochs, lr=0.0002, lambd=100, save_checkpoints=False):
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(epochs):
          loss_gen, loss_disc = self.fit_one_epoch()
          self.gen_losses.append(loss_gen)
          self.disc_losses.append(loss_disc)
          print("Epoch [{}/{}], loss_g: {:.5f}, loss_d: {:.5f}".format(epoch+1, epochs, loss_gen, loss_disc))
          if (epoch % 5 == 0): self.show_examples(num_examples=4, figsize=(16, 12), mode='train')
          if ((save_checkpoints == True) and (epoch % 50 == 0)):
              self.save_checkpoint()
              print('models saved')

        self.plot_losses()