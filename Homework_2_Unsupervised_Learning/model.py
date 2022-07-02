### Packages Import
import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import os

# Set seeds for reproducibility (PyTorch, Python, Numpy)
matricola = 2013031
torch.manual_seed(matricola)
random.seed(matricola)
np.random.seed(matricola)

### The class for the AUTOENCODER contains architecture, training, testing and ploting functions
class Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder = nn.Sequential(  
            nn.Conv2d(in_channels = 1,    # First convolutional layer
                      out_channels = 8,
                      kernel_size=3, 
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8,      # Second convolutional layer
                      out_channels=16,
                      kernel_size=3, 
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,     # Third convolutional layer
                      out_channels=32,
                      kernel_size=3, 
                      stride=2,
                      padding=0),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),            # Flatten layer
            nn.Linear(in_features= (32 * 3* 3),
                      out_features=64),                               # First linear layer
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=encoded_space_dim) # Second linear layer
        )
        

        ### Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=64), # First linear layer
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=(32*3*3)),          # Second linear layer
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),           #Unflaten
            nn.ConvTranspose2d(in_channels=32,                         # First transposed convolution
                               out_channels=16,
                               kernel_size=3, 
                               stride=2,
                               padding =0,
                               output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16,                         # Second transposed convolution
                               out_channels=8,
                               kernel_size=3, 
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8,                          # Third transposed convolution
                               out_channels=1,
                               kernel_size=3, 
                               stride=2,
                               padding=1,
                               output_padding=1)
        )
        
    def forward(self, x,mode):
        if (mode == "Train"):
            self.encoder.train()
            self.decoder.train()
        elif (mode == "Test"):
            self.encoder.eval()
            self.decoder.eval()
            
        # Apply encoder decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    ### Training function
    def train_epoch(self,device,dataloader, loss_fn, optimizer):
        """
        This function train the network for one epoch
        """
        # Train
        train_loss = []
        for sample_batched, _ in dataloader:
            # Move data to device
            sample_batched = sample_batched.to(device)
            # Encode Decode the data
            encoded_decoded_sample = self.forward(sample_batched,"Train")
            # Compute loss
            loss = loss_fn(encoded_decoded_sample, sample_batched)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            #Updata weights
            optimizer.step()
            #Save trai loss for this batch
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)
        #Save the average train loss
        train_loss = np.mean(train_loss)
        print(f"AVERAGE TRAIN LOSS: {train_loss}")

        return train_loss

    ### Testing function
    def test_epoch(self,device, dataloader, loss_fn):
        """
        This function test the network performance for one epoch of training
        """
        test_loss = []
        # Discable gradient tracking
        with torch.no_grad():
            for sample_batched, _ in dataloader:
                # Move data to device
                sample_batched = sample_batched.to(device)
                # Encode Decode the data
                encoded_decoded_sample = self.forward(sample_batched,"Test")
                # Compute loss
                loss = loss_fn(encoded_decoded_sample, sample_batched)
                 #Save test loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                test_loss.append(loss_batch)
            #Save the average train loss
            test_loss = np.mean(test_loss)
            print(f"AVERAGE TEST LOSS: {test_loss}")

        return test_loss
    
        
        
    def training_cycle(self, device, training_data, test_data, loss_fn, optim, num_epochs, test_dataset,encoded_space_dim, plot = False,
                       keep_plots = False, keep_model=False):
        """
        This function train the network for a desired number of epochs it also test the network 
        reconstruction performance and make plots comparing the input image and the reconstructed one every 5 epochs.
        """
        #I keep track of losses for plots
        train_loss = []
        test_loss  = []
        i = 0
        for epoch in range(num_epochs):
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            ### Training (use the training function)
            tr_l = self.train_epoch(
                device=device, 
                dataloader=training_data, 
                loss_fn=loss_fn, 
                optimizer=optim)
            train_loss.append(tr_l)
            ### Validation  (use the testing function)
            t_l = self.test_epoch(
                device=device, 
                dataloader=test_data, 
                loss_fn=loss_fn)
            test_loss.append(t_l)

            ### Plot progress
            if plot:
                if (i % 5 == 0): self.plot_progress(test_dataset,epoch,device,encoded_space_dim,keep_plots = keep_plots)
            i +=1 
            ### Save network parameters
            if keep_model:
                os.makedirs('./Models', exist_ok=True)
                torch.save(self.encoder.state_dict(), './Models/encoder_params.pth')
                torch.save(self.decoder.state_dict(), './Models/decoder_params.pth')
        return train_loss, test_loss
    
    
    
    def plot_progress(self,test_dataset,epoch,device,encoded_space_dim,keep_plots = False):
        """
        This function plot the image we send to the autoencoder and the one returned by the
        network.
        """
        categories = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        elements = [0,1,5,37]

        fig, axs = plt.subplots(2, 4, figsize=(12,6))
        fig.suptitle('Original images and reconstructed image (EPOCH %d)' % (epoch + 1),fontsize=15)
        fig.subplots_adjust(top=0.88)
        axs = axs.ravel()
        for i in range (4):
            img, label = test_dataset[elements[i]][0].unsqueeze(0).to(device),test_dataset[elements[i]][1]
            with torch.no_grad():
                rec_img  = self.forward(img,"Test")
            # Plot the reconstructed image  
            axs[i].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[i].set_title(categories[label])
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i+4].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[i+4].set_title('Reconstructed image')
            axs[i+4].set_xticks([])
            axs[i+4].set_yticks([])
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        # Save figures
        if keep_plots:
            os.makedirs('./Img/Training_plots/autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
            fig.savefig('./Img/Training_plots/autoencoder_progress_%d_features/epoch_%d.svg' % (encoded_space_dim, epoch + 1), format='svg')
        plt.show()
        plt.close()
    
#     ### Training function
#     def train_epoch(self,encoder, decoder, device,dataloader, loss_fn, optimizer):
#         """
#         This function train the network for one epoch
#         """
#         # Set train mode for both networks
#         encoder.train()
#         decoder.train()

#         # Train
#         train_loss = []
#         for sample_batched, _ in dataloader:
#             # Move data to device
#             sample_batched = sample_batched.to(device)
#             # Encode the data
#             encoded_sample = encoder(sample_batched)
#             # Decode the data
#             decoded_sample = decoder(encoded_sample)
#             # Compute loss
#             loss = loss_fn(decoded_sample, sample_batched)
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             #Updata weights
#             optimizer.step()
#             #Save trai loss for this batch
#             loss_batch = loss.detach().cpu().numpy()
#             train_loss.append(loss_batch)
#         #Save the average train loss
#         train_loss = np.mean(train_loss)
#         print(f"AVERAGE TRAIN LOSS: {train_loss}")

#         return train_loss

#     ### Testing function
#     def test_epoch(self,encoder, decoder, device, dataloader, loss_fn):
#         """
#         This function test the network performance for one epoch of training
#         """
#         # Set evaluation mode for both networks
#         encoder.eval()
#         decoder.eval()
#         #
#         test_loss = []
#         # Discable gradient tracking
#         with torch.no_grad():
#             for sample_batched, _ in dataloader:
#                 # Move data to device
#                 sample_batched = sample_batched.to(device)
#                 # Encode the data
#                 encoded_sample = encoder(sample_batched)
#                 # Decode the data
#                 decoded_sample = decoder(encoded_sample)
#                 # Compute loss
#                 loss = loss_fn(decoded_sample, sample_batched)
#                  #Save test loss for this batch
#                 loss_batch = loss.detach().cpu().numpy()
#                 test_loss.append(loss_batch)
#             #Save the average train loss
#             test_loss = np.mean(test_loss)
#             print(f"AVERAGE TEST LOSS: {test_loss}")

#         return test_loss
    
    
#     def plot_progress(self,encoder, decoder,test_dataset,epoch,device,keep =False):
#         """
#         This function plot the image we send to the autoencoder and the one returned by the
#         network.
#         """
#         categories = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
#         elements = [0,1,5,37]

#         encoder.eval()
#         decoder.eval()
#         fig, axs = plt.subplots(2, 4, figsize=(12,6))
#         fig.suptitle('Original images and reconstructed image (EPOCH %d)' % (epoch + 1),fontsize=15)
#         fig.subplots_adjust(top=0.88)
#         axs = axs.ravel()
#         for i in range (4):
#             img, label = test_dataset[elements[i]][0].unsqueeze(0).to(device),test_dataset[elements[i]][1]
#             with torch.no_grad():
#                 rec_img  = decoder(encoder(img))
#             # Plot the reconstructed image  
#             axs[i].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
#             axs[i].set_title(categories[label])
#             axs[i].set_xticks([])
#             axs[i].set_yticks([])
#             axs[i+4].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
#             axs[i+4].set_title('Reconstructed image')
#             axs[i+4].set_xticks([])
#             axs[i+4].set_yticks([])
#         plt.tight_layout()
#         fig.subplots_adjust(top=0.88)
#         # Save figures
#         if keep:
#             os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
#             fig.savefig('autoencoder_progress_%d_features/epoch_%d.jpg' % (encoded_space_dim, epoch + 1))
#         plt.show()
#         plt.close()
        
        
#     def training_cycle(self,encoder,decoder,device,training_data, test_data, loss_fn, optim,num_epochs,test_dataset,plot = False):
#         """
#         This function train the network for a desired number of epochs it also test the network 
#         reconstruction performance and make plots comparing the input image and the reconstructed one every 5 epochs.
#         """
#         #I keep track of losses for plots
#         train_loss = []
#         test_loss  = []
#         i = 0
#         for epoch in range(num_epochs):
#             print('EPOCH %d/%d' % (epoch + 1, num_epochs))
#             ### Training (use the training function)
#             tr_l = self.train_epoch(
#                 encoder=encoder, 
#                 decoder=decoder, 
#                 device=device, 
#                 dataloader=training_data, 
#                 loss_fn=loss_fn, 
#                 optimizer=optim)
#             train_loss.append(tr_l)
#             ### Validation  (use the testing function)
#             t_l = self.test_epoch(
#                 encoder=encoder, 
#                 decoder=decoder, 
#                 device=device, 
#                 dataloader=test_data, 
#                 loss_fn=loss_fn)
#             test_loss.append(t_l)
#             # Print Validationloss
#             #print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, t_l))

#             ### Plot progress
#             if plot:
#                 if (i % 5 == 0): self.plot_progress(encoder, decoder,test_dataset,epoch,device,keep = False)
#             #Save network parameters
#             i +=1 
#             torch.save(encoder.state_dict(), 'encoder_params.pth')
#             torch.save(decoder.state_dict(), 'decoder_params.pth')
#         return train_loss, test_loss
    
    
    
    
    
    
    
    
    
    
class Fine_Tuned_Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim, pre_trained_AE):
        super().__init__()
        
        self.encoder = pre_trained_AE.encoder
        self.new_layer   = nn.Linear(encoded_space_dim,10) 
        
    def forward(self,  x):
        x = self.encoder(x)
        x = self.new_layer(x)
        return x       
        
        
    ### Training function
    def tune_train_epoch(self,encoder, new_layer, device,dataloader, loss_fn, optimizer):
        """
        This function train the network for one epoch
        """
        # Set train mode for both networks
        encoder.train()
        new_layer.train()

        # Train
        train_loss = []
        for x_batched, y_batched in dataloader:
            # Move data to device
            x_batched = x_batched.to(device)
            y_batched = y_batched.to(device)       
            # Encode the data
            z = encoder(x_batched)
            out = new_layer(z)
            # Compute loss
            loss = loss_fn(out, y_batched)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            #Updata weights
            optimizer.step()
            #Save trai loss for this batch
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)
        #Save the average train loss
        train_loss = np.mean(train_loss)
        print(f"AVERAGE TRAIN LOSS: {train_loss}")

        return train_loss

    ### Testing function
    def tune_val_epoch(self,encoder, new_layer, device, dataloader, loss_fn):
        """
        This function test the network performance for one epoch of training
        """
        # Set evaluation mode for both networks
        encoder.eval()
        new_layer.eval()
        #
        val_loss = []
        # Discable gradient tracking
        with torch.no_grad():
            for x_batched, y_batched in dataloader:
                # Move data to device
                x_batched = x_batched.to(device)
                y_batched = y_batched.to(device)
                # Encode the data
                z   = encoder(x_batched)
                out = new_layer(z)
                # Compute loss
                loss = loss_fn(out, y_batched)
                 #Save test loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                val_loss.append(loss_batch)
            #Save the average train loss
            val_loss = np.mean(val_loss)
            print(f"AVERAGE VALIDATION LOSS: {val_loss}")

        return val_loss       
        
    def tune_training_cycle(self, encoder,new_layer,device,training_data, val_data, loss_fn, optim,num_epochs,test_dataset):
        """
        This function train the network for a desired number of epochs it also test the network 
        reconstruction performance and make plots comparing the input image and the reconstructed one.
        """
        #I keep track of losses for plots
        train_loss = []
        val_loss  = []
        for epoch in range(num_epochs):
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            ### Training (use the training function)
            tr_l = self.tune_train_epoch(
                encoder=encoder, 
                new_layer=new_layer, 
                device=device, 
                dataloader=training_data, 
                loss_fn=loss_fn, 
                optimizer=optim)
            train_loss.append(tr_l)
            ### Validation  (use the testing function)
            v_l = self.tune_val_epoch(
                encoder=encoder, 
                new_layer=new_layer,  
                device=device, 
                dataloader=val_data, 
                loss_fn=loss_fn)
            val_loss.append(v_l)
            # Print Validationloss
            #print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, t_l))

        return train_loss, val_loss        
        
        
        
        
        
        
### The class for the AUTOENCODER contains architecture, training, testing and ploting functions
class Variational_Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder = nn.Sequential(  
            nn.Conv2d(in_channels = 1,    # First convolutional layer
                      out_channels = 8,
                      kernel_size=3, 
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8,      # Second convolutional layer
                      out_channels=16,
                      kernel_size=3, 
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,     # Third convolutional layer
                      out_channels=32,
                      kernel_size=3, 
                      stride=2,
                      padding=0),
            nn.ReLU(True),
            nn.Flatten(start_dim=1)            # Flatten layer
                                  )
        
        ### Now we implement the variational part
        self.fc_mn = nn.Sequential(nn.Linear(in_features= (32 * 3* 3),out_features=64),                             
                                   nn.ReLU(True),          
                                   nn.Linear(in_features=64, out_features=encoded_space_dim)
                                  )
        
        self.fc_std = nn.Sequential(nn.Linear(in_features= (32 * 3* 3),out_features=64),                             
                                   nn.ReLU(True),          
                                   nn.Linear(in_features=64, out_features=encoded_space_dim)
                                  )
        

        ### Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=64), # First linear layer
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=(32*3*3)),          # Second linear layer
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),           #Unflaten
            nn.ConvTranspose2d(in_channels=32,                         # First transposed convolution
                               out_channels=16,
                               kernel_size=3, 
                               stride=2,
                               padding =0,
                               output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16,                         # Second transposed convolution
                               out_channels=8,
                               kernel_size=3, 
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8,                          # Third transposed convolution
                               out_channels=1,
                               kernel_size=3, 
                               stride=2,
                               padding=1,
                               output_padding=1)
                                )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x) 
        # Predict distribution mean and standard deviation
        mn  = self.fc_mn(z)
        std = self.fc_std(z)
        #sample distribution based on mean and standard deviation
        sample = mn + torch.exp(std/2)*torch.rand_like(mn)
        #decode
        #out = self.decoder(sample)
        return out, mn, std
    
    def loss_VAE(self,prediction, real,mu,sigma):
        loss = F.mse_loss(prediction, real, reduction='sum') 
        kl_div    = -0.5 * torch.sum(1. + sigma - mu**2 - torch.exp(sigma))
        return loss, kl_div
    
    def sampler(self,mu,sigma):
        return mu + torch.exp(sigma/2)*torch.rand_like(mu)
        
    
    ### Training function
    def train_epoch(self,encoder, decoder,fc_mn,fc_std, device,dataloader, optimizer):
        """
        This function train the network for one epoch
        """
        # Set train mode for both networks
        encoder.train()
        decoder.train()
        fc_mn.train()
        fc_std.train()
        
        # Train
        train_loss = []
        for sample_batched, _ in dataloader:
            # Move data to device
            sample_batched = sample_batched.to(device)
            # Encode the data
            encoded_sample = encoder(sample_batched)
            
            # Predict distribution mean and standard deviation
            mn  = fc_mn(encoded_sample)
            std = fc_std(encoded_sample)
        
            #sample distribution based on mean and standard deviation
            sample = self.sampler(mn,std)
            
            # Decode the data
            decoded_sample = decoder(sample)
            
            # Compute loss
            loss, kl_div = self.loss_VAE(decoded_sample, sample_batched,mn,std)

            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            #Updata weights
            optimizer.step()
            #Save trai loss for this batch
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)
        #Save the average train loss
        train_loss = np.mean(train_loss)
        print(f"AVERAGE TRAIN LOSS: {train_loss}")

        return train_loss

    ### Testing function
    def test_epoch(self,encoder, decoder,fc_mn,fc_std, device, dataloader):
        """
        This function test the network performance for one epoch of training
        """
        # Set evaluation mode for both networks
        encoder.eval()
        decoder.eval()
        fc_mn.train()
        fc_std.train()
        #
        test_loss = []
        # Discable gradient tracking
        with torch.no_grad():
            for sample_batched, _ in dataloader:
                # Move data to device
                sample_batched = sample_batched.to(device)
                # Encode the data
                encoded_sample = encoder(sample_batched)

                # Predict distribution mean and standard deviation
                mn  = fc_mn(encoded_sample)
                std = fc_std(encoded_sample)

                #sample distribution based on mean and standard deviation
                sample = self.sampler(mn,std)

                # Decode the data
                decoded_sample = decoder(sample)

                # Compute loss
                loss, kl_div = self.loss_VAE(decoded_sample, sample_batched,mn,std)
                #Save test loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                test_loss.append(loss_batch)
            #Save the average train loss
            test_loss = np.mean(test_loss)
            print(f"AVERAGE TEST LOSS: {test_loss}")

        return test_loss
    
    
    
    def training_cycle(self,encoder,decoder,fc_mn,fc_std,device,training_data, test_data, optim,num_epochs,test_dataset,plot = False):
        """
        This function train the network for a desired number of epochs it also test the network 
        reconstruction performance and make plots comparing the input image and the reconstructed one every 5 epochs.
        """
        #I keep track of losses for plots
        train_loss = []
        test_loss  = []
        i = 0
        for epoch in range(num_epochs):
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            ### Training (use the training function)
            tr_l = self.train_epoch(
                encoder=encoder, 
                decoder=decoder,
                fc_mn= fc_mn,
                fc_std=fc_std,
                device=device, 
                dataloader=training_data, 
                optimizer=optim)
            train_loss.append(tr_l)
            ### Validation  (use the testing function)
            t_l = self.test_epoch(
                encoder=encoder, 
                decoder=decoder,
                fc_mn= fc_mn,
                fc_std=fc_std,
                device=device, 
                dataloader=test_data)
            test_loss.append(t_l)
            # Print Validationloss
            #print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, t_l))

            ### Plot progress
            #if plot:
            #    if (i % 5 == 0): self.plot_progress(encoder, decoder,test_dataset,epoch,device,keep = False)
            #Save network parameters
            i +=1 
            torch.save(encoder.state_dict(), 'encoder_params.pth')
            torch.save(decoder.state_dict(), 'decoder_params.pth')
        return train_loss, test_loss
        