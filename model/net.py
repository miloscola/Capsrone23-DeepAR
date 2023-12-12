'''Defines the neural network, loss function and metrics'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

logger = logging.getLogger('DeepAR.Net')

class Net(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)

        self.lstm = nn.LSTM(input_size=1+params.cov_dim+params.embedding_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, train_batch, idx, hidden, cell, labels_batch, calc_loss=True):
        # x is now a training batch instead of a single timestep
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        
        mu = 0 #needed to fix "mu not defined" error in if statement
        
        #initialize loss
        loss = torch.zeros(1, device=self.params.device)
        
        # Embedding for the time series id
        onehot_embed = self.embedding(idx) 
        
        #iterate over timesteps in training window (min is to fix for loop when used in test function)
        for t in range(min(self.params.train_window, train_batch.shape[0])):
            
            if t == 0 or t == 10:
                #print("\nlabels batch ", labels_batch.shape)
                print("\ntrain_batch ", train_batch.shape)
                print("\nLSTM input ", train_batch[t, :, :].unsqueeze(0).shape)
                print("\ntrain window ", self.params.train_window)
            
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            
            if t > 0 and torch.sum(zero_index) > 0:
                # Replace missing values with the output mu from the last time step
                train_batch[t, zero_index, 0] = mu[zero_index][:, 0]
            
            if t == 0 or t == 10:
                print("\nonehot_embed ", onehot_embed.shape)
                
            
            #Concatenate x (z_{t-1} + x_t) with the one-hot embedding
            lstm_input = torch.cat((train_batch[t, :, :].unsqueeze(0), onehot_embed), dim=2)
           
            # Pass lstm_input input through the LSTM layer
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
           
            # use h from all three layers to calculate mu and sigma
            hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
           
            # Predict the pre-sigma values from the LSTM hidden states
            pre_sigma = self.distribution_presigma(hidden_permute)
          
            # Predict the mean (mu) of the distribution from the LSTM hidden states
            mu = self.distribution_mu(hidden_permute)
           
            # Predict the standard deviation (sigma) with a softplus activation
            sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive 
           
            # Compute the loss for the current time step and accumulate it
            if calc_loss:
                loss += loss_fn(mu, sigma, labels_batch[t]) #how do I deal with the loss?
            
            if t == 0 or t == 10:
                print("\nmu ", mu.shape)
                print("\nsigma ", sigma.shape)
                print("\n_________________________")
        
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell, loss

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, x, v_batch, id_batch, hidden, cell, input_mu, input_sigma, sampling=False):
        # Get the batch size from the input tensor x
        batch_size = x.shape[1]
    
#########################################################
#progress 12/3: fixed loss, created for loop inside test, 
    
        mu = 0 #needed to fix "mu not defined" error in if statement
        
        # Embedding for the time series id
        onehot_embed = self.embedding(id_batch) 
        
        #iterate over timesteps in training window
        for t in range(self.params.test_predict_start):    
            
            if t == 0 or t == 10:
                print("\nx ", x.shape)
                print("\nLSTM input ", x[t, :, :].unsqueeze(0).shape)
                print("\nx window ", self.params.train_window)
            
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (x[t, :, 0] == 0)
            
            if t > 0 and torch.sum(zero_index) > 0:
                # Replace missing values with the output mu from the last time step
                x[t, zero_index, 0] = mu[zero_index][:, 0]
            
            if t == 0 or t == 10:
                print("\nonehot_embed ", onehot_embed.shape)
                
            
            #Concatenate x (z_{t-1} + x_t) with the one-hot embedding
            lstm_input = torch.cat((x[t, :, :].unsqueeze(0), onehot_embed), dim=2)
           
            # Pass lstm_input input through the LSTM layer
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
           
            # use h from all three layers to calculate mu and sigma
            hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
           
            # Predict the pre-sigma values from the LSTM hidden states
            pre_sigma = self.distribution_presigma(hidden_permute)
          
            # Predict the mean (mu) of the distribution from the LSTM hidden states
            mu = self.distribution_mu(hidden_permute)
           
            # Predict the standard deviation (sigma) with a softplus activation
            sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive 
            
            if t == 0 or t == 10:
                print("\nmu ", mu.shape)
                print("\nsigma ", sigma.shape)
                print("\ninput_mu ", input_mu.shape)
                print("\ninput_sigma ", input_sigma.shape)
                print("\nv_batch[:, 0] * mu + v_batch[:, 1] ", (v_batch[:, 0] * mu + v_batch[:, 1]).shape)
                print("\nv_batch[:, 0] * sigma ", (v_batch[:, 0] * sigma).shape)
                print("\n", (v_batch[:, 0]).shape)
                print((v_batch[:, 1]).shape)
                print("\n_________________________")
                
            
            #update input mu and input sigma
            input_mu[:, t] = v_batch[:, 0] * mu[:, 0] + v_batch[:, 1] #collapsed mu
            input_sigma[:, t] = v_batch[:, 0] * sigma[:, 0] #collapsed sigma
    
########################################################################
    
        # If sampling is set to True
        if sampling:
            # Initialize a tensor to store samples with shape (sample_times, batch_size, predict_steps)
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                   device=self.params.device)
            
            # Iterate over sample times
            for j in range(self.params.sample_times):
                # Initialize decoder hidden and cell states
                decoder_hidden = hidden
                decoder_cell = cell
                
                # Iterate over prediction steps
                for t in range(self.params.predict_steps):
                    # Call the model to get mean, standard deviation, and updated hidden and cell states
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                         id_batch, decoder_hidden, decoder_cell, 0, calc_loss=False)
                    
                    # Create a normal distribution with mean and standard deviation
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    
                    # Sample from the normal distribution (not scaled)
                    pred = gaussian.sample()
                    
                    # Scale the sample and store it in the samples tensor
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    
                    # Update the input tensor x for the next time step
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred
    
            # Compute the median of the samples along the first dimension
            sample_mu = torch.median(samples, dim=0)[0]
            
            # Compute the standard deviation of the samples along the first dimension
            sample_sigma = samples.std(dim=0)
            
            # Return the samples, sample mean, and sample standard deviation
            return samples, sample_mu, sample_sigma, input_mu, input_sigma
    
        # If sampling is set to False
        else:
            # Initialize decoder hidden and cell states
            decoder_hidden = hidden
            decoder_cell = cell
            
            # Initialize tensors to store sample mean and sample standard deviation
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            
            # Iterate over prediction steps
            for t in range(self.params.predict_steps):
                # Call the model to get mean, standard deviation, and updated hidden and cell states
                mu_de, sigma_de, decoder_hidden, decoder_cell, _ = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     id_batch, decoder_hidden, decoder_cell, 0, calc_loss=False)
                
                # Scale the mean and standard deviation and store them in the corresponding tensors
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                
                # Update the input tensor x for the next time step
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
    
            # Return the sample mean and sample standard deviation
            return sample_mu, sample_sigma, input_mu, input_sigma



def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
