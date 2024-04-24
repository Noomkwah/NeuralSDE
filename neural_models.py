#!/usr/bin/env python
# coding: utf-8


############################ IMPORTS ###################################
from __future__ import annotations
import time
from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple
import warnings
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########################################################################


########################## BASIC CLASSES ###############################
class BrownianIncrements(nn.Module):
    def __init__(self, device: str):
        """
        Initialize BrownianIncrements module.
        
        --------------
        Arguments:
            device (str): Device to perform computations on, e.g., 'cpu' or 'cuda'.
        """
        super(BrownianIncrements, self).__init__()
        
        self.device = device

    def forward(self, t0: float, t1: float, d: int, N_trn: int, corr_matrix: torch.Tensor = None, seed: int = None) -> torch.Tensor:
        """
        Generate N_trn independent increments of a Brownian motion of dimension d between times t0 and t1.
        
        --------------
        Arguments:
            t0 (float): Initial time.
            t1 (float): Final time.
            d (int): Dimension of the Brownian motion.
            N_trn (int): Number of increments to generate.
            seed (int): Seed for reproducibility.
        
        --------------
        Returns:
            increments (tensor): Brownian increments of size (N_trn, d).
        """
        
        if seed is not None:
            torch.manual_seed(seed)
            
        delta_t = t1 - t0
        increments = torch.normal(mean=0, std=torch.sqrt(delta_t).clone().detach(), size=(N_trn, d), device=self.device)
        
        if corr_matrix is not None:
            increments = increments@corr_matrix.T
            
        return increments


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, hidden_activation='relu', output_activation='id'):
        """
        Initializes a Feed Forward Neural Network.
        
        --------------
        Arguments:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): Dimensions of the hidden layers.
            output_dim (int): Dimension of the output.
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_layers = len(self.hidden_dims)
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        self.input_layer = self.create_layer(input_dim, hidden_dims[0], hidden_activation)
        self.hidden_layers = nn.ModuleList([self.create_layer(hidden_dims[k], hidden_dims[k+1], hidden_activation) for k in range(self.n_layers-1)])
        self.output_layer = self.create_layer(hidden_dims[-1], output_dim, output_activation)
    
    def create_layer(self, in_dim, out_dim, activation):
        if activation == 'relu':
            layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        elif activation == 'id':
            layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Identity())
        elif activation == 'softplus':
            layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus())
        return layer
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        --------------
        Arguments:
            x (torch.tensor): input tensor of size (batch_size, input_dim).
        
        --------------
        Returns:
            out (torch.tensor): output tensor of size (batch_size, output_dim).
        """

        out = self.input_layer(x)
        for i in range(self.n_layers-1):
            out = self.hidden_layers[i](out)
        out = self.output_layer(out)
        return out
    

class TimeGridNetwork(nn.Module):
    """
    Neural network architecture with one feedforward network per maturity.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, N_maturities, hidden_activation='relu', output_activation='id'):
        """
        Initializes a TimeGridNetwork object.
        
        --------------
        Arguments:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): Dimensions of the hidden layers.
            output_dim (int): Dimension of the output.
            N_maturities (int): Number of timesteps (or maturities).
        """
        super(TimeGridNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_layers = len(self.hidden_dims)
        self.N_maturities = N_maturities
        
        self.networks = nn.ModuleList([FeedForwardNetwork(input_dim, hidden_dims, output_dim, 
                                                          hidden_activation, output_activation) 
                                       for i in range(N_maturities)
                                      ])
        
    def forward_id(self, id_network, x):
        """
        Performs a forward pass through the network corresponding to a specific timestep.
        
        --------------
        Arguments:
            id_network (int): Index of the network corresponding to the timestep.
            x (torch.tensor): Input tensor of size (batch_size, input_dim).
        
        --------------
        Returns:
            out (torch.tensor): Output tensor of size (batch_size, output_dim).
        """

        out = self.networks[id_network](x)
        return out
    
    def freeze(self, *id_networks):
        """
        Freezes the parameters of specified networks or all networks if no arguments are provided.
        
        --------------
        Arguments:
            *id_networks (tuple of int): Indices of the networks to freeze.
        """
        
        if not id_networks:
            for param in self.networks.parameters():
                param.requires_grad = False
        
        else:
            self.unfreeze()
            for id_network in id_networks:
                for param in self.networks[id_network].parameters():
                    param.requires_grad = False
        
    def unfreeze(self, *id_networks):
        """
        Unfreezes the parameters of specified networks or all networks if no arguments are provided.
        
        --------------
        Arguments:
            *id_networks (tuple of int): Indices of the networks to unfreeze.
        """
        
        if not id_networks:
            for param in self.networks.parameters():
                param.requires_grad = True
        
        else:
            self.freeze()
            for id_network in id_networks:
                for param in self.networks[id_network].parameters():
                    param.requires_grad = True
########################################################################


###################### MAIN NEURAL SDE CLASS ###########################
class NeuralSDE(nn.Module):
    """
    A generic abstract class representing a neural stochastic differential equation (SDE) model.
    
    --------------
    Arguments:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device to be used for computations ('cpu' or 'cuda').
        rate (float): The interest rate of the market.
    
    --------------
    Attributes:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device used for computations ('cpu' or 'cuda').
        rate (float): Learning rate for optimization.
        is_training_possible (bool): Indicates if training is possible.
        has_control_variates (bool): Indicates if control variates are used.
        data (pd.DataFrame): Data for training.
        time_grid (torch.Tensor): Time grid for the SDE.
        N_prices (int): Number of prices.
        N_steps (int): Number of steps in the time grid.
        maturities (torch.Tensor): Maturities for the SDE.
        N_maturities (int): Number of maturities.
    """
    
    def __init__(self, input_dim: int, hidden_dims: int, device: str, rate: float) -> None:
        super(NeuralSDE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.rate = rate
        
        # Architecture of the diffusion
        self.set_diffusion()
        
        # No data available, hence no control variates are instantiated
        self.is_calling_possible = False
        self.has_control_variates = False
        
        self.__name__ = 'NeuralSDE'
        
    def prepare_training(self, data: pd.core.frame.DataFrame, time_grid: torch.Tensor, retrain: bool = False, warn: bool = True) -> None:
        """
        Prepares the model for training.
        
        --------------
        Arguments:
            data (pd.DataFrame): Data for training. Must be a pandas DataFrame with at least the three columns 'strike', 'maturity' and 'call'.
            time_grid (torch.Tensor): Time grid for the SDE.
            retrain (bool, Optional): Set to True if you want to train a model whose method prepare_training has already been called. Default to False.
            warn (bool, Optional): Set to False to shutdown the device-related warning. Default to True.
        
        --------------
        Raises:
            TypeError: If the passed data is not a pandas DataFrame.
            KeyError: If the passed data does not contain required columns.
            ValueError: If some maturities of the call options in the data are not in the passed time grid.
        """
        if self.is_calling_possible and not retrain:
            warning_message = f"The model has already been prepared for training. Set model.prepare_training(data, time_grid, retrain = True) to retrain it."
            warnings.warn(warning_message, stacklevel=1)
            return self
        
        if not isinstance(data, pd.core.frame.DataFrame):
            raise TypeError('Passed data must be a pandas DataFrame.')
        if not ('strike' in data.columns and 'maturity' in data.columns and 'call' in data.columns):
            raise KeyError("Passed data must contains the three columns 'strike', 'maturity', and 'call'.")
        if not isinstance(time_grid, torch.Tensor):
            raise TypeError('Passed time_grid must be a torch.Tensor.')
        
        self.data = data.sort_values(by=['maturity', 'strike']).reset_index()
        self.data = self.data[['strike', 'maturity', 'call']]
        self.data = torch.tensor(self.data.values, dtype=torch.float32, device=self.device).round(decimals=4)
        self.N_prices = len(data)
        
        self.time_grid = time_grid.to(self.device).round(decimals=4) # 0 = [t_0 < t_1 < ... < t_{N_steps}] = 1
        self.N_steps = len(time_grid) - 1
        
        # Architecture of the diffusion
        maturities = self.data[:, 1].unique()
        if not all(m in self.time_grid for m in maturities):
            raise ValueError('Some maturities are not in the passed time_grid.')
        self.set_diffusion(maturities)
        
        # Architecture of the control variates
        if self.__name__ == 'NeuralLVModel':
            d = 1 # See documentation of method set_control_variates.
        elif self.__name__ == 'NeuralLSVModel':
            d = 2
        else:
            raise NotImplementedError(f'The model {self.__name__} is not supported for now.')
        self.set_control_variates(d)
        
        # Checking wheter the model is on the right device
        if next(self.parameters()).device != self.device and warn:
            warning_message = f"The passed device ('{self.device}') is different from the device of the parameters ('{next(self.parameters()).device}'). Do not forget to run YourModel = YourModel.to('{self.device}') before calling the model."
            warnings.warn(warning_message, stacklevel=1)
        self.is_calling_possible = True # Model is now ready
        self.train()
        
    
    def prepare_eval(self) -> None:
        """
        Prepares the model for evaluation.
        """
        
        self.eval()
    
    
    def set_diffusion(self, maturities: torch.Tensor = None) -> None:
        """
        Sets the diffusion models.
        
        --------------
        Arguments:
            maturities (torch.Tensor, optional): Maturities for the SDE.
        
        --------------
        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        
        if maturities is None:
            self.maturities = torch.tensor([1.], dtype=torch.float32, device=self.device).round(decimals=4)
        else:
            self.maturities = maturities
            
        self.N_maturities = len(self.maturities)
        
        raise NotImplementedError("The set_diffusion method needs to be implemented.")
    
    
    def set_control_variates(self, d) -> None:
        """
        Sets the control variates.
        
        --------------
        Arguments:
            d: Increase in the dimension of the input of the control variate exotics.
               For instance, the LV model requires both the whole path of S as well as the current time t (+1 dimension).
               Thus, d = 1 for the LV model.
               For the LSV model, the input of the controle variate is the whole path of S, t, and V_t. Thus, d = 2.
        """
        
        # input is (S_t, t), and there is one output for each call.
        self.control_variate_vanilla = TimeGridNetwork(input_dim = self.input_dim+1,
                                                       hidden_dims = [30, 30, 30], 
                                                       output_dim = self.N_prices, 
                                                       N_maturities = self.N_maturities)
        
        # input is the whole path ((S_t)_{t in [0,T]}, t)
        self.control_variate_exotics = TimeGridNetwork(input_dim = self.input_dim*(self.N_steps+1) + d,
                                                       hidden_dims = [20, 20, 20],
                                                       output_dim = 1,
                                                       N_maturities = self.N_maturities)

        self.has_control_variates = True
    
    
    def compute_vanilla_price(self, current_time: float, S_t: torch.Tensor, vanilla_cv: torch.Tensor, vanilla_cv_price: torch.Tensor, vanilla_cv_price_variance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the vanilla price.
        
        --------------
        Arguments:
            t (float): Time at which we compute the vanilla prices.
            S_t (torch.Tensor): Tensor of stock prices.
            vanilla_cv (torch.Tensor): Tensor of vanilla control variates.
            vanilla_cv_price (torch.Tensor): Tensor to store vanilla prices.
            vanilla_cv_price_variance (torch.Tensor): Tensor to store vanilla price variances.
        
        --------------
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated vanilla price and variance.
        """
        
        mask = (self.data[:, 1] - current_time).abs() < torch.tensor(1e-5)
        shift = mask.nonzero()[0,0]
        strike_series = self.data[mask][:, 0]
        
        for id_strike, strike in enumerate(strike_series):
            id_vanilla = id_strike + shift
            control_variate = vanilla_cv.view(-1, self.N_prices)
            discounted_vanilla_payoff = torch.exp(-self.rate*current_time) * torch.clamp(S_t - strike, 0).squeeze(1)
            vanilla_price = discounted_vanilla_payoff - control_variate[:, id_vanilla]

            vanilla_cv_price[id_vanilla] = vanilla_price.mean()
            vanilla_cv_price_variance[id_vanilla] = vanilla_price.var()
        
        return vanilla_cv_price, vanilla_cv_price_variance
                    
        
    def compute_exotic_price(self, monte_carlo_paths: torch.Tensor, exotic_cv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the exotic price.
        
        --------------
        Arguments:
            monte_carlo_paths (torch.Tensor): Monte Carlo paths.
            exotic_cv (torch.Tensor): Exotic control variates.
        
        --------------
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed exotic price and variance.
        
        --------------
        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("The compute_exotic_price method must be implemented.")
    
    
    def update_cv_attributes(self, id_network, paths, t, S_t, dS_bar, current_time, exotic_cv, vanilla_cv, vanilla_cv_price, vanilla_cv_price_variance, V_t=None):
       
        # Update of the exotic control variate
        if self.__name__ == 'NeuralLSVModel':
            if V_t is None:
                raise ValueError("Must provide value for volatility process V_t to update control variates.")
            exotic_cv += self.control_variate_exotics.forward_id(id_network, torch.cat([paths.detach(), V_t.detach(), t], dim=1)) * dS_bar
        elif self.__name__ == 'NeuralLVModel':
            exotic_cv += self.control_variate_exotics.forward_id(id_network, torch.cat([paths.detach(), t], dim=1)) * dS_bar
        else:
            raise NotImplementedError(f'The model {self.__name__} is not supported for now.')
        
        # Update of the vanilla control variates attributes
        if self.training:
            vanilla_cv += self.control_variate_vanilla.forward_id(id_network, torch.cat([S_t.detach(), t], dim=1)) * dS_bar.repeat(1, self.N_prices)

            if current_time in self.maturities:
                vanilla_cv_price, vanilla_cv_price_variance = self.compute_vanilla_price(current_time = current_time, 
                                                                                         S_t = S_t, 
                                                                                         vanilla_cv = vanilla_cv, 
                                                                                         vanilla_cv_price = vanilla_cv_price, 
                                                                                         vanilla_cv_price_variance = vanilla_cv_price_variance)

        return exotic_cv, vanilla_cv, vanilla_cv_price, vanilla_cv_price_variance
    
    
    def diffuse_pieces(self, t0, t1, S_t, N_trn, d=1, corr_matrix=None, seed=None):
        """
        Computes useful quantities for the diffusion.
        
        --------------
        Arguments:
            t0: Start time.
            t1: End time.
            S_t: Tensor of stock prices.
            N_trn: Number of training samples to generate.
            d: Dimension of the Brownian motion to generate
            corr_matrix: Correlation matrix of the brownian motion. Default to None.
            seed: Random seed for reprocudibility.
        
        --------------
        Returns:
            dt: elapsed time between t0 and t1.
            dW: brownian increments of size (N_trn, d), with correlation given by corr_matrix
            t (torch.Tensor): tensor of same shape as S_t, filled with t0.
            id_network: the id of the network to use for the range of time [t0, t1].   
        """
        
        dt = t1 - t0
        dW = BrownianIncrements(self.device)(t0 = t0, 
                                             t1 = t1, 
                                             d = d, 
                                             N_trn = N_trn,
                                             corr_matrix = corr_matrix,
                                             seed = seed) # To get the same randomness at each forward pass, the seed is fixed
        t = torch.full_like(S_t, t0)
        id_network = (self.maturities >= t0).nonzero()[0, 0] # Find i such that T_{i-1} <= t < T_{i} 
        
        return dt, dW, t, id_network
    
    
    def diffuse(self, t0, t1, S_t, N_trn, seed):
        """
        Diffuses the SDE from time t0 to one step ahead at time t1.
        
        --------------
        Arguments:
            t0: Start time.
            t1: End time.
            S_t: Tensor of stock prices.
            N_trn: Number of training samples to generate.
            seed: Random seed for reprocudibility.
        
        --------------
        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        
        raise NotImplementedError("The diffuse method needs to be implemented.")

    
    def forward(self, s0: float, N_trn: int = 40000):
        """
        Forward pass through the model.
        
        --------------
        Arguments:
            s0 (float): Initial stock price.
            N_trn (int, optional): Number of training samples.
        
        --------------
        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        
        raise NotImplementedError("The forward method needs to be implemented.")
########################################################################


######################## LV AND LSV MODELS #############################
class NeuralLVModel(NeuralSDE): # LV for local volatility
    """
    A class implementing the Neural Local Volatility model.
    
    --------------
    Arguments:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device to be used for computations ('cpu' or 'cuda').
        rate (float): The interest rate of the market.
    
    --------------
    Attributes:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device used for computations ('cpu' or 'cuda').
        rate (float): Learning rate for optimization.
        is_training_possible (bool): Indicates if training is possible.
        has_control_variates (bool): Indicates if control variates are used.
        data (pd.DataFrame): Data for training.
        time_grid (torch.Tensor): Time grid for the SDE.
        N_prices (int): Number of prices.
        N_steps (int): Number of steps in the time grid.
        maturities (torch.Tensor): Maturities for the SDE.
        N_maturities (int): Number of maturities.
    """
    
    def __init__(self, **kwargs):
        super(NeuralLVModel, self).__init__(**kwargs)
        
        self.__name__ = 'NeuralLVModel'
    
    def set_diffusion(self, maturities=None):
        if maturities is None:
            self.maturities = torch.tensor([1.], dtype=torch.float32, device=self.device).round(decimals=4)
        else:
            self.maturities = maturities
            
        self.N_maturities = len(self.maturities)
        self.volatility = TimeGridNetwork(input_dim = self.input_dim+1, # input = (S_t, t) 
                                          hidden_dims = self.hidden_dims, 
                                          output_dim = 1, 
                                          N_maturities = self.N_maturities,
                                          output_activation = 'softplus')
    
    
    def compute_exotic_price(self, monte_carlo_paths, exotic_cv):
        """
        Computes the price of an exotic Lookback call option.
        
        --------------
        Arguments:
            monte_carlo_paths (torch.Tensor): Monte Carlo paths.
            exotic_cv (torch.Tensor): Exotic control variates.
        
        --------------
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed exotic price and variance.
        """

        S_max, _ = monte_carlo_paths.max(dim=1)
        S_T, T = monte_carlo_paths[:, self.N_steps], self.time_grid[self.N_steps]
        discounted_exotic_payoff = torch.exp(-self.rate*T)*(S_max - S_T) - exotic_cv.squeeze(1)
        
        exotic_cv_price = discounted_exotic_payoff.mean()
        exotic_cv_price_variance = discounted_exotic_payoff.var()
        
        return exotic_cv_price, exotic_cv_price_variance
    
    
    def diffuse(self, t0, t1, S_t, N_trn, seed):
        """
        This function is the heart of the whole class.
        It implements the Tamed Euler Scheme for the diffusion associated to the model, 
        and propagates the state variable S_t forward in time.
    
        --------------
        Arguments:
            t0 (float): Initial time.
            t1 (float): Final time.
            S_t (torch.Tensor): State variable at time t0.
            N_trn (int): Number of training paths.
            seed (int): Seed for reproducibility of randomness.

        --------------
        Returns:
            dt (float): Time step t1 - t0.
            dW (torch.Tensor): Brownian increments.
            t (torch.Tensor): Time tensor filled with t0 values.
            S_t (torch.Tensor): State variable, same as the argument.
            id_network (int): Index of maturity time interval.
            drift (torch.Tensor): Drift term.
            vol (torch.Tensor): Volatility term.
            dS_bar (torch.Tensor): Increment of e^(-rt) * S_t.
        """
        
        # Useful pieces
        dt, dW, t, id_network = self.diffuse_pieces(t0, t1, S_t, N_trn, d=1, corr_matrix=None, seed=seed)
        
        # Tamed Euler scheme
        sigma = self.volatility.forward_id(id_network, torch.cat([S_t, t], dim=1)) # sigma_i(S_t, t)
        
        drift = self.rate*S_t / (1 + self.rate*S_t.detach()*torch.sqrt(dt))
        vol = S_t*sigma / (1 + S_t.detach()*sigma.detach()*torch.sqrt(dt))
        
        # Discounted asset price. We compute it to evaluate the hedge later on.
        dS_bar = torch.exp(-self.rate*t1) * S_t.detach() * sigma.detach() * dW
        
        return dt, dW, t, S_t, id_network, drift, vol, dS_bar
    
    
    def forward(self, s0: float, N_trn: int = 40000):
        """
        Forward pass through the model.
        
        --------------
        Arguments:
            s0 (float): Initial stock price.
            N_trn (int, optional): Number of training samples.
        
        --------------
        Returns:
            In training mode: vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance
            In evaluation mode: exotic_cv_price, exotic_cv_price_variance
        """
        
        if not self.is_calling_possible:
            raise RuntimeError("Can not perform forward pass as the model has not been trained. Use self.prepare_training beforehand.")
        
        ## Variables ##
        vanilla_cv = torch.zeros(N_trn, self.N_prices, device=self.device)
        vanilla_cv_price = torch.zeros(self.N_prices, device=self.device)
        vanilla_cv_price_variance = torch.zeros(self.N_prices, device=self.device)
        
        exotic_cv = torch.zeros(N_trn, 1, device=self.device)
        exotic_cv_price = torch.zeros(1, device=self.device)
        exotic_cv_price_variance = torch.zeros(1, device=self.device)

        monte_carlo_paths = torch.zeros(N_trn, self.N_steps+1, device=self.device)
        monte_carlo_paths[:, 0] = s0
        
        ## Diffusion ##
        for k in range(1, self.N_steps + 1):

            # Diffusion from time k to time k+1
            dt, dW, t, S_t, id_network, drift, vol, dS_bar = self.diffuse(t0 = self.time_grid[k-1], 
                                                                          t1 = self.time_grid[k], 
                                                                          S_t = monte_carlo_paths[:, k-1].unsqueeze(1).clone(), # of shape (N_trn, 1).
                                                                          N_trn = N_trn, 
                                                                          seed = None)
            
            # Update of the control variates attributes
            exotic_cv, vanilla_cv, vanilla_cv_price, vanilla_cv_price_variance = self.update_cv_attributes(id_network = id_network,
                                                                                                           paths = monte_carlo_paths,
                                                                                                           t = t,
                                                                                                           S_t = S_t,
                                                                                                           dS_bar = dS_bar,
                                                                                                           current_time = self.time_grid[k],
                                                                                                           exotic_cv = exotic_cv,
                                                                                                           vanilla_cv = vanilla_cv,
                                                                                                           vanilla_cv_price = vanilla_cv_price,
                                                                                                           vanilla_cv_price_variance = vanilla_cv_price_variance)
            
            monte_carlo_paths[:, k] = (S_t + drift*dt + vol*dW).squeeze(1)
        
        ## Exotic price ##
        exotic_cv_price, exotic_cv_price_variance = self.compute_exotic_price(monte_carlo_paths, exotic_cv)
        
        if self.training:
            return vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance
        else:
            return exotic_cv_price, exotic_cv_price_variance

        
class NeuralLSVModel(NeuralSDE): # LV for local stochastic volatility
    """
    A class implementing the Neural Local Volatility model.
    
    --------------
    Arguments:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device to be used for computations ('cpu' or 'cuda').
        rate (float): The interest rate of the market.
    
    --------------
    Attributes:
        input_dim (int): Dimensionality of the input.
        hidden_dims (int): Dimensionality of the hidden layers.
        device (str): Device used for computations ('cpu' or 'cuda').
        rate (float): Learning rate for optimization.
        is_training_possible (bool): Indicates if training is possible.
        has_control_variates (bool): Indicates if control variates are used.
        data (pd.DataFrame): Data for training.
        time_grid (torch.Tensor): Time grid for the SDE.
        N_prices (int): Number of prices.
        N_steps (int): Number of steps in the time grid.
        maturities (torch.Tensor): Maturities for the SDE.
        N_maturities (int): Number of maturities.
    """
    
    def __init__(self, **kwargs):
        super(NeuralLSVModel, self).__init__(**kwargs)
        
        self.__name__ = 'NeuralLSVModel'
    
    def set_diffusion(self, maturities=None):
        if maturities is None:
            self.maturities = torch.tensor([1.], dtype=torch.float32, device=self.device).round(decimals=4)
        else:
            self.maturities = maturities
            
        self.N_maturities = len(self.maturities)
        self.volatility_S = TimeGridNetwork(input_dim = self.input_dim+2, # input = (S_t, V_t, t) 
                                            hidden_dims = self.hidden_dims, 
                                            output_dim = 1, 
                                            N_maturities = self.N_maturities,
                                            output_activation = 'softplus')
        self.volatility_V = TimeGridNetwork(input_dim = 1, # input = V_t
                                            hidden_dims = self.hidden_dims, 
                                            output_dim = 1, 
                                            N_maturities = self.N_maturities,
                                            output_activation = 'softplus')
        self.drift_V = TimeGridNetwork(input_dim = 1, # input = V_t
                                       hidden_dims = self.hidden_dims, 
                                       output_dim = 1, 
                                       N_maturities = self.N_maturities,
                                       output_activation = 'id')
        self.v0 = nn.Parameter(torch.rand(1)-3) # Uniformely drawn from [-2, -1), but I don't know why LOL.
        self.rho = nn.Parameter(2*torch.rand(1)-1)
        self.corr_matrix = torch.tensor([[1, 0], [self.rho, torch.sqrt(1-self.rho**2)]], device=self.device)
    
    
    def compute_exotic_price(self, monte_carlo_paths, exotic_cv):
        """
        Computes the price of an exotic Lookback call option.
        
        --------------
        Arguments:
            monte_carlo_paths (torch.Tensor): Monte Carlo paths.
            exotic_cv (torch.Tensor): Exotic control variates.
        
        --------------
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed exotic price and variance.
        """

        S_max, _ = monte_carlo_paths.max(dim=1)
        S_T, T = monte_carlo_paths[:, self.N_steps], self.time_grid[self.N_steps]
        discounted_exotic_payoff = torch.exp(-self.rate*T)*(S_max - S_T) - exotic_cv.squeeze(1)
        
        exotic_cv_price = discounted_exotic_payoff.mean()
        exotic_cv_price_variance = discounted_exotic_payoff.var()
        
        return exotic_cv_price, exotic_cv_price_variance
    
    
    def diffuse(self, t0, t1, S_t, V_t, N_trn, seed):
        """
        This function is the heart of the whole class.
        It implements the Tamed Euler Scheme for the diffusion associated to the model, 
        and propagates the state variable S_t forward in time.
    
        --------------
        Arguments:
            t0 (float): Initial time.
            t1 (float): Final time.
            S_t (torch.Tensor): State variable at time t0.
            V_t (torch.Tensor): Volatility variable at time t0.
            N_trn (int): Number of training paths.
            seed (int): Seed for reproducibility of randomness.

        --------------
        Returns:
            dt (float): Time step t1 - t0.
            dW_S (torch.Tensor): Brownian increments of dimension 1.
            dW_V (torch.Tensor): Brownian increments of dimension 1, correlated with dW_V (self.rho)
            t (torch.Tensor): Time tensor filled with t0 values.
            S_t (torch.Tensor): State variable, same as the argument.
            V_t (torch.Tensor): Volatility variable, same as the argument.
            id_network (int): Index of maturity time interval.
            drift_S (torch.Tensor): Drift term of S_t.
            vol_S (torch.Tensor): Volatility term of S_t.
            drift_V (torch.Tensor): Drift term of V_t.
            vol_V (torch.Tensor): Volatility term of V_t.
            dS_bar (torch.Tensor): Increment of e^(-rt) * S_t.
        """
        
        # Useful pieces
        dt, dW, t, id_network = self.diffuse_pieces(t0, t1, S_t, N_trn, d=2, corr_matrix=self.corr_matrix, seed=seed)
        dW_S, dW_V = dW[:, :1], dW[:, 1:]
        
        # Tamed Euler scheme
        sigma_S = self.volatility_S.forward_id(id_network, torch.cat([S_t, V_t, t], dim=1)) # sigma_i(S_t, V_t, t)
        
        drift_S = self.rate*S_t / (1 + self.rate*S_t.detach()*torch.sqrt(dt))
        vol_S = S_t*sigma_S / (1 + S_t.detach()*sigma_S.detach()*torch.sqrt(dt))
        
        drift_V = self.drift_V.forward_id(id_network, V_t)
        vol_V = self.volatility_V.forward_id(id_network, V_t)

        # Discounted asset price. We compute it to evaluate the hedge later on.
        dS_bar = torch.exp(-self.rate*t1) * S_t.detach() * sigma_S.detach() * dW_S
        
        return dt, dW_S, dW_V, t, S_t, V_t, id_network, drift_S, vol_S, drift_V, vol_V, dS_bar
    
    
    def forward(self, s0: float, N_trn: int = 40000):
        """
        Forward pass through the model.
        
        --------------
        Arguments:
            s0 (float): Initial stock price.
            N_trn (int, optional): Number of training samples.
        
        --------------
        Returns:
            In training mode: vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance
            In evaluation mode: exotic_cv_price, exotic_cv_price_variance
        """
        
        if not self.is_calling_possible:
            raise RuntimeError("Can not perform forward pass as the model has not been trained. Use self.prepare_training beforehand.")
        
        ## Variables ##
        vanilla_cv = torch.zeros(N_trn, self.N_prices, device=self.device)
        vanilla_cv_price = torch.zeros(self.N_prices, device=self.device)
        vanilla_cv_price_variance = torch.zeros(self.N_prices, device=self.device)
        
        exotic_cv = torch.zeros(N_trn, 1, device=self.device)
        exotic_cv_price = torch.zeros(1, device=self.device)
        exotic_cv_price_variance = torch.zeros(1, device=self.device)

        monte_carlo_paths = torch.zeros(N_trn, self.N_steps+1, device=self.device)
        monte_carlo_paths[:, 0] = s0
        
        # To ensure that v0 is well-defined, we define it as follows
        V_t = torch.ones((N_trn, 1), device=self.device)*torch.sigmoid(self.v0)*1/2
        
        ## Diffusion ##
        for k in range(1, self.N_steps + 1):

            # Diffusion from time k to time k+1
            dt, dW_S, dW_V, t, S_t, V_t, id_network, drift_S, vol_S, drift_V, vol_V, dS_bar = self.diffuse(t0 = self.time_grid[k-1], 
                                                                                                           t1 = self.time_grid[k], 
                                                                                                           S_t = monte_carlo_paths[:, k-1].unsqueeze(1).clone(), # of shape (N_trn, 1).
                                                                                                           V_t = V_t,
                                                                                                           N_trn = N_trn, 
                                                                                                           seed = None)
            
            # Update of the control variates attributes
            exotic_cv, vanilla_cv, vanilla_cv_price, vanilla_cv_price_variance = self.update_cv_attributes(id_network = id_network,
                                                                                                           paths = monte_carlo_paths,
                                                                                                           t = t,
                                                                                                           S_t = S_t,
                                                                                                           V_t = V_t,
                                                                                                           dS_bar = dS_bar,
                                                                                                           current_time = self.time_grid[k],
                                                                                                           exotic_cv = exotic_cv,
                                                                                                           vanilla_cv = vanilla_cv,
                                                                                                           vanilla_cv_price = vanilla_cv_price,
                                                                                                           vanilla_cv_price_variance = vanilla_cv_price_variance)
            
            monte_carlo_paths[:, k] = (S_t + drift_S*dt + vol_S*dW_S).squeeze(1) # Saving (S_t) path
            V_t = torch.clamp(V_t + drift_V*dt + vol_V*dW_V, 0) # From V_{t_{k-1}} to V_{t_k}
        
        ## Exotic price ##
        exotic_cv_price, exotic_cv_price_variance = self.compute_exotic_price(monte_carlo_paths, exotic_cv)
        
        if self.training:
            return vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance
        else:
            return exotic_cv_price, exotic_cv_price_variance
########################################################################


########################## TRAINER CLASS ###############################
class NeuralTrainer:
    """
    A class to ease Neural SDEs training.
    """
    
    allowed_models = ['NeuralLVModel', 'NeuralLSVModel']
    allowed_training_problems = ['standard', 'lower bound', 'upper bound']
    
    def __init__(self, model, problem):
        """

        Initializes the Neural trainer.
        --------------
        Arguments:
            model (NeuralSDE): The model to train. Check Neural.Trainer.allowed_models to get the list of trainable models.
            problem (str): The problem solved. Must be one of 'standard', 'lower bound', 'upper bound'.
        """
        
        if model.__name__ not in NeuralTrainer.allowed_models:
            raise RuntimeError(f"The model {model.__name__} is not recognized. Allowed models are {NeuralTrainer.allowed_models}.")
        
        if problem not in NeuralTrainer.allowed_training_problems:
            raise ValueError(f"Training problem '{problem}' unknown. Allowed problems are {NeuralTrainer.allowed_training_problems}.")
            
        self.model = model
        self.problem = problem
        self.training_done = False
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=1.5)
            
    
    def prepare_training(self, data, time_grid, device, schedule="steps", milestones=[500, 800], patience=30, cooldown=10, factor=0.2, gamma=0.2):
        """
        Prepare the passed model for training.
        Do not use this function if you want to resume an already started training. Directly use "NeuralTrainer.train(...)".
        
        --------------
        Arguments:
            data (pd.DataFrame): Data for training. Must be a pandas DataFrame with at least the three columns 'strike', 'maturity' and 'call'.
            time_grid (torch.Tensor): Time grid for the SDE.
            device (str): Device to perform computations on, e.g., 'cpu' or 'cuda'.
            schedule (str, optional): Which kind of learning rate scheduler to use. Either 'plateau' for ReduceLROnPlateau or 'steps' for MultiStepLR. Default to 'steps'.
            milestones (list of int): Milestones for the reduction of learning rate of the MultiStepLR. Ignored if schedule = 'plateau'. Default to [500, 800].
            patience (int): Patience of the ReduceLROnPlateau. Ignored if schedule = 'steps'. Default to 30.
            cooldown (int): Cool down of the ReduceLROnPlateau. Ignored if schedule = 'steps'. Default to 10.
            factor (float): Factor of the decrease of the learning rate for the ReduceLROnPlateau. Ignored if schedule = 'steps'. Default to 0.2.
            gamma (float): Factor of the decrease of the learning rate for the MultiStepLR. Ignored if schedule = 'plateau'. Default to 0.2.
        """
        
        self.model.prepare_training(data, time_grid, warn=False)
        self.model = self.model.to(device)
        self.model.apply(self.init_weights)
        
        if self.model.__name__ == 'NeuralLVModel':
            params_SDE = list(self.model.volatility.parameters())
        elif self.model.__name__ == 'NeuralLSVModel':
            params_SDE = list(self.model.volatility_S.parameters()) + list(self.model.volatility_V.parameters()) \
                       + list(self.model.drift_V.parameters()) + [self.model.v0, self.model.rho]
        else:
            raise NotImplementedError(f'The model {self.__name__} is not supported for now.')
            
        params_CV = list(self.model.control_variate_vanilla.parameters()) + list(self.model.control_variate_exotics.parameters())

        self.schedule = schedule
        self.optimizer_SDE = optim.Adam(params_SDE, lr=0.001)
        self.optimizer_CV = optim.Adam(params_CV, lr=0.001)
        if self.schedule == 'plateau':
            self.scheduler_SDE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_SDE, mode='min', factor=factor, patience=patience, threshold=1e-4, threshold_mode='rel', cooldown=cooldown, min_lr=0.00001)
        elif self.schedule == 'steps':
            self.scheduler_SDE = optim.lr_scheduler.MultiStepLR(self.optimizer_SDE, milestones=milestones, gamma=gamma)
        else:
            raise ValueError("Entered schedule is not supported. Must be one of 'plateau' or 'steps'.")
        
        # We sort the data because it is done internally in self.model.prepare_training.
        self.targets = torch.tensor(data.sort_values(by=['maturity', 'strike'])['call'].values, dtype=torch.float32, device=device).round(decimals=4)
        self.N_prices = self.targets.numel()
        self.loss_SDE = nn.MSELoss()#(reduction="sum")

    def save(self, accuracy=None):
        """
        Save the current model. 

        --------------
        Arguments:
            accuracy (float): the model accuracy. For instance, if the current model has a MSE of 2e-6, accuracy = 2e-6.
        """
        
        model_name = self.model.__name__ + '_' + self.problem.replace(' ', '')
        if accuracy is not None:
            model_name += '_{:1.0e}'.format(accuracy)
        model_name += '.pt'
        torch.save(self.model, model_name)
        torch.save(self.training_records, 'records_' + model_name)

        return model_name
    
    def set_params_require_grad(self, mode):
        if mode == 'SDE':
            if self.model.__name__ == 'NeuralLVModel':
                self.model.volatility.unfreeze()
            elif self.model.__name__ == 'NeuralLSVModel':
                self.model.volatility_S.unfreeze()
                self.model.volatility_V.unfreeze()
                self.model.drift_V.unfreeze()
                self.model.v0.requires_grad_(True)
                self.model.rho.requires_grad_(True)
            else:
                raise NotImplementedError(f'The model {self.__name__} is not supported for now.')
            self.model.control_variate_vanilla.freeze()
            self.model.control_variate_exotics.freeze()
        elif mode == 'CV':
            self.model.control_variate_vanilla.unfreeze() 
            self.model.control_variate_exotics.unfreeze() 
            if self.model.__name__ == 'NeuralLVModel':
                self.model.volatility.freeze()
            elif self.model.__name__ == 'NeuralLSVModel':
                self.model.volatility_S.freeze()
                self.model.volatility_V.freeze()
                self.model.drift_V.freeze()
                self.model.v0.requires_grad_(False)
                self.model.rho.requires_grad_(False)
            else:
                raise NotImplementedError(f'The model {self.__name__} is not supported for now.')
        else:
            raise ValueError(f"{mode} is not an accepted mode. Choose between 'SDE' and 'CV'.")

            
    def train_SDE(self, s0: float, N_trn: int, N_batchs: int = 20, verbose: bool = False):
        
        t0 = time.time()
        mse_record = np.zeros(N_batchs)
        time_record = np.zeros(N_batchs)
        for i in range(1, N_batchs+1):
            t_batch_start = time.time()
            self.optimizer_SDE.zero_grad()
            
            # Forward stage
            vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance = self.model(s0, N_trn)
            
            # Backward stage
            mse = self.loss_SDE(vanilla_cv_price, self.targets)
            if self.problem == 'standard':
                loss = mse
            else:
                objective = exotic_cv_price if self.problem == 'lower bound' else - exotic_cv_price
                loss = objective + self.lambda_ * mse + self.c/2 * mse**2
            loss.backward()
            
            self.optimizer_SDE.step()
            
            # Save record
            t_batch_end = time.time()
            mse_record[i-1] = mse.item()
            time_record[i-1] = t_batch_end - t_batch_start
            
            if verbose:
                print('Iteration {}/{} | SDE loss = {:2.3e}, MSE = {:2.3e}, exotic = {:2.4e}, time = {:.2f}'.format(i, N_batchs, loss.item(), mse.item(), exotic_cv_price.item(), t_batch_end - t_batch_start))
            
        mse_SDE = mse_record.mean()
        if self.schedule == 'plateau':
            self.scheduler_SDE.step(mse_SDE)
            if verbose:
                print('Iteration {}/{} | SDE number of bad epochs = {}/{}, current MSE = {:2.3e}, best MSE = {:2.3e}.'.format(i, N_batchs, self.scheduler_SDE.num_bad_epochs, self.scheduler_SDE.patience, mse_SDE, self.scheduler_SDE.best))
        elif self.schedule == 'steps':
            self.scheduler_SDE.step()
            
            
        t1 = time.time()
        
        return mse_record, time_record, t1 - t0
    
    
    def train_CV(self, s0: float, N_trn: int, N_batchs: int = 20, verbose: bool = False):
        
        t0 = time.time()
        loss_record = np.zeros(N_batchs)
        time_record = np.zeros(N_batchs)
        for i in range(1, N_batchs+1):
            t_batch_start = time.time()
            self.optimizer_CV.zero_grad()
            
            # Forward stage
            vanilla_cv_price, vanilla_cv_price_variance, exotic_cv_price, exotic_cv_price_variance = self.model(s0, N_trn)
            
            # Backward stage
            loss = vanilla_cv_price_variance.sum() + (exotic_cv_price_variance if self.problem != 'standard' else 0)
            loss.backward()
            
            self.optimizer_CV.step()
            
            # Save record
            t_batch_end = time.time()
            loss_record[i-1] = loss.item()
            time_record[i-1] = t_batch_end - t_batch_start
            
            if verbose:
                print('Iteration {}/{} | CV loss = {:3.3e}, time = {:.2f}'.format(i, N_batchs, loss.item(), t_batch_end - t_batch_start))
        
        t1 = time.time()
        
        return loss_record, time_record, t1 - t0
    
    
    def train(self, s0: float, N_trn: int = 40000, N_batchs: int = 20, N_epochs: int = 100, checkpoints: list = None, verbose: bool = False):
        """
        Trains the passed model to solve a given problem (among 'standard', 'lower bound', and 'upper bound'; see the article).
        
        --------------
        Arguments:
            s0 (float): Starting value of the underlying.
            N_trn (int, Optional): Number of Monte Carlo paths used to compute options prices. Default to 100.
            checkpoints (list, Optional): List of loss threshold at which the current model version will be saved in an external file. Default to None.
            verbose (bool, Optional): Whether to print training information or not. Default to True. It is recommended to let verbose = True.
        
        --------------
        Returns:
            training_records (numpy array): it consists in four columns: the MSE during training, the SDE training time, the CV loss, and the CV training time.
        """

        self.lambda_ = 10000.
        self.c = 20000. # lambda_ and c are only used for lower and upper bounds computation.

        self.N_batchs = N_batchs # Useful to plot graphs
        
        if checkpoints is not None:
            checkpoints = np.array(checkpoints)
            N_checkpoints = len(checkpoints)
            next_checkpoint_id = 0
        
        t_training_start = time.time()
        self.training_records = np.zeros((N_epochs*N_batchs, 4))
        mse_SDE, time_SDE, loss_CV, time_CV = np.zeros((4, N_batchs))
        for epoch in range(N_epochs):
            
            ## Training of SDE
            if epoch%10 != 1 and epoch%10 != 2:
                if verbose:
                    print("Epoch {}/{} | SDE training launched".format(epoch+1, N_epochs))
                    
                self.set_params_require_grad('SDE')
                mse_SDE, time_SDE, _ = self.train_SDE(s0, N_trn, N_batchs, verbose)
                time_CV = np.zeros(N_batchs) # for the record
                
                ## Saving the model if checkpoint reached
                if checkpoints is not None and next_checkpoint_id < N_checkpoints:
                    if mse_SDE[-1] < checkpoints[next_checkpoint_id]:
                        model_name = self.save(accuracy=checkpoints[next_checkpoint_id])
                        print("Epoch {}/{} | Checkpoint {} reached. Model {} saved.".format(epoch+1, N_epochs, checkpoints[next_checkpoint_id], model_name))
                        next_checkpoint_id += 1
                        
                if verbose:
                    print("Epoch {}/{} | SDE average MSE = {:2.3e}, time = {:.2f}".format(epoch+1, N_epochs, mse_SDE.mean(), time_SDE.sum()))
            
            ## Training of CV
            else:
                if verbose:
                    print("Epoch {}/{} | CV training launched".format(epoch+1, N_epochs))
                    
                self.set_params_require_grad('CV')
                loss_CV, time_CV, _ = self.train_CV(s0, N_trn, N_batchs, verbose)
                time_SDE = np.zeros(N_batchs) # for the record
                if verbose:
                    print("Epoch {}/{} | CV average loss = {:2.3e}, time = {:.2f}".format(epoch+1, N_epochs, loss_CV.mean(), time_CV.sum()))
            
            ## Recording training logs
            self.training_records[epoch*N_batchs:(epoch+1)*N_batchs] = np.vstack((mse_SDE, time_SDE, loss_CV, time_CV)).T
            if verbose and epoch>0 and epoch%10 == 0:
                SDE_loss_previous_avg, CV_loss_previous_avg = self.training_records[(epoch-10)*N_batchs:(epoch-9)*N_batchs, [0, 2]].mean(axis=0)
                SDE_loss_current_avg, CV_loss_current_avg = self.training_records[epoch*N_batchs:(epoch+1)*N_batchs, [0, 2]].mean(axis=0)
                elapsed_time = self.training_records[(epoch-9)*N_batchs:(epoch+1)*N_batchs, [1, 3]].sum()
                print("Epochs {} - {} | SDE loss: {:2.3e} -> {:2.3e} | CV loss: {:2.3e} -> {:2.3e} | elapsed time: {:.2f}".format(epoch-9, epoch+1, SDE_loss_previous_avg, SDE_loss_current_avg, CV_loss_previous_avg, CV_loss_current_avg, elapsed_time))
                print("Epoch 1 - {} | elapsed time: {:.2f}".format(epoch+1, time.time() - t_training_start))
            
            ## Updates of lambda_ and c for the Augmented Lagrangian method.
            if epoch%20 in [4, 7, 9, 14, 16, 19]: # on average, every 50 batchs of train_SDE.
                # self.lambda_ = self.lambda_ + self.c * mse_SDE[-1]
                # self.c = 2 * self.c
                self.lambda_ = (self.lambda_ + self.c * mse_SDE[-1] if self.lambda_ < 1e+6 else self.lambda_) # Strangely, this works well better than without the upper bound 1e+6. Same comment for self.c < 1e+10
                self.c = (2 * self.c if self.c < 1e+10 else self.c)
            
                if verbose and self.problem != 'standard':
                    print("Epoch {}/{} | Update of (lambda_, c) to ({:2.3e}, {:2.3e})".format(epoch+1, N_epochs, self.lambda_, self.c))
                
        self.training_done = True
        
        if verbose:
            first_SDE_loss, first_CV_loss = self.training_records[0, [0, 2]]
            last_SDE_loss, last_CV_loss = self.training_records[-1, [0, 2]]
            elapsed_time = self.training_records[:, [1, 3]].sum()
            print("Training finished | SDE loss: {:2.3e} -> {:2.3e} | CV loss: {:2.3e} -> {:2.3e} | elapsed time: {:.2f}".format(first_SDE_loss, last_SDE_loss, first_CV_loss, last_CV_loss, elapsed_time))
        
        self.save(accuracy=last_SDE_loss)
        return self.training_records
    
    def plot_metrics(self, training_records=None, model_name=None, N_batchs=None, fig_size=(12,6)):
        
        if training_records is None:
            if not self.training_done:
                raise RuntimeError("No training has been performed. Can not plot inexistant metrics. Perform yourNeuralTrainer.train(...) beforehand.")
            else:
                training_records = self.training_records
                model_name = self.model.__name__
                N_batchs = self.N_batchs

        fig, axes = plt.subplots(1, 2, figsize=fig_size)

        # Plot SDE loss evolution
        axes[0].plot(training_records[:, 0])
        axes[0].set_yscale('log')
        axes[0].set_xlabel(f'Batchs (1 epoch = {N_batchs} batchs)')
        axes[0].set_ylabel('MSE of calibration')
        axes[0].grid()
        axes[0].set_title(f"{model_name}, MSE evolution")

        # Plot CV loss evolution
        axes[1].plot(training_records[:, 2])
        axes[1].set_yscale('log')
        axes[1].set_xlabel(f'Batchs (1 epoch = {N_batchs} batchs)')
        axes[1].set_ylabel('CV variance')
        axes[1].grid()
        axes[1].set_title(f"{model_name}, control variates variance evolution")

        plt.tight_layout()
        plt.show()
########################################################################

