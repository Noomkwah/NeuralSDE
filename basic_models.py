#!/usr/bin/env python
# coding: utf-8


############################ IMPORTS ###################################
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Union

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
########################################################################


########################## BASIC CLASSES ###############################
class BrownianIncrements:
    def __init__(self):
        super(BrownianIncrements, self).__init__()

    def __call__(self, t0: float, t1: float, d: int, N: int, corr_matrix: np.ndarray = None) -> np.ndarray:
        """
        Generate N independent increments of a Brownian motion of dimension d between times t0 and t1.
        
        ---------------
        Arguments:
            t0: float, initial time
            t1: float, final time
            d: int, dimension of the Brownian motion
            N: int, number of increments to generate
            corr_matrix: np.ndarray, optional (default = None), correlation matrix of the brownian motions to generate.
                If it is not passed, the brownian motion is independent.
        
        ---------------
        Returns:
            increments: array, Brownian increments of size (N, d)
        """
        delta_t = t1 - t0
        increments = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(N, d))
        
        if corr_matrix is not None:
            increments = corr_matrix@increments.T
            
        return increments
    
    
class MonteCarloPaths:
    def __init__(self, paths: np.ndarray, times: np.ndarray) -> None:
        assert paths.shape[0] == times.size, 'Incompatible dimensions for paths and times.'
        
        self.times = times
        self.paths = paths
        self.N_samples = paths.shape[-1]
    
    def __call__(self, t: float) -> np.ndarray:
        i = np.where(self.times < t)[0][-1]
        t0 = self.times[i]
        t1 = self.times[i+1]
        return (t1-t)/(t1-t0) * self.paths[i] + (t-t0)/(t1-t0) * self.paths[i+1]

    
class Diffusion(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def forward(self, t0: float, t1: float, N: int) -> np.ndarray:
        pass
    
    @abstractmethod    
    def diffuse(self, T: float, N_samples: int = 1000000, pi: np.ndarray = None, N_steps: int = 100) -> MonteCarloPaths:
        pass
########################################################################


######################### DIFFUSION MODELS #############################
class BlackScholesModel(Diffusion):
    """
    Black & Scholes model. Mainly used to compute Implied Volatility.
    """
    
    ############################### INIT ####################################
    def __init__(self, r: float = None, sigma: float = None, verbose: bool = False):
        """
        Initialize the Black & Scholes model parameters.

        ---------------
        Arguments:
            r: float, optional (default = None), risk-free interest rate
            sigma: float, optional (default = None), volatility of the asset process
            verbose: bool, optional (default = False), controls the logs displayed during Monte Carlo simulation
        ---------------
        Returns:
            None
        """
        super(BlackScholesModel, self).__init__()
        
        self.r = r
        self.sigma = sigma
        self.verbose = verbose

        
    def __str__(self) -> str:
        return f"BlackScholesModel(r = {self.r}, sigma = {self.sigma})"
    #########################################################################
    
    ######################## MONTE CARLO SIMULATION #########################
    def forward(self, t0: float, t1: float, N: int, x0: Union[np.ndarray, float]) -> np.ndarray:
        """
        Simulate the asset price forward in time using the Black & Scholes model.

        ---------------
        Arguments:
            t0: float, initial time
            t1: float, final time
            N: int, number of samples
            x0: np.ndarray or float, initial asset price(s)

        ---------------
        Returns:
            x1: np.ndarray, Simulated asset prices
        """
        
        if not hasattr(x0, 'size'):
            x0 = np.array([x0] * N)
        
        # Randomness
        Delta_W = BrownianIncrements()(t0, t1, 1, N)
        Delta_t = t1 - t0
        
        # Compute x1 from x0
        x1 = x0 + self.r * x0 * Delta_t + self.sigma * x0 * Delta_W
        
        return x1
    
    
    def diffuse(self, T: float, x0: float, N_samples: int = 1000000, pi: np.ndarray = None, N_steps: int = 100) -> MonteCarloPaths:
        """
        Simulate asset price paths using the Black & Scholes model.

        ---------------
        Arguments:
            T: float, maturity time
            x0: float, initial asset price
            N_samples: int, optional (default = 1,000,000), number of samples
            pi: np.ndarray, optional (default = None), array of intermediate times
            N_steps: int, optional (default = 100), number of intermediate steps

        ---------------
        Returns:
            X: MonteCarloPaths, simulated asset prices paths
        """
        X = np.zeros((N_steps, N_samples))
        if pi is None:
            pi = np.linspace(0, T, N_steps)
        N_steps = len(pi)
        
        X[0] = x0
        for k in range(N_steps-1):
            X[k+1] = self.forward(t0=pi[k], t1=pi[k+1], N=N_samples, x0=X[k])
            
            if self.verbose:
                print(k, end = ' ' if k < N_steps-2 else '\n')
        
        return MonteCarloPaths(X, pi)
    #########################################################################
    
    ######################### IMPLIED VOLATILITY ############################
    def priceCall(self, strike: float, maturity: float, x0: float, sigma: float) -> float:
        """
        Compute the price of a European call option using the Black & Scholes model.

        ---------------
        Arguments:
            strike: float, strike price of the option
            maturity: float, maturity time of the option
            x0: float, initial asset price
            sigma: float, volatility of the asset

        ---------------
        Returns:
            float: Price of the European call option
        """
        
        d1 = (np.log(x0/strike) + (self.r + 0.5*sigma**2)*maturity)/(sigma*np.sqrt(maturity))
        d2 = d1 - sigma*np.sqrt(maturity)
        d2 = d1 - sigma * np.sqrt(maturity)
        
        call_price = x0*norm.cdf(d1) - strike*np.exp(-self.r*maturity)*norm.cdf(d2)
        
        return call_price
        
        
    def computeImpliedVolatility(self, strike: float, maturity: float, x0: float, call_price: float) -> float:
        """
        Compute the implied volatility of an European call option.

        ---------------
        Arguments:
            strike: float, strike price of the option
            maturity: float, maturity time of the option
            x0: float, initial asset price
            call_price: float, european call option price

        ---------------
        Returns:
            float: Implied volatility
        """
        
        def BS_call_price(sigma):
            return self.priceCall(strike, maturity, x0, sigma) - call_price
        
        implied_vol = brentq(BS_call_price, -1, 1e10)
        
        return implied_vol     
               
    
    def computeImpliedVolatilitySurface(self, x0: float, data: np.ndarray) -> np.ndarray:
        """
        Compute the Implied Volatility Surface.

        ---------------
        Arguments:
            x0: float, initial asset price
            data: np.ndarray of size (N, 3) where data[n] = (strike, maturity, call_price)
    
        ---------------
        Returns:
            surface: np.ndarray of shape (N, 3) where surface[n] = (strike, maturity, implied_volatility)
                Implied volatility Surface
        """
        assert data.shape[1] == 3, f'Passed data shape is {data.shape} and does not match the format (N, 3).'
        
        N = data.shape[0]
        surface = np.zeros(data.shape)
        for n in range(N):
            strike, maturity, call_price = data[n]
            implied_vol = self.computeImpliedVolatility(strike, maturity, x0, call_price)
            surface[n] = strike, maturity, implied_vol
        
        return surface

        
    def drawImpliedVolatilityCurves(self, surface, estimate=None, fig_size=(10, 5), plot_title=None):
        """
        Draw the Implied Volatility Curves.

        ---------------
        Arguments:
            surface: np.ndarray of shape (N, 3) where surface[n] = (strike, maturity, implied_volatility)
            estimate: np.ndarray of shape (N, 3), optional, where estimate[n] = (strike, maturity, estimated_implied_volatility)
            fig_size: Tuple specifying the figure size (width, height) in inches. Default is (10, 5).
            plot_title: str, title for the whole plot. Default is None.
            
        ---------------
        Plot:
            2D plot of Implied Volatility curves for each maturity.
        """
        
        if estimate is not None:
            if not np.allclose(surface[:, :2], estimate[:, :2]):
                raise ValueError("The maturities and strikes in 'estimate' must be the same (same order as well) as in 'surface'.")
        
        maturities = np.unique(surface[:, 1]) # unique maturities
        
        # Create subplots
        cols = (len(maturities) + 1) // 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Plot each maturity's implied volatility curve
        for i, maturity in enumerate(maturities):
            
            # Filter data for the current maturity
            maturity_data = surface[surface[:, 1] == maturity]
            strikes = maturity_data[:, 0]
            implied_vols = maturity_data[:, 2]
            if estimate is not None:
                maturity_estimate = estimate[estimate[:, 1] == maturity]
                estimated_implied_vols = maturity_estimate[:, 2]
            
            # Sort data for plotting
            sorted_indices = np.argsort(strikes)
            strikes_sorted = strikes[sorted_indices]
            implied_vols_sorted = implied_vols[sorted_indices]
            if estimate is not None:
                estimated_implied_vols_sorted = estimated_implied_vols[sorted_indices]
            
            # Plot implied volatility curve on the appropriate subplot
            axes[i].plot(strikes_sorted, implied_vols_sorted, label='True IV', color='blue')
            if estimate is not None:
                axes[i].plot(strikes_sorted, estimated_implied_vols_sorted, label='Estimated IV', color='red', linestyle='--')
            axes[i].set_title(f'Maturity: {maturity:.3f} year(s)')
            axes[i].set_xlabel('Strike')
            axes[i].set_ylabel('Implied Volatility' if i%cols == 0 else '')
            
            # Hide legend for all plots except the last one
            if i < len(maturities) - 1:
                axes[i].legend().set_visible(False)
        
        # Position the legend outside of the subplots
        if estimate is not None:
            axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add title to the whole plot
        if plot_title is not None:
            plt.suptitle(plot_title)
        
        plt.tight_layout()
        plt.show()


    def drawImpliedVolatilitySurface(self, surface):
        """
         Draw the Implied Volatility Surface.

        ---------------
        Arguments:
            surface: np.ndarray of shape (N, 3) where surface[n] = (strike, maturity, implied_volatility)
    
        ---------------
        Plot:
            3D plot of the Implied Volatility Surface
        """
        strikes = np.unique(surface[:, 0])  # Strikes
        maturities = np.unique(surface[:, 1])  # Maturities
        maturities, strikes = np.meshgrid(maturities, strikes)

        implied_volatility = surface[:, 2].reshape(maturities.shape) # Implied Vol

        # Plotting
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        surf = ax.plot_surface(strikes, maturities, implied_volatility, cmap='viridis', edgecolor='none')

        # Set labels and title
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Implied Volatility Surface')

        # Rotate the plot for better visualization
        ax.view_init(elev=22, azim=-50, roll=None, vertical_axis='z')

        # Add color bar
        color_bar = fig.colorbar(surf, shrink=0.5, aspect=5)
        color_bar.ax.yaxis.set_ticks_position('right')

        # Get the current position of the color bar axes
        box = color_bar.ax.get_position()

        # Set a new position for the color bar axes
        color_bar.ax.set_position([box.x0 + 0.05, box.y0, box.width * 0.8, box.height])

        # Adjust ticks and grid
        ax.xaxis.set_tick_params(pad=2)
        ax.yaxis.set_tick_params(pad=2)
        ax.zaxis.set_tick_params(pad=2)

        plt.show()
    #########################################################################


class HestonModel(Diffusion): 
    """
    Heston Stochastic Volatility model.
    """
    
    ############################### INIT ####################################
    def __init__(self, r: float = None, kappa: float = None, mu: float = None, eta: float = None, rho: float = None, verbose: bool = False):
        """
        Initialize the Heston model parameters.

        ---------------
        Arguments:
            r: float, optional (default = None), risk-free interest rate
            kappa: float, optional (default = None), rate of reversion to the mean for the variance process
            mu: float, optional (default = None), long-term mean variance
            eta: float, optional (default = None), volatility of the variance process
            rho: float, optional (default = None), correlation between the Brownian motions for the asset price and its variance
            verbose: bool, optional (default = False), controls the logs displayed during Monte Carlo simulation

        ---------------
        Returns:
            None
        """
        super(HestonModel, self).__init__()
        
        self.r = r
        self.kappa = kappa
        self.mu = mu
        self.eta = eta
        self.rho = rho
        self.corr_matrix = np.array([[1, 0], [self.rho, np.sqrt(1-self.rho**2)]])
        
        self.verbose = verbose
            
            
    def __str__(self) -> str:
        return f'HestonModel(r = {self.r}, kappa = {self.kappa}, mu = {self.mu}, eta = {self.eta}, rho = {self.rho})'
    #########################################################################
    
    ######################## MONTE CARLO SIMULATION #########################
    def forward(self, t0: float, t1: float, N: int, x0: Union[np.ndarray, float], v0: Union[np.ndarray, float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the asset price and variance forward in time using the Heston model.

        ---------------
        Arguments:
            t0: float, initial time
            t1: float, final time
            N: int, number of samples
            x0: np.ndarray or float, initial asset price(s)
            v0: np.ndarray or float, initial variance(s)

        ---------------
        Returns:
            x1: np.ndarray, Simulated asset prices
            v1: np.ndarray, Simulated asset variances
        """
        
        if not hasattr(x0, 'size'):
            x0 = np.array([x0] * N)
        if not hasattr(v0, 'size'):
            v0 = np.array([v0] * N)
        
        # Randomness
        Delta_W, Delta_B = BrownianIncrements()(t0, t1, 2, N, self.corr_matrix)
        Delta_t = t1 - t0
        
        # Full truncation scheme
        v0_plus = np.maximum(v0, 0) 
        sigma0 = np.sqrt(v0_plus)
        
        # Compute (x1, v1) from (x0, v0)
        v1 = v0_plus + self.kappa*(self.mu - v0_plus)*Delta_t + self.eta*sigma0*Delta_B
        x1 = x0 + self.r*x0*Delta_t + sigma0*x0*Delta_W
        
        return x1, v1
    
    
    def diffuse(self, T: float, x0: float, v0: float, N_samples: int = 1000000, pi: np.ndarray = None, N_steps: int = 100) -> tuple[MonteCarloPaths, MonteCarloPaths]:
        """
        Simulate asset price and variance paths using the Heston model.

        ---------------
        Arguments:
            T: float, maturity time
            x0: float, initial asset price
            v0: float, initial variance
            N_samples: int, optional (default = 1,000,000), number of samples
            pi: np.ndarray, optional (default = None), array of intermediate times
            N_steps: int, optional (default = 100), number of intermediate steps

        ---------------
        Returns:
            X: MonteCarloPaths, simulated asset prices paths
            V: MonteCarloPaths, simulated asset variances paths
        """
        X = np.zeros((N_steps, N_samples))
        V = np.zeros((N_steps, N_samples))
        if pi is None:
            pi = np.linspace(0, T, N_steps)
        N_steps = len(pi)
        
        X[0], V[0] = x0, v0
        for k in range(N_steps-1):
            X[k+1], V[k+1] = self.forward(t0=pi[k], t1=pi[k+1], N=N_samples, x0=X[k], v0=V[k])
            
            if self.verbose:
                print(k, end = ' ' if k < N_steps-2 else '\n')
        
        return MonteCarloPaths(X, pi), MonteCarloPaths(V, pi)
    #########################################################################
    
    ############################ OPTION PRICING #############################
    def priceCall(self, strike: float, maturity: float, x0: float = None, v0: float = None, X: MonteCarloPaths = None) -> float:
        """
        Compute the price of a European call option using the Heston model.

        ---------------
        Arguments:
            strike: float, strike price of the option
            maturity: float, maturity time of the option
            x0: float, optional (default = None), initial asset price
            v0: float, optional (default = None), initial variance
            X: MonteCarloPaths, optional (default = None), simulated asset prices and variances.
                If X is not passed, paths are simulated from (x0, v0).

        ---------------
        Returns:
            float: Price of the European call option
        """
        
        if X is None:
            X, _ = self.diffuse(T=maturity, x0=x0, v0=v0)

        call_price = np.mean(np.maximum(X(maturity) - strike, 0)) * np.exp(-self.r * maturity)
        
        return call_price
    
    
    def pricePut(self, strike: float, maturity: float, x0: float = None, v0: float = None, X: MonteCarloPaths = None) -> float:
        """
        Compute the price of a European put option using the Heston model.

        ---------------
        Arguments:
            strike: float, strike price of the option
            maturity: float, maturity time of the option
            x0: float, optional (default = None), initial asset price
            v0: float, optional (default = None), initial variance
            X: MonteCarloPaths, optional (default = None), simulated asset prices and variances.
                If X is not passed, paths are simulated from (x0, v0).

        ---------------
        Returns:
            float: Price of the European put option
        """
        
        if X is None:
            X, _ = self.diffuse(T=maturity, x0=x0, v0=v0)

        put_price = np.mean(np.maximum(strike - X(maturity), 0)) * np.exp(-self.r * maturity)
        
        return put_price
    
    
    def generateCallPutData(self, strikes: np.ndarray, maturities: np.ndarray, x0: float, v0: float) -> np.ndarray:
        """
        Generate call and put option prices using the Heston model.

        ---------------
        Arguments:
            strikes: np.ndarray, array of strike prices
            maturities: np.ndarray, array of maturity times
            x0: float, initial asset price
            v0: float, initial variance

        ---------------
        Returns:
            np.ndarray: Array of call and put option prices
        """
        
        X, V = self.diffuse(T=max(maturities), x0=x0, v0=v0, N_samples=10000000)
        
        prices = np.zeros((strikes.size, maturities.size, 2))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                prices[i, j, 0] = self.priceCall(strike=strike, maturity=maturity, x0=x0, v0=v0, X=X)
                prices[i, j, 1] = self.pricePut(strike=strike, maturity=maturity, x0=x0, v0=v0, X=X)
                
        
        return prices
    #########################################################################
########################################################################