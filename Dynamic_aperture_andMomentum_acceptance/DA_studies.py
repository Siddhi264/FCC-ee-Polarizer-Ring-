import os
#import h5py
import yaml
import json
import time
import copy
import glob
import shutil

import numpy as np
import math
import re
import pandas as pd
#import nafflib as pnf
import xcoll as xc
import xpart as xp
import xtrack as xt
import xfields as xf

from tqdm import tqdm
from pathlib import Path
import scipy.constants as sp_co
from scipy.linalg import cholesky
from scipy.stats import uniform, truncnorm, gamma
import scipy.optimize as opt
from multiprocessing import Pool
from contextlib import redirect_stdout, redirect_stderr, contextmanager
#from pylhc_submitter.job_submitter import main as htcondor_submit
import matplotlib.pyplot as plt

def inital_conditions_grid (study, energy_spread=None, ini_cond_type=None, min_r_y=None, max_r_y=None, num_r_y_points=None, min_theta_x=None, max_theta_x=None, num_theta_x_points=None, r_range_x=(5,10), r_range_y=(5,10), theta_range_x=(0,2*np.pi), theta_range_y=(0,2*np.pi), delta_initial_values=None, num_particles=None, rnd_seed=101):

    """
    Generate initial conditions grid for a given study.

    Parameters
    ----------
    study : str
        Study type. Can be 'DA', 'MA' or 'circulating_halo'.
    energy_spread : float
        Energy spread of the beam.
    ini_cond_type : str
        Type of initial condition for 'DA' and 'MA' studies. Can be 'cartesian' or 'polar'.
    min_r_y : float
        Minimum y for cartesian initial conditions or minimum radius for polar.
    max_r_y : float
        Maximum y for cartesian initial conditions or maximum radius for polar.
    num_r_y_points : int
        Number of points in y or radial plane.
    min_theta_x : float
        Minimum x for cartesian initial conditions or minimum theta for polar.
    max_theta_x : float
        Maximum x for cartesian initial conditions or minimum theta for polar.
    num_theta_x_points : int
        Number of points in x or theta plane.
    r_range_x : tuple
        Range of radial distances in phase space for horizontal coordinates for halo.
    r_range_y : tuple
        Range of radial distances in phase space for vertical coordinates for halo.
    theta_range_x : tuple
        Range of thata angles in phase space for horizontal coordinates for halo.
    theta_range_y : tuple
        Range of thata angles in phase space for vertical coordinates for halo.
    delta_initial_values : array_like
        Initial values of delta.
    num_particles : int
        Number of particles needed only for halo distribution.
    rnd_seed : int
        Random number seed.

    Returns for 'MA' and 'DA'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    delta_init : array_like
        Initial values of delta.
    num_theta_x_points : int
        Number of points in x plane.
    num_r_y_points : int
        Number of points in y plane.
    num_delta : int
        Number of initial values of delta.
    num_particles : int
        Number of particles.

    Returns for 'circulating_halo'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    px_normalized : array_like
        Normalized px coordinates of the particles.
    py_normalized : array_like
        Normalized py coordinates of the particles.
    """

    np.random.seed(rnd_seed)

    if min_r_y is None:
        if study in ['DA','MA']:
            min_r_y = 0
    
    if max_r_y is None:
        if study=='DA':
            max_r_y = 50
        elif study=='MA':
            max_r_y = 20

    if num_r_y_points is None:
        if study=='DA':
            num_r_y_points = 51
        elif study=='MA':
            num_r_y_points = 31

    if min_theta_x is None:
        if study=='DA':
            min_theta_x = -20
        elif study=='MA':
            min_theta_x = np.pi/4

    if max_theta_x is None:
        if study=='DA':
            max_theta_x= 20
        elif study=='MA':
            max_theta_x= np.pi/4

    if num_theta_x_points is None:
        if study=='DA':
            num_theta_x_points = 71
        elif study=='MA':
            num_theta_x_points = 1

    if delta_initial_values is None:
        if study=='DA':
            delta_initial_values = 0
        elif study=='MA':
            delta_initial_values = np.linspace(-45*energy_spread, 45*energy_spread, 51) 

    
    if study=='DA':
        if ini_cond_type is None or ini_cond_type=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

        elif ini_cond_type=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
    if study=='MA':
        if ini_cond_type is None or ini_cond_type=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
        elif ini_cond_type=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

    if study in ['DA','MA']:
        num_delta = np.size(delta_initial_values)
        num_particles = num_delta*num_theta_x_points*num_r_y_points
        if num_delta != 1:
            x_normalized = np.tile(x_normalized, num_delta)
            y_normalized = np.tile(y_normalized, num_delta)
            delta_init = np.repeat(delta_initial_values, np.size(x_normalized)/num_delta)
        else:
            delta_init = np.full(len(x_normalized), delta_initial_values)

    if study == 'circulating.halo':
        (x_normalized, px_normalized, r_points, theta_points)= xp.generate_2D_uniform_circular_sector(
                                            num_particles=num_particles,
                                            r_range=r_range_x, # beam sigmas
                                            theta_range=theta_range_x
                                            )   

        (y_normalized, py_normalized, r_points, theta_points)= xp.generate_2D_uniform_circular_sector(
                                            num_particles=num_particles,
                                            r_range=r_range_y, # beam sigmas
                                            theta_range=theta_range_y
                                            )
        return (x_normalized, y_normalized, px_normalized, py_normalized)
    
    return (x_normalized, y_normalized, delta_init, num_theta_x_points, num_r_y_points, num_delta, num_particles)





import matplotlib.colors as mcolors
def DA_vs_turns(particles, num_r_steps, num_theta_steps, x_norm, y_norm, delta_initial, delta_plots=True):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    if delta_plots and np.size(delta_initial) > 1:

        for ii in np.unique(delta_initial):
            delta_index = np.where(delta_initial==ii)[0]
            
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
                  
            x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
            y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
            part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
                        
            x_DA = np.full(num_theta_steps, np.nan)
            y_DA = np.full(num_theta_steps, np.nan)
            for jj in range(num_theta_steps):
                for kk in range(num_r_steps):
                    if part_at_turn_2d[kk,jj] != max_turns:
                        x_DA[jj] = x_norm_2d[kk,jj]
                        y_DA[jj] = y_norm_2d[kk,jj]
                        break

            min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
            where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
            
            # Plot DA using scatter and pcolormesh
            fig = plt.subplots()
            plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.3f'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.3f$\sigma$'%(min_DA))
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            cb = plt.colorbar()
            cb.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')

            fig = plt.subplots()
            plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.3f'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.3f$\sigma$'%(min_DA))    
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            ax = plt.colorbar()
            ax.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')
    
    else:

        if not delta_plots and np.size(delta_initial) > 1:
            closest_to_zero_delta = delta_initial[(np.abs(delta_initial - 0)).argmin()]
            delta_index = np.where(delta_initial==closest_to_zero_delta)[0]
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
        else:
            x_norm_1d = x_norm
            y_norm_1d = y_norm      
            part_at_turn_1d = part_at_turn

        x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
        y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
        part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
        x_DA = np.full(num_theta_steps, np.nan)
        y_DA = np.full(num_theta_steps, np.nan)
        for jj in range(num_theta_steps):
            for kk in range(num_r_steps):
                if part_at_turn_2d[kk,jj] != max_turns:
                    x_DA[jj] = x_norm_2d[kk,jj]
                    y_DA[jj] = y_norm_2d[kk,jj]
                    break

        min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
        where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
        
        # Plot DA using scatter and pcolormesh
        fig = plt.subplots()
        plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.3f$\sigma$'%(min_DA))
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        cb = plt.colorbar()
        cb.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')

        fig = plt.subplots()
        plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
        plt.plot(x_DA, y_DA, '-', color='r', label='DA')
        plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.3f$\sigma$'%(min_DA))    
        plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        ax = plt.colorbar()
        ax.set_label('Lost at turn')
        plt.legend(fontsize='small', loc='best')

    return (x_DA, y_DA, where_min_DA)


def MA_vs_turns(particles, num_r_steps, num_delta_steps, x_norm, y_norm, delta_initial):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    theta = np.max(np.unique(np.arctan2(y_norm,x_norm)))

    x_norm_2d = x_norm.reshape(num_delta_steps, num_r_steps)
    y_norm_2d = y_norm.reshape(num_delta_steps, num_r_steps)
    delta_norm_2d = delta_initial.reshape(num_delta_steps, num_r_steps)
    part_at_turn = part_at_turn
    part_at_turn_2d = part_at_turn.reshape(num_delta_steps, num_r_steps)
    x_MA = np.full(num_delta_steps, np.nan)
    y_MA = np.full(num_delta_steps, np.nan)
    delta_MA = np.full(num_delta_steps, np.nan)
    for jj in range(num_delta_steps):
        for ii in range(num_r_steps):
            if part_at_turn_2d[jj,ii] != max_turns:
                x_MA[jj] = x_norm_2d[jj,ii]
                y_MA[jj] = y_norm_2d[jj,ii]
                delta_MA[jj] = delta_norm_2d[jj,ii]
                break

    min_MA = np.nanmin(np.round(np.sqrt(x_MA**2+y_MA**2),1)) 
    where_min_MA = np.where(np.round(np.sqrt(x_MA**2+y_MA**2),1) == min_MA)[0]
    
    # Plot MA using scatter and pcolormesh
    fig = plt.subplots()
    plt.scatter(delta_initial*100, x_norm, c=part_at_turn)
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.3f$\sigma$'%(min_MA))
    plt.xlabel(r'$\delta$ [%]')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]'%(theta*180/np.pi))
    cb = plt.colorbar()
    cb.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')

    fig = plt.subplots()
    plt.pcolormesh(delta_norm_2d*100, x_norm_2d, part_at_turn_2d, shading='gouraud')
    plt.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    plt.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r', label='MA$_{min}$=%.3f$\sigma$'%(min_MA))    
    plt.xlabel(r'$\delta$ [%]')
    plt.ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]'%(theta*180/np.pi))
    ax = plt.colorbar()
    ax.set_label('Lost at turn')
    plt.legend(fontsize='small', loc='best')

    return (x_MA, delta_MA, where_min_MA)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap