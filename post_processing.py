#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 André Glatzl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import optparse
import imageio
import shutil
import h5py
import glob
import sys
import os

# Ghia et al. (1982) - Re = 100
reference_ux_RE_100 = {
    128: 1.00000,
    125: 0.84123,
    124: 0.78871,
    123: 0.73722,
    122: 0.68717,
    109: 0.23151,
    94: 0.00332,
    79: -0.13641,
    64: -0.20581,
    58: -0.21090,
    36: -0.15662,
    22: -0.10150,
    13: -0.06434,
    9: -0.04775,
    8: -0.04192,
    7: -0.03717,
    0: 0.00000
}

# Ghia et al. (1982) - Re = 100
reference_uy_RE_100 = {
    128: 0.00000,
    124: -0.05906,
    123: -0.07391,
    122: -0.08864,
    121: -0.10313,
    116: -0.16914,
    110: -0.22445,
    103: -0.24533,
    64: 0.05454,
    30: 0.17527,
    29: 0.17507,
    20: 0.16077,
    12: 0.12317,
    10: 0.10890,
    9: 0.10091,
    8: 0.09233,
    0: 0.00000
}

def post_processing(vars, output_file):
    length_x = vars['length_x']
    length_y = vars['length_y']
    grid_size_x = vars['grid_size_x']
    grid_size_y = vars['grid_size_y']
    dt = vars['dt']
    dx = vars['dx']
    dy = vars['dy']
    dx2 = vars['dx2']
    dy2 = vars['dy2']
    rho = vars['rho']
    nu = vars['nu']
    min_tol = vars['min_tol']
    max_tol = vars['max_tol']
    uxo = vars['uxo']
    uyo = vars['uyo']
    po = vars['po']
    ro = vars['ro']
    uxn = vars['uxn']
    uyn = vars['uyn']
    pn = vars['pn']
    rn = vars['rn']
    t = vars['t']
    tot_p = vars['tot_p']
    tot_v = vars['tot_v']
    iterations = vars['iterations']
    diffs = vars['diffs']
    p_diffs = vars['p_diffs']

    vmo = np.sqrt(uxo**2 + uyo**2)
    X, Y = np.meshgrid(np.linspace(0, length_x, grid_size_x), np.linspace(0, length_y, grid_size_y))	
    fig, axs = plt.subplots(6, 2, figsize=(14, 30))
    fig.tight_layout(pad=5.0)
    plt.subplots_adjust(top=0.9)
    plt.xlabel('X')
    plt.ylabel('Y')
    textlist = [
        f"dt = {dt} s, t = {t} s",
        f"dx = {dx} m, dy = {dy} m",
        f"Grid size = {grid_size_x} x {grid_size_y}",
        f"iter = {iterations}",
    ]
    plt.suptitle("\n".join(textlist), fontsize=24)
    
    # First Plot
    axs[0,0].set_title(f"Velocity streamlines")
    axs[0,0].streamplot(X, Y, uxo.T, uyo.T, color="blue", density=1.5)
    axs[0,0].set_xlabel('$x$')
    axs[0,0].set_ylabel('$y$')

    # Second Plot
    axs[0,1].set_title("Velocity streamliens with color gradient")
    axs[0,1].streamplot(X, Y, uxo.T, uyo.T, color=vmo.T, density=10)

    # Third Plot
    axs[1,0].remove()
    axs[1,0]=fig.add_subplot(6,2,3,projection='3d')
    plot10 = axs[1,0].plot_surface(X,Y, po.T, cmap='plasma', rstride=1, cstride=1)
    axs[1,0].set_xlabel('$x$')
    axs[1,0].set_zlabel('$p$')
    axs[1,0].set_ylabel('$y$');
    axs[1,0].text2D(0.35, 0.95, "Pressure in 3D", transform=axs[1,0].transAxes);
    fig.colorbar(plot10, ax=axs[1,0])

    # Fourth Plot
    axs[1,1].set_title("Pressure contour")
    plot11 = axs[1,1].contourf(po.T, origin='lower', cmap='plasma', levels=20)
    fig.colorbar(plot11, ax=axs[1,1])

    # Fifth Plot
    axs[2,0].remove()
    axs[2,0]=fig.add_subplot(6,2,5,projection='3d')
    plot20 = axs[2,0].plot_surface(X,Y, vmo.T, cmap='plasma', rstride=1, cstride=1)
    axs[2,0].set_xlabel('$x$')
    axs[2,0].set_zlabel('$|u|$')
    axs[2,0].set_ylabel('$y$');
    axs[2,0].text2D(0.35, 0.95, "Velocity magnitude 3D", transform=axs[2,0].transAxes);
    fig.colorbar(plot20, ax=axs[2,0])

    # Sixth Plot
    axs[2,1].set_title("Velocity magnitude contour")
    plot21 = axs[2,1].contourf(vmo.T, origin='lower', cmap='plasma', levels=20)
    fig.colorbar(plot21, ax=axs[2,1])

    # Seventh Plot
    axs[3,0].set_title("Total pressure over iterations")
    axs[3,0].plot(tot_p, label="Total pressure")
    axs[3,0].axhline(y=0, color='r', linestyle='--', label="Zero pressure")
    axs[3,0].set_xlabel('$iterations$')
    axs[3,0].set_ylabel('$sum(p)$')
    axs[3,0].legend()

    # Eighth Plot
    axs[3,1].set_title("Total velocity over iterations")
    axs[3,1].plot(tot_v, label="Total velocity")
    axs[3,1].axhline(y=0, color='r', linestyle='--', label="Zero velocity")
    axs[3,1].set_xlabel('$iterations$')
    axs[3,1].set_ylabel('$sum(v)$')
    axs[3,1].legend()

    # Ninth Plot
    axs[4,0].set_title("Residual log plot of velocity magnitude")
    axs[4,0].loglog(diffs, label="Velocity diff")
    axs[4,0].axhline(y=min_tol, color='r', linestyle='--', label=f"Minimum tolerance = {min_tol}")
    axs[4,0].set_xlabel('$iterations$')
    axs[4,0].set_ylabel('$|u_{n+1}-u_{n}|$')
    axs[4,0].legend()

    # Tenth Plot
    axs[4,1].set_title("Residual log plot of pressure")
    axs[4,1].loglog(p_diffs, label="Velocity diff")
    axs[4,1].axhline(y=min_tol, color='r', linestyle='--', label=f"Minimum tolerance = {min_tol}")
    axs[4,1].set_xlabel('$iterations$')
    axs[4,1].set_ylabel('$|p_{n+1}-p_{n}|$')
    axs[4,1].legend()

    # Eleventh Plot
    axs[5,0].set_title("X-velocity along Vertical Line through Geometric Center of Cavity")
    axs[5,0].plot(uxn[int(grid_size_x/2), :], label="X-velocity (André Glatzl)", color="orange")
    axs[5,0].scatter(reference_ux_RE_100.keys(), reference_ux_RE_100.values(), label="X-velocity (Ghia et al.)", color="blue")
    axs[5,0].set_xlabel('$x$')
    axs[5,0].set_ylabel('$Vx$')
    axs[5,0].legend()

    # Twelfth Plot
    axs[5,1].set_title("Y-velocity along Horizontal Line through Geometric Center of Cavity")
    axs[5,1].plot(uyn[:,int(grid_size_y/2)], label="Y-velocity (André Glatzl)", color="orange")
    axs[5,1].scatter(reference_uy_RE_100.keys(), reference_uy_RE_100.values(), label="Y-velocity (Ghia et al.)", color="blue")
    axs[5,1].set_xlabel('$y$')
    axs[5,1].set_ylabel('$Vy$')
    axs[5,1].legend()
    
    plt.savefig(f"{output_file}")

def prepare_post_processing(args):
    timestep, virtual_timestep, output_dir = args
    
    with h5py.File(options.input, 'r') as f:
        datasets = np.array(f[f"timestep_{timestep}"])
        vars = {}
        for i in range(len(datasets)):
            dataset = f[f"timestep_{timestep}"][datasets[i]]
            vars[datasets[i]] = dataset[()].astype(dataset.dtype).reshape(dataset.shape)

    post_processing(vars, f"{output_dir}/{virtual_timestep}.jpg")

def save_last_timestep(timestep, filename):
    with h5py.File(options.input, 'r') as f:
        datasets = np.array(f[f"timestep_{timestep}"])
        vars = {}
        for i in range(len(datasets)):
            dataset = f[f"timestep_{timestep}"][datasets[i]]
            vars[datasets[i]] = dataset[()].astype(dataset.dtype).reshape(dataset.shape)
    
    post_processing(vars, filename)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input', dest='input', help='Input file name [default: %default]', default="simulation.h5")
    parser.add_option('-l', '--last-timestep-file', dest='last_timestep_file', help='Last timestep file name with extension [required]') 
    parser.add_option('-o', '--output', dest='output', help='Video Output file name with extension [required]')
    (options, args) = parser.parse_args()
    required = ['last_timestep_file', 'output']
    for r in required:
        if options.__dict__[r] is None:
            parser.print_help()
            sys.exit(1)

    output_dir = 'output_img'
    # delete folder if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with h5py.File(options.input, 'r') as f:
        timesteps = sorted(list(map(lambda x: int(x.replace('timestep_', '')), list(f.keys()))))

    if len(timesteps) > 300:
        # split timestemps in 300 chunks
        chunked_timesteps = np.array_split(timesteps, 300)
        # select last timestep of each chunk
        timesteps = list(map(lambda x: x[-1], chunked_timesteps))
    virtual_timesteps = list(range(len(timesteps)))

    args = list(map(lambda x: (timesteps[x], x, output_dir), virtual_timesteps))

    with mp.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(prepare_post_processing, args), total=len(timesteps)):
            pass
    
    output_files = list(map(lambda x: f"{output_dir}/{x}.jpg", virtual_timesteps))
    os.system(f"ffmpeg -framerate 30 -i {output_dir}/%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {options.output}")

    save_last_timestep(timesteps[-1], options.last_timestep_file)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)