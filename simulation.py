import pickle
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from tqdm import tqdm
import multiprocessing
import time

MAX_ITERATIONS = int(30e6)

"""
2017 A. R. Malipeddi
A simple 2D geometric multigrid solver on Cartesian grids and unit square.
"""

import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, fastmath=True)
def relax(uxo, uyo, po, ro, pn, rn, dt, dx, dy, dx2, dy2, rho, grid_size_x, grid_size_y, iters=1):
    Ax=1.0/dx**2; Ay=1.0/dy**2
    Ap=1.0/(2.0*(Ax+Ay))
    Ap=1
    for _ in range(iters):
        uxo, uyo, po = update_boundaries(uxo, uyo, po)
        for i in range(1, grid_size_x - 1):
            for j in range(1, grid_size_y - 1):
                duxo = (uxo[i+1,j] - uxo[i-1,j]) / (2*dx)
                duyo = (uyo[i,j+1] - uyo[i,j-1]) / (2*dy)
                RHS_L = duxo**2
                RHS_C = 2*duxo*duyo
                RHS_R = duyo**2
                RHS = rho*((1/(dt)) * (duxo + duyo ) - RHS_L - RHS_C - RHS_R )
                pn[i,j] = Ap*((-1)*(dx2*dy2/(dx2 + dy2))*( RHS - (po[i+1,j]/2*dx2) - (po[i-1,j]/2*dx2) - (po[i,j+1]/2*dy2) - (po[i,j-1]/2*dy2) ))
                rn[i,j] = ro[i,j] + (pn[i,j] - po[i,j]) * dt
        uxo, uyo, pn = update_boundaries(uxo, uyo, pn)
    return pn, rn

@jit(nopython=True, nogil=True, fastmath=True)
def restrict(nx,ny,v):
  '''
  restrict 'v' to the coarser grid
  '''
  v_c=np.zeros((nx+2,ny+2))
  
  v_c[1:nx+1,1:ny+1]=0.25*(v[1:2*nx:2,1:2*ny:2]+v[1:2*nx:2,2:2*ny+1:2]+v[2:2*nx+1:2,1:2*ny:2]+v[2:2*nx+1:2,2:2*ny+1:2])

  return v_c

@jit(nopython=True, nogil=True, fastmath=True)
def prolong(nx,ny,v):
  '''
  interpolate 'v' to the fine grid
  '''
  v_f=np.zeros((2*nx+2,2*ny+2))

  a=0.5625; b=0.1875; c= 0.0625

  v_f[1:2*nx:2  ,1:2*ny:2  ] = a*v[1:nx+1,1:ny+1]+b*(v[0:nx  ,1:ny+1]+v[1:nx+1,0:ny]  )+c*v[0:nx  ,0:ny  ]
  v_f[2:2*nx+1:2,1:2*ny:2  ] = a*v[1:nx+1,1:ny+1]+b*(v[2:nx+2,1:ny+1]+v[1:nx+1,0:ny]  )+c*v[2:nx+2,0:ny  ]
  v_f[1:2*nx:2  ,2:2*ny+1:2] = a*v[1:nx+1,1:ny+1]+b*(v[0:nx  ,1:ny+1]+v[1:nx+1,2:ny+2])+c*v[0:nx  ,2:ny+2]
  v_f[2:2*nx+1:2,2:2*ny+1:2] = a*v[1:nx+1,1:ny+1]+b*(v[2:nx+2,1:ny+1]+v[1:nx+1,2:ny+2])+c*v[2:nx+2,2:ny+2]

  return v_f

@jit(nopython=True, nogil=True, fastmath=True)
def V_cycle(uxo, uyo, po, ro, pn, rn, dt, length_x, length_y, rho, grid_size_x, grid_size_y,num_levels,level=1):
  dx = length_x / (grid_size_x - 1)
  dy = length_y / (grid_size_y - 1)
  dx2 = dx**2
  dy2 = dy**2
  '''
  V cycle
  '''
  if(level==num_levels):#bottom solve
    u,res=relax(uxo, uyo, po, ro, pn, rn, dt, dx, dy, dx2, dy2, rho, grid_size_x, grid_size_y, iters=50)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=relax(uxo, uyo, po, ro, pn, rn, dt, dx, dy, dx2, dy2, rho, grid_size_x, grid_size_y, iters=2)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(grid_size_x//2,grid_size_y//2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(uxo, uyo, e_c, res_c, e_c, res_c, dt, length_x, length_y, rho, grid_size_x//2, grid_size_y//2,num_levels,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u+=prolong(grid_size_x//2,grid_size_y//2,e_c)
  
  #Step 5: Relax Au=f on this grid
  u,res=relax(uxo, uyo, po, ro, pn, rn, dt, dx, dy, dx2, dy2, rho, grid_size_x, grid_size_y, iters=1)
  return u,res

# 1. Functions
@jit(nopython=True, nogil=True, fastmath=True)
def update_boundaries(uxo, uyo, po):
    # Velocity
    uxo[0, :] = 0.0 # Left X (Dirichlet)
    uyo[0, :] = 0.0 # Left Y (Dirichlet)
    uxo[-1, :] = 0.0 # Right X (Dirichlet)
    uyo[-1, :] = 0.0 # Right Y (Dirichlet)
    uxo[:, 0] = 0.0 # Down X (Dirichlet)
    uyo[:, 0] = 0.0 # Down Y (Dirichlet)
    uxo[:, -1] = 1.0 # Up X (Dirichlet)
    uyo[:, -1] = 0.0 # Up Y (Dirichlet)
    # Pressure
    po[0, :] = po[1, :] # Left (Neumann)
    po[-1, :] = po[-2, :] # Right (Neumann)
    po[:, 0] = po[:, 1] # Down (Neumann)
    po[:, -1] = 0 # Up (Dirichlet)

    return uxo, uyo, po

@jit(nopython=True, nogil=True, fastmath=True)
def solve_navier_stokes(uxo, uyo, po, ro, uxn, uyn, pn, rn, dt, dx, dy, dx2, dy2, length_x, length_y, rho, nu, grid_size_x, grid_size_y, min_tol, max_tol):    
    # Step 1: Solve pressure equation
    # http://www.thevisualroom.com/poisson_for_pressure.html
    iterations = 0
    diff = np.inf
    diffs = []
    # Calculate pressure
    ##### JACOBI START #####
    # pn, rn = relax(uxo, uyo, po, ro, pn, rn, dt, dx, dy, dx2, dy2, rho, grid_size_x, grid_size_y, iters=50)
    ##### JACOBI END #####
    ##### OR #####
    ##### GEOMETRIC MULTIGRID START #####
    pn, rn = V_cycle(uxo, uyo, po, ro, pn, rn, dt, length_x, length_y, rho, grid_size_x, grid_size_y, int(np.log2(grid_size_x)))
    ##### GEOMETRIC MULTIGRID END #####
    norm_pn = np.linalg.norm(pn)
    norm_po = np.linalg.norm(po)
    diff = np.abs(norm_pn - norm_po)
    diffs.append(diff)
    po = pn.copy()
    iterations += 50
    diffs = np.average(diffs)

    # Step 2: Solve momentum equation without pressure gradient
    for i in range(1, grid_size_x - 1):
        for j in range(1, grid_size_y - 1):
            uxn[i,j] = uxo[i,j] - dt*( ((uxo[i,j]*(uxo[i+1,j]-uxo[i-1,j]))/(2*dx)) + ((uyo[i,j]*(uxo[i,j+1]-uxo[i,j-1]))/(2*dy)) ) + (nu*dt*( (uxo[i+1,j] - 2*uxo[i,j] + uxo[i-1,j])/(dx**2) + (uxo[i,j+1] - 2*uxo[i,j] + uxo[i,j-1])/(dy**2) ))
            uyn[i,j] = uyo[i,j] - dt*( ((uxo[i,j]*(uyo[i+1,j]-uyo[i-1,j]))/(2*dx)) + ((uyo[i,j]*(uyo[i,j+1]-uyo[i,j-1]))/(2*dy)) ) + (nu*dt*( (uyo[i+1,j] - 2*uyo[i,j] + uyo[i-1,j])/(dx**2) + (uyo[i,j+1] - 2*uyo[i,j] + uyo[i,j-1])/(dy**2) ))
    
            # Step 3: Correct velocity field
            uxn[i,j] = uxn[i,j] - ((dt/rho) * ( (pn[i+1,j] - pn[i-1,j]) / (2*dx) ))
            uyn[i,j] = uyn[i,j] - ((dt/rho) * ( (pn[i,j+1] - pn[i,j-1]) / (2*dy) ))

    return uxn, uyn, pn, rn, diffs

def save_results(length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, uxo, uyo, po, ro, uxn, uyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs):
    uxo, uyo, po = update_boundaries(uxo, uyo, po)
    vars = [length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, uxo, uyo, po, ro, uxn, uyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs]
    with open('simulation.pickle', 'wb') as f:
        pickle.dump(vars, f)

# 2. Main Loop
# simulation main loop
def run_simulation():
    ## 1. Define variables
    # Variables
    length_x = 1 # m
    length_y = 1 # m
    grid_size_x = 129
    grid_size_y = 129
    density = 1.0 # kg/m^3
    cinematic_viscosity = 0.01 # m^2/s
    min_tolerance = 1e-15
    max_tolerance = 1e15
    time_step = 0.00001 # s # Durch herausprobieren herausgefunden
    grid_ux_old = np.zeros((grid_size_x, grid_size_y))
    grid_uy_old = np.zeros((grid_size_x, grid_size_y))
    grid_p_old = np.zeros((grid_size_x, grid_size_y))
    grid_r_old = np.zeros((grid_size_x, grid_size_y)) # Residual for pressure equation since pressure is the problem

    # Simplify variables
    dt = time_step
    dx = length_x / (grid_size_x - 1)
    dy = length_y / (grid_size_y - 1)
    dx2 = dx**2
    dy2 = dy**2
    rho = density
    nu = cinematic_viscosity
    min_tol = min_tolerance
    max_tol = max_tolerance
    uxo = grid_ux_old
    uyo = grid_uy_old
    po = grid_p_old
    ro = grid_r_old
    uxn = uxo.copy()
    uyn = uyo.copy()
    pn = po.copy()
    rn = ro.copy()
    # Simulation loop
    t = 0
    diff = np.inf
    iterations = 0
    tot_p = []
    tot_v = []
    diffs = []
    p_diffs = []

    pbar = tqdm(range(MAX_ITERATIONS))
    last_time = time.time()
    pool = multiprocessing.Pool()
    for i in pbar:
        uxo, uyo, po = update_boundaries(uxo, uyo, po)
        uxn, uyn, pn, rn, p_diff = solve_navier_stokes(uxo, uyo, po, ro, uxn, uyn, pn, rn, dt, dx, dy, dx2, dy2, length_x, length_y, rho, nu, grid_size_x, grid_size_y, min_tol, max_tol)
        uxn, uyn, pn = update_boundaries(uxn, uyn, pn)
        tot_p.append(np.sum(pn))
        tot_v.append(np.sum(np.sqrt(uxn**2 + uyn**2)))
        norm_vn = np.linalg.norm(uxn-uyn)
        norm_vo = np.linalg.norm(uxo-uyo)
        diff = np.abs(norm_vn - norm_vo)
        # check if diff is not nan
        diffs.append(diff)
        p_diffs.append(p_diff)
        if not (norm_vn == 0 or norm_vo == 0):
            if diff <= min_tol or diff >= max_tol:
                break
        t = t + dt
        uxo, uyo, po = uxn.copy(), uyn.copy(), pn.copy()
        iterations += 1
        pbar.set_description(f"Pressure residual: {p_diff:.2E}, Velocity residual: {diff:.2E}")
        # check if 60 seconds have passed
        if time.time() - last_time >= 60:
            last_time = time.time()
            pool.apply_async(save_results, args=(length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, uxo, uyo, po, ro, uxn, uyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs))
    pool.close()

    if iterations == MAX_ITERATIONS:
        print("Warning: Velocity solver did not converge after MAX_ITERATIONS iterations")

    print("")
    print(f"Pressure residual: {p_diffs[-1]}")
    print(f"Velocity residual: {diffs[-1]}")

    save_results(length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, uxo, uyo, po, ro, uxn, uyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs)

if __name__ == "__main__": 
    run_simulation()