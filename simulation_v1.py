import pickle
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as anime

# 1. Functions
@jit(nopython=True, nogil=True, fastmath=True)
def update_boundaries(vxo, vyo, po):
    # Velocity
    vxo[0, :] = 0.0 # Left X (Dirichlet)
    vyo[0, :] = 0.0 # Left Y (Dirichlet)
    vxo[-1, :] = 0.0 # Right X (Dirichlet)
    vyo[-1, :] = 0.0 # Right Y (Dirichlet)
    vxo[:, 0] = 0.0 # Down X (Dirichlet)
    vyo[:, 0] = 0.0 # Down Y (Dirichlet)
    vxo[:, -1] = 1.0 # Up X (Dirichlet)
    vyo[:, -1] = 0.0 # Up Y (Dirichlet)
    # Pressure
    po[0, :] = po[1, :] # Left (Neumann)
    po[-1, :] = po[-2, :] # Right (Neumann)
    po[:, 0] = po[:, 1] # Down (Neumann)
    po[:, -1] = 0 # Up (Dirichlet)

    return vxo, vyo, po

@jit(nopython=True, nogil=True, fastmath=True)
def solve_navier_stokes(vxo, vyo, po, ro, vxn, vyn, pn, rn, dt, dx, dy, dx2, dy2, rho, nu, grid_size, min_tol, max_tol):    
    # Step 1: Solve pressure equation
    # http://www.thevisualroom.com/poisson_for_pressure.html
    iterations = 0
    diff = np.inf
    MAX_LOOP = 100
    diffs = []
    for i in range(MAX_LOOP):
        vxo, vyo, po = update_boundaries(vxo, vyo, po)
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                dvxo = (vxo[i+1,j] - vxo[i-1,j]) / (2*dx)
                dvyo = (vyo[i,j+1] - vyo[i,j-1]) / (2*dy)
                RHS_L = dvxo**2
                RHS_C = 2*dvxo*dvyo
                RHS_R = dvyo**2
                RHS = rho*((1/(dt)) * (dvxo + dvyo ) - RHS_L - RHS_C - RHS_R )
                pn[i,j] = (-1)*(dx2*dy2/(dx2 + dy2))*( RHS - (po[i+1,j]/2*dx2) - (po[i-1,j]/2*dx2) - (po[i,j+1]/2*dy2) - (po[i,j-1]/2*dy2) )
                rn[i,j] = ro[i,j] + (pn[i,j] - po[i,j]) * dt
        diff = np.linalg.norm(pn - po)
        diffs.append(diff)
        if diff <= min_tol or diff >= max_tol:
            break
        iterations += 1
    diffs = np.average(diffs)

    # Step 2: Solve momentum equation without pressure gradient
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            vxn[i,j] = vxo[i,j] - dt*( ((vxo[i,j]*(vxo[i,j]-vxo[i-1,j]))/dx) + ((vyo[i,j]*(vxo[i,j]-vxo[i,j-1]))/dy) ) + (nu*dt*( (vxo[i+1,j] - 2*vxo[i,j] + vxo[i-1,j])/(dx**2) + (vxo[i,j+1] - 2*vxo[i,j] + vxo[i,j-1])/(dy**2) ))
            vyn[i,j] = vyo[i,j] - dt*( ((vxo[i,j]*(vyo[i,j]-vyo[i-1,j]))/dx) + ((vyo[i,j]*(vyo[i,j]-vyo[i,j-1]))/dy) ) + (nu*dt*( (vyo[i+1,j] - 2*vyo[i,j] + vyo[i-1,j])/(dx**2) + (vyo[i,j+1] - 2*vyo[i,j] + vyo[i,j-1])/(dy**2) ))
    
            # Step 3: Correct velocity field
            vxn[i,j] = vxn[i,j] - ((dt/rho) * ( (pn[i+1,j] - pn[i-1,j]) / (2*dx) ))
            vyn[i,j] = vyn[i,j] - ((dt/rho) * ( (pn[i,j+1] - pn[i,j-1]) / (2*dy) ))

    return vxn, vyn, pn, rn, diffs

# 2. Main Loop
# simulation main loop
def run_simulation():
    ## 1. Define variables
    # Variables
    length_x = 1 # m
    length_y = 1 # m
    grid_size = 41
    density = 1.0 # kg/m^3
    cinematic_viscosity = 0.001 # m^2/s
    min_tolerance = 1e-16
    max_tolerance = 1e6
    time_step = 0.05 # s
    grid_vx_old = np.zeros((grid_size, grid_size))
    grid_vy_old = np.zeros((grid_size, grid_size))
    grid_p_old = np.zeros((grid_size, grid_size))
    grid_r_old = np.zeros((grid_size, grid_size)) # Residual for pressure equation since pressure is the problem

    # Simplify variables
    dt = time_step
    dx = length_x / (grid_size - 1)
    dy = length_y / (grid_size - 1)
    dx2 = dx**2
    dy2 = dy**2
    rho = density
    nu = cinematic_viscosity
    min_tol = min_tolerance
    max_tol = max_tolerance
    vxo = grid_vx_old
    vyo = grid_vy_old
    po = grid_p_old
    ro = grid_r_old
    vxn = vxo.copy()
    vyn = vyo.copy()
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

    MAX_LOOP = 100
    for i in range(MAX_LOOP):
        vxo, vyo, po = update_boundaries(vxo, vyo, po)
        vxn, vyn, pn, rn, p_diff = solve_navier_stokes(vxo, vyo, po, ro, vxn, vyn, pn, rn, dt, dx, dy, dx2, dy2, rho, nu, grid_size, min_tol, max_tol)
        tot_p.append(np.sum(pn))
        tot_v.append(np.sum(np.sqrt(vxn**2 + vyn**2)))
        diff = np.linalg.norm(np.linalg.norm(vxn-vyn) - np.linalg.norm(vxo-vyo))
        # check if diff is not nan
        diffs.append(diff)
        p_diffs.append(p_diff)
        if diff <= min_tol or diff >= max_tol:
            break
        vxo, vyo, po = vxn.copy(), vyn.copy(), pn.copy()
        t = t + dt
        iterations += 1

    if iterations == MAX_LOOP:
        print("Warning: Velocity solver did not converge after MAX_LOOP iterations")

    vxo, vyo, po = update_boundaries(vxo, vyo, po)
    vars = [length_x, length_y, grid_size, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, vxo, vyo, po, ro, vxn, vyn, pn, rn, t, tot_p, tot_v ,iterations, diffs, p_diffs]
    with open('simulation_v1.pickle', 'wb') as f:
        pickle.dump(vars, f)

if __name__ == "__main__":
    run_simulation()