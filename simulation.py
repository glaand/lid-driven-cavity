import pickle
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from tqdm import tqdm

MAX_ITERATIONS = 100000

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
def solve_navier_stokes(vxo, vyo, po, ro, vxn, vyn, pn, rn, dt, dx, dy, dx2, dy2, rho, nu, grid_size_x, grid_size_y, min_tol, max_tol):    
    # Step 1: Solve pressure equation
    # http://www.thevisualroom.com/poisson_for_pressure.html
    iterations = 0
    diff = np.inf
    diffs = []
    for i in range(MAX_ITERATIONS):
        vxo, vyo, po = update_boundaries(vxo, vyo, po)
        # Calculate pressure
        for i in range(1, grid_size_x - 1):
            for j in range(1, grid_size_y - 1):
                dvxo = (vxo[i+1,j] - vxo[i-1,j]) / (2*dx)
                dvyo = (vyo[i,j+1] - vyo[i,j-1]) / (2*dy)
                RHS_L = dvxo**2
                RHS_C = 2*dvxo*dvyo
                RHS_R = dvyo**2
                RHS = rho*((1/(dt)) * (dvxo + dvyo ) - RHS_L - RHS_C - RHS_R )
                pn[i,j] = (-1)*(dx2*dy2/(dx2 + dy2))*( RHS - (po[i+1,j]/2*dx2) - (po[i-1,j]/2*dx2) - (po[i,j+1]/2*dy2) - (po[i,j-1]/2*dy2) )
                rn[i,j] = ro[i,j] + (pn[i,j] - po[i,j]) * dt
        vxn, vyn, pn = update_boundaries(vxn, vyn, pn)
        norm_pn = np.linalg.norm(pn)
        norm_po = np.linalg.norm(po)
        diff = np.abs(norm_pn - norm_po)
        diffs.append(diff)
        if not (norm_pn == 0 or norm_po == 0):
            if diff <= min_tol or diff >= max_tol:
                break
        po = pn.copy()
        iterations += 1
    diffs = np.average(diffs)

    # Step 2: Solve momentum equation without pressure gradient
    for i in range(1, grid_size_x - 1):
        for j in range(1, grid_size_y - 1):
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
    grid_size_x = 129
    grid_size_y = 129
    density = 1.0 # kg/m^3
    cinematic_viscosity = 0.01 # m^2/s
    min_tolerance = 1e-8
    max_tolerance = 1e8
    time_step = 0.000001 # s # Durch herausprobieren herausgefunden
    grid_vx_old = np.zeros((grid_size_x, grid_size_y))
    grid_vy_old = np.zeros((grid_size_x, grid_size_y))
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

    pbar = tqdm(range(MAX_ITERATIONS))
    for i in pbar:
        vxo, vyo, po = update_boundaries(vxo, vyo, po)
        vxn, vyn, pn, rn, p_diff = solve_navier_stokes(vxo, vyo, po, ro, vxn, vyn, pn, rn, dt, dx, dy, dx2, dy2, rho, nu, grid_size_x, grid_size_y, min_tol, max_tol)
        vxn, vyn, pn = update_boundaries(vxn, vyn, pn)
        tot_p.append(np.sum(pn))
        tot_v.append(np.sum(np.sqrt(vxn**2 + vyn**2)))
        norm_vn = np.linalg.norm(vxn-vyn)
        norm_vo = np.linalg.norm(vxo-vyo)
        diff = np.abs(norm_vn - norm_vo)
        # check if diff is not nan
        diffs.append(diff)
        p_diffs.append(p_diff)
        if not (norm_vn == 0 or norm_vo == 0):
            if diff <= min_tol or diff >= max_tol:
                break
        t = t + dt
        vxo, vyo, po = vxn.copy(), vyn.copy(), pn.copy()
        iterations += 1
        pbar.set_description(f"Pressure residual: {p_diff:.2E}, Velocity residual: {diff:.2E}")

    if iterations == MAX_ITERATIONS:
        print("Warning: Velocity solver did not converge after MAX_ITERATIONS iterations")

    print("")
    print(f"Pressure residual: {p_diffs[-1]}")
    print(f"Velocity residual: {diffs[-1]}")

    vxo, vyo, po = update_boundaries(vxo, vyo, po)
    vars = [length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, vxo, vyo, po, ro, vxn, vyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs]
    print(f"Total iterations: {iterations}")
    with open('simulation.pickle', 'wb') as f:
        pickle.dump(vars, f)

if __name__ == "__main__": 
    run_simulation()