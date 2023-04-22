import numpy as np
from numba import jit
from tqdm import tqdm
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
def solve_navier_stokes(vxo, vyo, po, vxn, vyn, pn, dt, dx, dy, dx2, dy2, rho, nu, grid_size, ptol):    
    # Step 1: Solve pressure equation
    # http://www.thevisualroom.com/poisson_for_pressure.html
    last_diff = None
    while True:
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
        diff = np.linalg.norm(pn - po)
        if diff < ptol or (last_diff is not None and diff >= last_diff):
            break
        last_diff = diff

    # Step 2: Solve momentum equation without pressure gradient
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            vxn[i,j] = vxo[i,j] - dt*( ((vxo[i,j]*(vxo[i,j]-vxo[i-1,j]))/dx) + ((vyo[i,j]*(vxo[i,j]-vxo[i,j-1]))/dy) ) + (nu*dt*( (vxo[i+1,j] - 2*vxo[i,j] + vxo[i-1,j])/(dx**2) + (vxo[i,j+1] - 2*vxo[i,j] + vxo[i,j-1])/(dy**2) ))
            vyn[i,j] = vyo[i,j] - dt*( ((vxo[i,j]*(vyo[i,j]-vyo[i-1,j]))/dx) + ((vyo[i,j]*(vyo[i,j]-vyo[i,j-1]))/dy) ) + (nu*dt*( (vyo[i+1,j] - 2*vyo[i,j] + vyo[i-1,j])/(dx**2) + (vyo[i,j+1] - 2*vyo[i,j] + vyo[i,j-1])/(dy**2) ))
    
            # Step 3: Correct velocity field
            vxn[i,j] = vxn[i,j] - ((dt/rho) * ( (pn[i+1,j] - pn[i-1,j]) / (2*dx) ))
            vyn[i,j] = vyn[i,j] - ((dt/rho) * ( (pn[i,j+1] - pn[i,j-1]) / (2*dy) ))

    return vxn, vyn, pn

# 2. Main Loop
# simulation main loop
def main():
    ## 1. Define variables
    # Variables
    length_x = 1 # m
    length_y = 1 # m
    grid_size = 41
    density = 1.0 # kg/m^3
    cinematic_viscosity = 0.001 # m^2/s
    pressure_tolerance = 1e-5
    time_step = 0.001 # s
    grid_vx_old = np.zeros((grid_size, grid_size))
    grid_vy_old = np.zeros((grid_size, grid_size))
    grid_p_old = np.zeros((grid_size, grid_size))

    # Simplify variables
    dt = time_step
    dx = length_x / (grid_size - 1)
    dy = length_y / (grid_size - 1)
    dx2 = dx**2
    dy2 = dy**2
    rho = density
    nu = cinematic_viscosity
    ptol = pressure_tolerance
    vxo = grid_vx_old
    vyo = grid_vy_old
    po = grid_p_old
    vxn = vxo.copy()
    vyn = vyo.copy()
    pn = po.copy()

    t = 0
    print(f'dt: {dt:.5f}')
    
    # Plot initial conditions
    X, Y = np.meshgrid(np.linspace(0, length_x, grid_size), np.linspace(0, length_y, grid_size))	
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.xlabel('X')
    plt.ylabel('Y')

    i = 0
    last_diff = None
    total_pressure = []
    while True:
        vxo, vyo, po = update_boundaries(vxo, vyo, po)
        vxn, vyn, pn = solve_navier_stokes(vxo, vyo, po, vxn, vyn, pn, dt, dx, dy, dx2, dy2, rho, nu, grid_size, ptol)
        t = t + dt
        i += 1
        diff = np.linalg.norm(np.linalg.norm(vxn-vyn) - np.linalg.norm(vxo-vyo))
        total_pressure.append(np.sum(pn))
        if last_diff is not None and diff >= last_diff:
            print(f"diff: {diff:.5f}")
            break
        last_diff = diff
        vxo, vyo, po = vxn.copy(), vyn.copy(), pn.copy()

    vxo, vyo, po = update_boundaries(vxo, vyo, po)
    
    plt.suptitle(f"for dt = {dt:.5f} s", fontsize=16)
    ax1.set_title(f"t = {t:.2f} s, i = {i}")
    cf = ax1.contourf(X, Y, po.T, cmap="PuBuGn")
    ax1.streamplot(X, Y, vxo.T, vyo.T, color="black")
    cbar = fig.colorbar(cf, ax=ax1)
    ax2.set_title("Total pressure over iterations")
    ax2.plot(total_pressure)
    
    plt.subplots_adjust(top=0.85)
    plt.savefig("lid_cavity.png")

if __name__ == "__main__":
    main()