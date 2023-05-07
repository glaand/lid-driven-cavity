import pickle
import numpy as np
import matplotlib.pyplot as plt

# Ghia et al. (1982) - Re = 100
reference_vx_RE_100 = {
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
reference_vy_RE_100 = {
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

def post_processing(vars, filename):
    length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, vxo, vyo, po, ro, vxn, vyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs = vars
    vmo = np.sqrt(vxo**2 + vyo**2)
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
    axs[0,0].streamplot(X, Y, vxo.T, vyo.T, color="blue", density=1.5)
    axs[0,0].set_xlabel('$x$')
    axs[0,0].set_ylabel('$y$')

    # Second Plot
    axs[0,1].set_title("Velocity streamliens with color gradient")
    axs[0,1].streamplot(X, Y, vxo.T, vyo.T, color=vmo.T, density=10)

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
    #axs[4,0].axhline(y=max_tol, color='b', linestyle='--', label="Maximum tolerance")
    axs[4,0].set_xlabel('$iterations$')
    axs[4,0].set_ylabel('$|u_{n+1}-u_{n}|$')
    axs[4,0].legend()

    # Tenth Plot
    axs[4,1].set_title("Residual log plot of pressure")
    axs[4,1].loglog(p_diffs, label="Velocity diff")
    axs[4,1].axhline(y=min_tol, color='r', linestyle='--', label=f"Minimum tolerance = {min_tol}")
    #axs[4,1].axhline(y=max_tol, color='b', linestyle='--', label="Maximum Tolerance")
    axs[4,1].set_xlabel('$iterations$')
    axs[4,1].set_ylabel('$|p_{n+1}-p_{n}|$')
    axs[4,1].legend()

    # Eleventh Plot
    axs[5,0].set_title("X-velocity along Vertical Line through Geometric Center of Cavity")
    axs[5,0].plot(vxn[int(grid_size_x/2), :], label="X-velocity (André Glatzl)", color="orange")
    axs[5,0].scatter(reference_vx_RE_100.keys(), reference_vx_RE_100.values(), label="X-velocity (Ghia et al.)", color="blue")
    axs[5,0].set_xlabel('$x$')
    axs[5,0].set_ylabel('$Vx$')
    axs[5,0].legend()

    # Twelfth Plot
    axs[5,1].set_title("Y-velocity along Horizontal Line through Geometric Center of Cavity")
    axs[5,1].plot(vyn[:,int(grid_size_y/2)], label="Y-velocity (André Glatzl)", color="orange")
    axs[5,1].scatter(reference_vy_RE_100.keys(), reference_vy_RE_100.values(), label="Y-velocity (Ghia et al.)", color="blue")
    axs[5,1].set_xlabel('$y$')
    axs[5,1].set_ylabel('$Vy$')
    axs[5,1].legend()
    
    plt.savefig(f"{filename}.png")
    plt.show()

if __name__ == "__main__":
    with open("simulation.pickle", "rb") as f:
        vars = pickle.load(f)
    post_processing(vars, "simulation")