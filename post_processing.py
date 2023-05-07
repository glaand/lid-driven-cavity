import pickle
import numpy as np
import matplotlib.pyplot as plt

def post_processing(vars, filename):
    length_x, length_y, grid_size_x, grid_size_y, dt, dx, dy, dx2, dy2, rho, nu, min_tol, max_tol, vxo, vyo, po, ro, vxn, vyn, pn, rn, t, tot_p, tot_v, iterations, diffs, p_diffs = vars
    vmo = np.sqrt(vxo**2 + vyo**2)
    X, Y = np.meshgrid(np.linspace(0, length_x, grid_size_x), np.linspace(0, length_y, grid_size_y))	
    fig, axs = plt.subplots(5, 2, figsize=(16, 25))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.suptitle(f"for dt = {dt} s, dx = {dx} m, dy = {dy} m, t = {t} s, iter = {iterations}", fontsize=24)
    
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
    axs[1,0]=fig.add_subplot(5,2,3,projection='3d')
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
    axs[2,0]=fig.add_subplot(5,2,5,projection='3d')
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
    
    plt.savefig(f"{filename}.png")
    plt.show()

if __name__ == "__main__":
    # Simulation 1
    ##with open("simulation_v1.pickle", "rb") as f:
    #    vars1 = pickle.load(f)
    #post_processing(vars1, "simulation_v1")
    with open("simulation_v2.pickle", "rb") as f:
        vars2 = pickle.load(f)
    post_processing(vars2, "simulation_v2")