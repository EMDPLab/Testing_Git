import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate_bubble(nozzle_radius=1.0, membrane_thickness=0.05, simulation_time=5.0, dt=0.01,
                    num_nodes=60, internal_pressure=0.5):
    """Simulate a 2D bubble as a closed membrane of mass-spring nodes.

    Parameters
    ----------
    nozzle_radius : float
        Radius of the nozzle where the bubble originates.
    membrane_thickness : float
        Thickness of the soap film. Higher thickness means stronger surface tension.
    simulation_time : float
        Total time of simulation (seconds).
    dt : float
        Time step for numerical integration (seconds).
    num_nodes : int
        Number of discrete nodes around the membrane.
    internal_pressure : float
        Pressure applied inside the bubble.

    Returns
    -------
    np.ndarray
        Positions of nodes over time with shape (steps, num_nodes, 2).
    """
    # Surface tension constant scaled by thickness
    base_k = 50.0
    spring_k = base_k * (membrane_thickness / 0.05)
    damping = 0.2 / (membrane_thickness / 0.05)

    # initial geometry
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * nozzle_radius
    velocities = np.zeros_like(positions)

    rest_length = 2 * np.pi * nozzle_radius / num_nodes
    mass = 1.0 / num_nodes
    steps = int(simulation_time / dt)
    history = np.zeros((steps, num_nodes, 2))

    for step in range(steps):
        # Forces initialized to zero
        forces = np.zeros_like(positions)

        # Spring forces between neighboring nodes
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            edge = positions[j] - positions[i]
            dist = np.linalg.norm(edge)
            if dist == 0:
                continue
            direction = edge / dist
            extension = dist - rest_length
            f = spring_k * extension * direction
            forces[i] += f
            forces[j] -= f

        # Internal pressure forces acting outward
        radius = np.linalg.norm(positions, axis=1)
        normals = positions / radius[:, None]
        pressure_force = internal_pressure * normals
        forces += pressure_force

        # Damping
        forces -= damping * velocities

        # Integrate
        velocities += (forces / mass) * dt
        positions += velocities * dt
        history[step] = positions

    return history


def animate_bubble(history, interval=30):
    num_frames, num_nodes, _ = history.shape
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    line, = ax.plot([], [], 'b-')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        data = history[frame]
        closed = np.vstack([data, data[0]])
        line.set_data(closed[:, 0], closed[:, 1])
        return line,

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                         interval=interval, blit=True)
    plt.show()


if __name__ == '__main__':
    hist = simulate_bubble(nozzle_radius=1.0, membrane_thickness=0.05,
                           simulation_time=5.0, dt=0.01, num_nodes=60,
                           internal_pressure=0.5)
    animate_bubble(hist)
