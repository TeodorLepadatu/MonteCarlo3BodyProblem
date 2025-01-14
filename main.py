import numpy as np
import matplotlib.pyplot as plt
import torch
from torchquad import Simpson, set_up_backend

G = 6.6743e-11
xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = 0, 1, 0, 1, 0, 1
xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = 2, 3, 0, 1, 0, 1
xmin3, xmax3, ymin3, ymax3, zmin3, zmax3 = 4, 5, 0, 1, 0, 1


def generate_body(xmin, xmax, ymin, ymax, zmin, zmax):
    x = np.random.uniform(xmin, xmax)
    y = np.random.uniform(ymin, ymax)
    z = np.random.uniform(zmin, zmax)
    return x, y, z


def calc_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def calc_energy(m1, m2, x1, y1, z1, x2, y2, z2):
    r = calc_distance(x1, y1, z1, x2, y2, z2)
    return -G * m1 * m2 / r


def calc_total_energy(m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return calc_energy(m1, m2, x1, y1, z1, x2, y2, z2) + calc_energy(m1, m3, x1, y1, z1, x3, y3, z3) + calc_energy(m2, m3, x2, y2, z2, x3, y3, z3)

def simulate(m1, m2, m3, error=0.01, trust=0.95):
    total = 0
    a = -G * (m1 * m2 / np.sqrt(6) + m2 * m3 / 3 + m1 * m3 / 3 * np.sqrt(3))
    b = -G * (m1 * m2 + m2 * m3 + m1 * m3 / 3)
    num_sims = int(np.ceil(np.abs(((b - a) ** 2) / (2 * error ** 2) * np.log(2 / (1 - trust)))))
    print(f"Numar de simulari: {num_sims}")
    intermediate_values = np.zeros(num_sims)
    # pt a genera graficul punctelor intermediare
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for i in range(num_sims):
        x1, y1, z1 = generate_body(xmin1, xmax1, ymin1, ymax1, zmin1, zmax1)
        x2, y2, z2 = generate_body(xmin2, xmax2, ymin2, ymax2, zmin2, zmax2)
        x3, y3, z3 = generate_body(xmin3, xmax3, ymin3, ymax3, zmin3, zmax3)
        s = calc_total_energy(m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3)
        total += s
        intermediate_values[i] = total / (i + 1)
        # desenam distributia punctelor
    #     ax.cla()
    #     draw_cube(ax, xmin1, xmax1, ymin1, ymax1, zmin1, zmax1, color='red')
    #     draw_cube(ax, xmin2, xmax2, ymin2, ymax2, zmin2, zmax2, color='green')
    #     draw_cube(ax, xmin3, xmax3, ymin3, ymax3, zmin3, zmax3, color='blue')
    #
    #     ax.scatter(x1, y1, z1, c='r', label='Cub 1')
    #     ax.scatter(x2, y2, z2, c='g', label='Cub 2')
    #     ax.scatter(x3, y3, z3, c='b', label='Cub 3')
    #
    #     ax.set_xlim(-5, 5)
    #     ax.set_ylim(-5, 5)
    #     ax.set_zlim(-5, 5)
    #     ax.set_title(f"Simulare {i + 1}")
    #     ax.legend(loc='upper left')
    #     plt.pause(0.001)
    # plt.show()

    mean_value = total / num_sims
    print(f"Energia potentiala gravitationala medie (Monte Carlo): {mean_value:.4f}")
    # draw_graph_simulation(intermediate_values, mean_value)
    return mean_value, intermediate_values

def draw_cube(ax, xmin, xmax, ymin, ymax, zmin, zmax, color='gray'):

    vertices = [
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],  # baza inferioară
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]   # baza superioară
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0]]
        y = [vertices[edge[0]][1], vertices[edge[1]][1]]
        z = [vertices[edge[0]][2], vertices[edge[1]][2]]
        ax.plot(x, y, z, alpha = 0.4, c=color)

def draw_graph_simulation(intermediate_values, mean_value_emp, mean_value_int):
    plt.figure(figsize=(10, 6))
    plt.plot(intermediate_values, label='Evoluția mediei în simulări', color='blue', linewidth=1)
    plt.axhline(y=mean_value_emp, color='red', linestyle='--', label=f'Energia medie Monte Carlo: {mean_value_emp:.4f}')
    plt.axhline(y=mean_value_int, color='black', linestyle='--', label=f'Energia medie integrală: {mean_value_int:.4f}')
    plt.title("Evoluția calculului mediei energiei cu Monte Carlo", fontsize=16)
    plt.xlabel("Numărul simulării", fontsize=12)
    plt.ylabel("Media intermediară", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_up_backend("torch", data_type="float32")
m1, m2, m3 = 30 * 100000, 1000, 100000


def result():
    def integrand_pytorch1(x):
        x1, x2, y1, y2, z1, z2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        return -G * (m1 * m2 / torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))

    def integrand_pytorch2(x):
        x2, x3, y2, y3, z2, z3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        return -G * (m2 * m3 / torch.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2))

    def integrand_pytorch3(x):
        x1, x3, y1, y3, z1, z3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        return -G * (m1 * m3 / torch.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2))

    dimension = 6
    integration_domain1 = [[0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [0, 1]]
    integration_domain2 = [[2, 3], [0, 1], [0, 1], [4, 5], [0, 1], [0, 1]]
    integration_domain3 = [[0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [0, 1]]

    N = 10 ** dimension
    simp = Simpson()

    result_pytorch1 = simp.integrate(integrand_pytorch1, dim=dimension, N=N, integration_domain=integration_domain1)
    result_pytorch2 = simp.integrate(integrand_pytorch2, dim=dimension, N=N, integration_domain=integration_domain2)
    result_pytorch3 = simp.integrate(integrand_pytorch3, dim=dimension, N=N, integration_domain=integration_domain3)

    return result_pytorch1 + result_pytorch2 + result_pytorch3


# Simulare Monte Carlo
monte_carlo_result, intermediate_sim = simulate(30 * 100000, 1000, 100000)

# Calculul integralei
integral_result = result().item()
print(f"Energia potentiala gravitationala medie (Integrala): {integral_result:.4f}")

# Compararea rezultatelor
print(f"Diferența dintre rezultate: {abs(monte_carlo_result - integral_result):.4f}")

draw_graph_simulation(intermediate_sim, monte_carlo_result, integral_result)