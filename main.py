import numpy as np
import matplotlib.pyplot as plt

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
    G = 6.6743e-11
    r = calc_distance(x1, y1, z1, x2, y2, z2)
    return -G * m1 * m2 / r

def calc_total_energy(m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return calc_energy(m1, m2, x1, y1, z1, x2, y2, z2) + calc_energy(m1, m3, x1, y1, z1, x3, y3, z3) + calc_energy(m2, m3, x2, y2, z2, x3, y3, z3)

def simulate(m1, m2, m3, error = 0.001, trust = 0.99):
    total = 0
    a = -G*(m1*m2/np.sqrt(6) + m2*m3/3 + m1*m3/3*np.sqrt(3))
    b = -G*(m1*m2+m2*m3+m1*m3/3)
    #print((np.abs(((b-a)**2)/(2*error*2)*np.log(2/(1-trust)))))
    num_sims = int(np.ceil(np.abs(((b-a)**2)/(2*error*2)*np.log(2/(1-trust)))))
    #print(num_sims)
    intermediate_values = np.zeros(num_sims)
    for i in range(num_sims):
        x1, y1, z1 = generate_body(xmin1, xmax1, ymin1, ymax1, zmin1, zmax1)
        x2, y2, z2 = generate_body(xmin2, xmax2, ymin2, ymax2, zmin2, zmax2)
        x3, y3, z3 = generate_body(xmin3, xmax3, ymin3, ymax3, zmin3, zmax3)
        s = calc_total_energy(m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3)
        total += s
        intermediate_values[i] = total / (i + 1)
    mean_value = total / num_sims
    print(f"Energia potentiala gravitationala medie: {mean_value:.4f}")
    draw_graph_simulation(intermediate_values, mean_value)
    return mean_value

def draw_graph_simulation(intermediate_values, mean_value):
    plt.figure(figsize=(10, 6))
    plt.plot(intermediate_values, label='Evoluția mediei', color='blue', linewidth=1)
    plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Energia medie: {mean_value:.4f}')
    plt.title("Evoluția calculului mediei energiei", fontsize=16)
    plt.xlabel("Numărul simulării", fontsize=12)
    plt.ylabel("Media intermediară", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

simulate(1000000, 1000, 100000)
