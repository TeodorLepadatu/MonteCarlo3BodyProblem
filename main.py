import numpy as np
from scipy.integrate import quad

def generate_body(xmin,xmax,ymin,ymax,zmin,zmax):
    x = np.random.uniform(xmin,xmax)
    y = np.random.uniform(ymin,ymax)
    z = np.random.uniform(zmin,zmax)
    return x,y,z

def calc_distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def calc_energy(m1,m2,x1,y1,z1,x2,y2,z2):
    G = 6.6743e-11
    r = calc_distance(x1,y1,z1,x2,y2,z2)
    return -G*m1*m2/r

def calc_total_energy(m1,m2,m3,x1,y1,z1,x2,y2,z2,x3,y3,z3):
    return calc_energy(m1,m2,x1,y1,z1,x2,y2,z2) + calc_energy(m1,m3,x1,y1,z1,x3,y3,z3) + calc_energy(m2,m3,x2,y2,z2,x3,y3,z3)

def simulate(xmin,xmax,ymin,ymax,zmin,zmax,m1,m2,m3,num_sims=10000):
    total = 0
    for _ in range(num_sims):
        x1,y1,z1 = generate_body(xmin,xmax,ymin,ymax,zmin,zmax)
        x2,y2,z2 = generate_body(xmin,xmax,ymin,ymax,zmin,zmax)
        x3,y3,z3 = generate_body(xmin,xmax,ymin,ymax,zmin,zmax)
        s = calc_total_energy(m1,m2,m3,x1,y1,z1,x2,y2,z2,x3,y3,z3)
        total += s
    return total/num_sims
'''
def calculate_real(m1,m2,m3,xmin1,xmax1,ymin1,ymax1,zmin1,zmax1,xmin2,xmax2,ymin2,ymax2,zmin2,zmax2,xmin3,xmax3,ymin3,ymax3,zmin3,zmax3):

if __name__ == '__main__':
'''