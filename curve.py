import numpy as np
import torch
import matplotlib.pyplot as plt

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def curve_length_r2(f1, f2, N):
    """
    Compute length of c = [f1 -> R, f2 -> R].T 
    f1, f2 parameterized by t in [0, 1]
    """
    all_t = np.linspace(0, 1, N)
    total_d = 0
    for i in range(len(all_t) - 1):
        t1 = all_t[i]
        t2 = all_t[i+1]

        v1 = np.array([f1(t1), f2(t1)])
        v2 = np.array([f1(t2), f2(t2)])
        d = euclidean_distance(v1, v2)
        total_d += d
    
    return total_d

def curve_length(c, N):
    all_t = np.linspace(0, 1, N)
    x = c(all_t)
    return np.sum(np.linalg.norm(np.diff(x), axis=0))

def integral_curve_length(d_c_t_mag, N):
    ts = np.linspace(0, 1, N)
    return np.sum(d_c_t_mag(ts)) / N

def f1(t):
    return 2*t + 1

def f2(t):
    return -t**2

def c(t):
    return np.array([f1(t), f2(t)])

def d_c_t_mag(t):
    return np.sqrt(4 + 4*t**2)

def compute_curve_energy(c, c_prime, G, dim, N=100):
    """
    c: R_1 -> R_n, a curve parameterized on t
    c_prime: R_1 -> R_n, the derivative of the curve
    G: R_n -> R_nxn, the metric function
    dim: the number of pieces in the model
    N: number of samples
    """
    total = torch.zeros(dim-1)
    for t in range(0, 1, N):
        c_t = c(t)
        G_t = G(c_t)
        c_prime_t = c_prime(t)
        result = torch.einsum('bj,bij,bj->b', c_prime_t,G_t,  c_prime_t)
        total += result

    total = torch.mean(total)
    
    return total

def G_assignment(x):
    """
    x: R_2
    returns: R_2x2
    """
    return torch.eye(2).expand(len(x),-1, -1) * (1 + torch.norm(x, dim=1) ** 2).unsqueeze(-1).unsqueeze(-1)


def piecewise_direct_minimization(x1, x2, G, N_pieces=100, steps=1000):
    """
    x1, x2: R_2
    G: R_2 -> R_2x2, a metric
    N_pieces: number of pieces to divide the curve into
    steps: number of optimization steps
    """
    v_sp = x2-x1 # shortest euclidean path
    ts = torch.linspace(0, 1, N_pieces).expand(2,-1).T * v_sp + x1
    ts_optim = torch.nn.Parameter(torch.clone(ts[1:-1]))

    opt = torch.optim.Adam([ts_optim], lr=.01)
    energies = []
    for _ in range(steps):
        all_ts = torch.cat([x1.unsqueeze(0), ts_optim, x2.unsqueeze(0)])

        t_dot = torch.diff(all_ts, dim=0)
        opt.zero_grad()
        c = lambda t: t_dot * t + all_ts[:-1]
        c_prime = lambda t: t_dot

        energy = compute_curve_energy(c, c_prime, G, N_pieces)
        energies.append(energy.item())
        print(energy.item())

        energy.backward()
        opt.step()

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(energies)
    
    ts_final = torch.cat([x1.unsqueeze(0), ts_optim, x2.unsqueeze(0)], dim=0)
    ax2.plot(ts_final.detach().numpy()[:,0], ts_final.detach().numpy()[:,1])
    plt.show()
    breakpoint()
     


def e6():
    x1 = torch.tensor([1, 0])
    x2 = torch.tensor([3, -1])
    piecewise_direct_minimization(x1, x2, G_assignment)


def e5():
    print(curve_length(c, 1000))
    print(curve_length_r2(f1, f2, 1000))
    print(integral_curve_length(d_c_t_mag, 10000))

if __name__=="__main__":
    e6()    