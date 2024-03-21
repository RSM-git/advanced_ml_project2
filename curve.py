import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence as KL
from tqdm import tqdm
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

def piecewise_curve_length(curve_points):
    diff = torch.diff(curve_points,dim=0)
    l = diff.norm(dim=1).sum()
    return l

def point_in_segment(x1, x2, x3):
    return x1 <= max(x2, x3) and x1 >= min(x2, x3)

def distance_along_piecewise(p1, p2, points):
    distance = 0
    x1, y1 = p1
    x2, y2 = p2
    
    segment_distances = torch.norm(torch.diff(points, dim=0), dim=1)
    # Iterate through each pair of consecutive points
    for i in range(len(points) - 1):
        xi, yi = points[i]
        xi1, yi1 = points[i + 1]

        x1_in_segment = point_in_segment(x1, xi, xi1)
        x2_in_segment = point_in_segment(x2, xi, xi1)
        
                
        if x1_in_segment and x2_in_segment:
            distance += torch.norm(p1-p2)
        elif x1_in_segment and not x2_in_segment:
            distance += torch.norm(xi1 - x1) 
        elif x2_in_segment and not x1_in_segment:
            distance += torch.norm(x2 - xi) 
        elif x1 < xi and x2 > xi1:
            distance += segment_distances[i]
    
    return distance

def evaluate_piecewise_curve(x, curve_points, eps=1e-5):
    for i in range(len(curve_points) - 1):
        lower =  min(curve_points[i][0], curve_points[i + 1][0])
        upper =  max(curve_points[i][0], curve_points[i + 1][0])
        if x >= lower - eps and x <= upper + eps:
            slope = (curve_points[i+1][1] - curve_points[i][1]) / (curve_points[i+1][0] - curve_points[i][0])
            intercept = curve_points[i][1] - slope * curve_points[i][0]
            return slope * x + intercept
    
    raise Exception('x out of bounds of curve')

def compute_curve_energy_FR(curve_points, decoder, device='cuda'):
    """
    Compute the fisher rao curve energy (i.e., integral of KL divergence along the curve)
    QUESTION: do we need to scale by each curve length for the integral???
    """
    #energy = 0
    # for i in range(len(curve_points) - 1):
    #     kl = KL(decoder(curve_points[i]), decoder(curve_points[i+1]))
    #     energy += kl.item() * curve_distances[i]

    kl = KL(decoder(curve_points[1:]), decoder(curve_points[:-1]))
    energy = kl.sum()

    return energy

## Should KL be scaled by distance?
## Should KL be per point, or across all points and all points shifted left?
## Why are the contours different


def compute_curve_energy_G(c, c_prime, G, n_pieces, N=50):
    """
    Compute the curve energy given a metric function defined by G
    c: R_1 -> R_n, a curve parameterized on t
    c_prime: R_1 -> R_n, the derivative of the curve
    G: R_n -> R_nxn, the metric function
    n_pieces: the number of pieces in the model
    N: number of samples
    """
    total = torch.zeros(n_pieces-1)
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

def linear_piecewise_function(points):
    """
    Returns a function f(t) that parameterizes the piecewise curve defined by points on just one variable (t)
    """
    segments = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        segment_function = lambda t, slope=slope, intercept=intercept: slope * t + intercept
        segments.append(segment_function)
    
    def f(t):
        for i, segment_function in enumerate(segments):
            if t >= points[i][0] and t <= points[i + 1][0]:
                return segment_function(t)
        raise Exception('t is out of the domain of the curve')
    
    return f

def compute_geodesic_dm(x1, x2, energy_function, N_pieces=50, steps=20, lr=.1, plot=False, device='cuda'):
    """
    Compute a geodesic using linear piecewise direct minimization
    x1, x2: R_2
    energy_function: computes curve energy between x1 and x2
    N_pieces: number of pieces to divide the curve into
    steps: number of optimization steps
    """
    v_sp = x2-x1 # shortest euclidean path
    curve_points = torch.linspace(0, 1, N_pieces).expand(2,-1).T * v_sp + x1
    curve_points_optim = torch.nn.Parameter(torch.clone(curve_points[1:-1]))

    opt = torch.optim.AdamW([curve_points_optim], lr=lr)
    with tqdm(range(steps)) as pbar:
        for step in pbar:
            opt.zero_grad()

            all_curve_points = torch.cat([x1.unsqueeze(0), curve_points_optim, x2.unsqueeze(0)]).to(device)

            energy = energy_function(all_curve_points)
            energy.backward()
            opt.step()
            
            energy_p = energy.detach().cpu()
            pbar.set_description(f"step={step}, energy={energy_p}")
    
    curve_points_final = torch.cat([x1.unsqueeze(0), curve_points_optim, x2.unsqueeze(0)], dim=0)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(energies)
        ax2.plot(curve_points_final.detach().numpy()[:,0], curve_points_final.detach().numpy()[:,1])
        plt.show()

    return curve_points_final

     


def e6():
    x1 = torch.tensor([1, 0])
    x2 = torch.tensor([3, -1])
    compute_geodesic_dm(x1, x2, G_assignment)


def e5():
    print(curve_length(c, 1000))
    print(curve_length_r2(f1, f2, 1000))
    print(integral_curve_length(d_c_t_mag, 10000))

if __name__=="__main__":
    e6()    