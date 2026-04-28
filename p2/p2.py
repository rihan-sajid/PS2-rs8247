import numpy as np
import matplotlib.pyplot as plt

def gaussian_5pt_integrate(f, a, b, N):
    # 5-point Gauss-Legendre weights and nodes on [-1, 1]
    nodes = np.array([
        -0.9061798459386640, -0.5384693101056831, 0.0,
        0.5384693101056831, 0.9061798459386640
    ])
    weights = np.array([
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891
    ])
    
    edges = np.linspace(a, b, N + 1)
    total_integral = 0
    
    for i in range(N):
        x_left, x_right = edges[i], edges[i+1]
        mid = (x_left + x_right) / 2
        half_width = (x_right - x_left) / 2
        
        # Map nodes to the current interval
        x_mapped = mid + half_width * nodes
        total_integral += half_width * np.sum(weights * f(x_mapped))
        
    return total_integral

# --- Problem Definitions ---
c = 1 / np.sqrt(2)
functions = [
    {'f': lambda x: x**8, 'range': (-1, 1), 'exact': 2/9, 'label': 'x^8', 'slope': 10},
    {'f': lambda x: np.abs(x - c)**3, 'range': (-1, 1), 'exact': ( (1-c)**4 + (1+c)**4 ) / 4, 'label': '|x-c|^3', 'slope': 4},
    {'f': lambda x: np.where(x > c, 1.0, 0.0), 'range': (-1, 1), 'exact': 1 - c, 'label': 'H(x-c)', 'slope': 1},
    {'f': lambda x: 1/np.sqrt(x + 1e-15), 'range': (0, 1), 'exact': 2.0, 'label': '1/sqrt(x)', 'slope': 0.5}
]

N_vals = np.logspace(1, 5, 10, dtype=int)
L = 2 # Interval length for (a,b,c), adjusted for (d) in loop
plt.figure(figsize=(10, 6))

for entry in functions:
    errors = []
    h_vals = []
    f, a, b, exact, label = entry['f'], entry['range'][0], entry['range'][1], entry['exact'], entry['label']
    
    for N in N_vals:
        approx = gaussian_5pt_integrate(f, a, b, N)
        h = (b - a) / N
        rel_error = np.abs(approx - exact) / np.abs(exact)
        if rel_error > 1e-15: # Ignore noise below precision
            errors.append(rel_error)
            h_vals.append(h)
    
    plt.loglog(h_vals, errors, 'o-', label=f"Error: {label}")
    
    # Power law reference line
    h_ref = np.array(h_vals)
    y_ref = (h_ref**entry['slope']) * (errors[0] / h_ref[0]**entry['slope'])
    plt.loglog(h_ref, y_ref, '--', alpha=0.6, label=f"Ref: h^{entry['slope']}")

plt.xlabel('h = (b-a)/N')
plt.ylabel('Relative Error')
plt.title('Convergence of 5-point Composite Gaussian Quadrature')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig("p2\convergence_plots.png")