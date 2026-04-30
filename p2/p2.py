import numpy as np
import matplotlib.pyplot as plt

def gaussian_5pt_integrate(f, N, a, b):
    """Composite 5-point Gaussian Quadrature."""
    # Standard 5-point Gauss-Legendre weights and nodes for [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(5)
    
    edges = np.linspace(a, b, N + 1)
    total_integral = 0.0
    
    for i in range(N):
        a_j, b_j = edges[i], edges[i+1]
        mid = (a_j + b_j) / 2.0
        half_width = (b_j - a_j) / 2.0
        
        # Map nodes from [-1, 1] to the local sub-interval [a_j, b_j]
        mapped_nodes = mid + half_width * nodes
        total_integral += half_width * np.sum(weights * f(mapped_nodes))
        
    return total_integral

# Define the functions and analytical solutions
c = 1 / np.sqrt(2)

# (a) x^8 on [-1, 1] -> Exact integral = 2/9
# (b) |x - c|^3 on [-1, 1] -> Exact integral = ((1+c)^4 + (1-c)^4)/4
# (c) H(x - c) on [-1, 1] -> Exact integral = 1 - c
# (d) 1/sqrt(x) on [0, 1] -> Exact integral = 2.0

configs = {
    "a": (lambda x: x**8, 2/9, -1, 1, 10, "h^{10}"),
    "b": (lambda x: np.abs(x - c)**3, ((1+c)**4 + (1-c)**4)/4, -1, 1, 3, "h^3"),
    "c": (lambda x: np.where(x > c, 1, 0), 1 - c, -1, 1, 1, "h^1"),
    "d": (lambda x: 1/np.sqrt(np.where(x == 0, 1e-15, x)), 2.0, 0, 1, 0.5, "h^{0.5}")
}

N_vals = np.logspace(1, 4.5, 15, dtype=int)
plt.figure(figsize=(12, 10))

for i, (key, (f, exact, a, b, p, p_lab)) in enumerate(configs.items()):
    h_vals = (b - a) / N_vals
    errors = [np.abs((gaussian_5pt_integrate(f, N, a, b) - exact) / exact) for N in N_vals]
    
    plt.subplot(2, 2, i+1)
    plt.loglog(h_vals, errors, 'o-', label=f"Function ({key}) Error")
    
    # Reference trend line
    ref_y = h_vals**p * (errors[0] / h_vals[0]**p)
    plt.loglog(h_vals, ref_y, 'k--', alpha=0.7, label=f"Ref: ${p_lab}$")
    
    plt.xlabel('$h = (b-a)/N$')
    plt.ylabel('Relative Error')
    plt.title(f'Convergence for Function ({key})')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('p2/integration_errors.png')