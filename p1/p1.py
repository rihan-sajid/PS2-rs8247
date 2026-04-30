import numpy as np
import matplotlib.pyplot as plt

# 1. Core Functions
def get_chebyshev_nodes(p):
    n = p + 1
    k = np.arange(1, n + 1)
    return np.cos((2*k - 1) / (2*n) * np.pi)

def lagrange_interpolation(x_eval, x_nodes, y_nodes):
    """Computes the Lagrange interpolant g(x) at points x_eval."""
    n = len(x_nodes)
    m = len(x_eval)
    g = np.zeros(m)
    for i in range(n):
        # Calculate the Lagrange basis L_i
        li = np.ones(m)
        for j in range(n):
            if i != j:
                li *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        g += y_nodes[i] * li
    return g

# Define the functions to interpolate
f_a = lambda x: 1 / (1 + 25 * x**2)
f_b = lambda x: np.abs(x)
f_c = lambda x: np.where(x >= 0.25, 1.0, 0.0)

functions = [
    (f_a, "f(x) = 1/(1+25x^2)"),
    (f_b, "f(x) = |x|"),
    (f_c, "f(x) = H(x - 0.25)")
]

# Evaluation grid
N_fine = 1000
x_fine = np.linspace(-1, 1, N_fine)
degrees = [10, 20, 40]

# --- Figure 1: Interpolation Plots ---
fig1, axes = plt.subplots(3, 1, figsize=(10, 15))

for idx, (func, name) in enumerate(functions):
    ax = axes[idx]
    ax.plot(x_fine, func(x_fine), 'k--', label="Actual f(x)", alpha=0.5)
    
    for p in degrees:
        nodes = get_chebyshev_nodes(p)
        y_nodes = func(nodes)
        g_x = lagrange_interpolation(x_fine, nodes, y_nodes)
        
        line, = ax.plot(x_fine, g_x, label=f"p={p}")
        ax.scatter(nodes, y_nodes, marker='o', s=15, color=line.get_color())
    
    ax.set_title(f"Interpolation for {name}")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("p1\interpolation_plots.png")

# --- Figure 2: Error Analysis (Log-Log) ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
p_vals = np.unique(np.logspace(0, 8, num=15, base=2).astype(int)) # p from 1 to 256

for func, name in functions:
    l2_errors = []
    max_errors = []
    
    for p in p_vals:
        nodes = get_chebyshev_nodes(p)
        y_nodes = func(nodes)
        g_x = lagrange_interpolation(x_fine, nodes, y_nodes)
        f_val = func(x_fine)
        
        l2 = np.sqrt(np.mean((g_x - f_val)**2))
        
        l2_errors.append(l2)
    
    ax2.loglog(p_vals, l2_errors, label=f"L2 - {name}")

ax2.set_xlabel("Degree p")
ax2.set_ylabel("Error")
ax2.set_title("Error Convergence (Log-Log Scale)")
ax2.legend(fontsize='small', ncol=2)
ax2.grid(True, which="both", ls="-")

plt.savefig("p1\error_convergence.png")