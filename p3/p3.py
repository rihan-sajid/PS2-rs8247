import numpy as np
import matplotlib.pyplot as plt

def solve_jacobi(m, N=100, iterations=1000):
    # Set up the grid and the exact theoretical solution (the sine mode)
    n = np.arange(N)
    v_m = np.sin(2 * np.pi * m * n / N)
    
    # Calculate the exact eigenvalue and construct the right-hand side b
    lambda_m = 2 - 2 * np.cos(2 * np.pi * m / N)
    b = lambda_m * v_m
    
    # Initialize the guess x^(0) = 0
    x = np.zeros(N)
    errors = []
    
    for k in range(iterations):
        x_new = np.zeros(N)
        
        # Jacobi interior update rule
        x_new[1:-1] = 0.5 * (x[0:-2] + x[2:] + b[1:-1])
        
        # Cyclic boundary conditions
        x_new[0] = 0.5 * (x[-1] + x[1] + b[0])
        x_new[-1] = 0.5 * (x[-2] + x[0] + b[-1])
        
        # Calculate max absolute error compared to the exact solution v_m
        error = np.max(np.abs(v_m - x_new))
        errors.append(error)
        
        # Update x for the next iteration
        x = x_new
        
    return errors

# Run the simulation for m=1 and m=2
errors_m1 = solve_jacobi(m=1)
errors_m2 = solve_jacobi(m=2)

# Calculate the observed convergence rates from the final iterations
rate_m1 = errors_m1[-1] / errors_m1[-2]
rate_m2 = errors_m2[-1] / errors_m2[-2]

print(f"Observed Convergence Rate (m=1): {rate_m1:.7f}")
print(f"Observed Convergence Rate (m=2): {rate_m2:.7f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(errors_m1, label='m = 1')
plt.semilogy(errors_m2, label='m = 2')
plt.xlabel('Iteration count (k)')
plt.ylabel('Max Error (log scale)')
plt.title('Jacobi Scheme Error vs. Iterations')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("p3\error_plots.png")