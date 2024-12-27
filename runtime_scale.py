import time
import numpy as np
import matplotlib.pyplot as plt
from main import solve_linear_program
from data.gen_std_form_input import generate_matrix_and_vectors

def test_solver_scaling():
    # Define range of problem sizes to test
    m_sizes = np.linspace(5, 300, 20, dtype=int)  
    runtimes = []
    problem_sizes = []
    num_trials = 20  # Number of trials for each m
    # Define file paths relative to data folder
    input_path = "data/LP_input.in"
    output_path = "data/LP_expected_output.out"
    input_file = "LP_input.in"
    output_file = "LP_expected_output.out"

    for m in m_sizes:
        m_runtimes = []  # Store runtimes for each trial of this m
        print(f"\nTesting problem size m={m}")
        
        for trial in range(num_trials):
            # Randomly select n within range (m - m//3, m)
            n = np.random.randint(m - m//3, m + 1)
            
            # Generate test problem
            generate_matrix_and_vectors(n, m, input_path, output_path)
            
            # Measure runtime
            start_time = time.time()
            solution = solve_linear_program(input_file, output_file)
            end_time = time.time()
            
            runtime = end_time - start_time
            m_runtimes.append(runtime)
            print(f"  Trial {trial + 1}/20 (n={n}): Runtime = {runtime:.4f} seconds")
        
        # Calculate average runtime for this m
        avg_runtime = np.mean(m_runtimes)
        runtimes.append(avg_runtime)
        problem_sizes.append(m)
        print(f"Average runtime for m={m}: {avg_runtime:.4f} seconds")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(problem_sizes, runtimes, 'bo-')
    plt.xlabel('Problem Size (m)')
    plt.ylabel('Runtime (seconds)')
    plt.title('LP Solver Runtime Scaling')
    plt.grid(True)
    
    # Add polynomial fit
    z = np.polyfit(problem_sizes, runtimes, 2)
    p = np.poly1d(z)
    plt.plot(problem_sizes, p(problem_sizes), 'r--', 
             label=f'Polynomial fit: {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}')
    
    plt.legend()
    plt.savefig(f'plots/runtime_scaling.png')
    plt.close()
    
    # Save data to file
    with open('plots/scaling_results.txt', 'w') as f:
        f.write("Problem Size (m)\tRuntime (s)\n")
        for size, runtime in zip(problem_sizes, runtimes):
            f.write(f"{size}\t{runtime}\n")

if __name__ == "__main__":
    test_solver_scaling()
