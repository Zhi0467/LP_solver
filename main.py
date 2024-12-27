from modules.standard_form import read_and_standardize
from modules.feasibility_check import check_and_fix_rank
from modules.big_m_initialization import initialize_tableau
from modules.simplex import solve_simplex
from modules.verification import verify_solution

def solve_linear_program(input_file="LP_input.in", output_file="LP_expected_output.out"):
    # Step 1: Read and convert to standard form
    c, A, b = read_and_standardize(input_file)
    # Step 2: Check feasibility
    is_feasible, c, A, b = check_and_fix_rank(c, A, b)
    if not is_feasible:
        print("Problem is infeasible")
        return None
    
    # Step 3: Initialize with Big-M method
    tableau_big_m = initialize_tableau(c, A, b, M = 1e6)
    
    # Step 4: Solve using Simplex method
    final_tableau, status, basic_vars, solution = solve_simplex(tableau_big_m)
    obj_val = -final_tableau[0, -1]
    # Step 5: Verify and report solution
    if status == "optimal":
        is_correct, results = verify_solution(obj_val, solution, output_file)
        if is_correct:
            print("Solution verified successfully!")
        else:
            print("Solution verification failed!")
            print("Detailed results:")
            for key, value in results.items():
                print(f"{key}: {value}\n")
            return obj_val, None
    elif status == "unbounded":
        print("This LP problem is unbounded!")
        return obj_val, None
    else:
        print("This LP problem is infeasible!")
        return obj_val, None
    
    return obj_val, solution

if __name__ == "__main__":
    obj_val, solution = solve_linear_program()
    if solution is not None:
        print("Optimization completed successfully")
        print(f"Optimized Value: {obj_val}\n")
        # print(f"Solution: {solution}\n")