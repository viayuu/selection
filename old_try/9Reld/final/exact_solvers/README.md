# File Structure
- exact_solver_main.py
- configs.yaml
- param_setting.py
    - hgs_cvrp.py
    - lkh_tsp.py
    - lkh_cvrp.py
    - lkh_atsp.py
    - ortools_tsp.py
    - ortools_cvrp.py 
    - eax_tsp.py
    - gurobi_tsp.py
    - gurobi_cvrp.py
    - cplex_tsp.py
    - cplex_cvrp.py
    - pyvrp_run.py
- solution.py
------------------------------------------------------------------------------------------------------------------------
# File Function
- First Class:
    - exact_solver_main.py: Use inference
    - configs.yaml: Set parameters, such as LKH's "MAX_TRIALS"
    - param_setting.py: Set parameters, such as LKH's "MAX_TRIALS"
    - solution.py: Export the results of each solver in the format of 'tour, cost'
- Second Class:
    seven solvers: lkh, eax, hgs, ortools, pyvrp, gurobi, cplex
------------------------------------------------------------------------------------------------------------------------
# Using instruction

INPUT FORMAT:
- single process
    - tsp
        + node_coords: [(x_0, y_0), (x_1, y_1), ..., (x_n, y_n)]
    - cvrp
        + depot: [(x_0, y_0)]
        + locs: [(x_1, y_1), ..., (x_n, y_n)]
        + demand: [d_1, d_2, ..., d_n]
        + capacity: C

- multi process
    - tsp
        + data_list:[node_coords_1, node_coords_2, ..., node_coords_n]
    - cvrp
        + data_list:[[depot_1, locs_1, demand_1, capacity_1], ... ,]


There are two usage interfaces: one for directly using exact solvers to solve problems, and the other for serving as a function interface that is part of other neural combinatorial optimization methods.

## 1.Directly using exact solvers to solve problems

**1.Set the parameters in the file "configs.yaml"**

**2.Generate the input data**

**3.Run the file "use_main.py"**

FOR EXAMPLE: If you use HGS solver to solve a CVRP problem, and you want to solve a instance each times using 'single process' mode

In the file 'configs.yaml', set the paramters
- solver: "hgs"
- ptype: "cvrp"
- solver_mode: "single" # choices=['single', 'multi'], 'single' means solving one instance each time, 'multi' means solving a batch of instances each 
- problem_number: 10 # number of problem instances
- problem_scale: 20 # number of nodes
- batch_size: 5 # batch size for dataloader
- int_coord_scale: 100 # Expand coordinate values, transform coordinates into integers
- capacity: 50 # vehicle capacity, it is only used in CVRP
time    
- SEED: 1234
- time_threshold: 10 # time limit for HGS

## 2.Serving as a function interface that is part of other neural

**1.Call function "get_option()" in the file "param_setting.py"**

**2.Set the parameters to specify solver and problem type**

**3.Call function "exact_solver_main(cfg, data_list=None, node_coords=None, depot=None, locs=None, demand=None, capacity=None, node_matrix=None)" in the file "exact_solver_main.py"**

FOR EXAMPLE: If you use LKH solver to solve a ATSP problem, and you want to solve a batch of instances each times using 'multi process' mode
    
    from EasyNCO.exact_solvers.param_setting import get_option
    from EasyNCO.exact_solvers.exact_solver_main import exact_solver_main

    opts = get_option()
    opts.solver = 'lkh'
    opts.ptype = 'atsp'
    opts.solver_mode = 'multi'
    opts.cpus = 10
    opts.int_matrix_scale = 100
    opts.MAX_TRIALS = 100
    ...
    result = exact_solver_main(opts, data_list)

OUTPUT FORMAT: [(tour_1, cost_1),(tour_2, cost_2),......(tour_n,cost_n)]

the results will be stored in "./EasyNCO/exact_solver/result_lkh_atsp"

the executable will be stored in "./EasyNCO/exact_solver/use_exe/lkh"
    
------------------------------------------------------------------------------------------------------------------------

