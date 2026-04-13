import numpy as np
import scipy.optimize as opt
def solve(n_jobs: int, n_machines: int, times: list, machines: list) -> dict:
    """
    Solves a single open shop scheduling test case.
    Input kwargs:
        - n_jobs (int): Number of jobs.
        - n_machines (int): Number of machines (and operations per job).
        - times (list of list of int): A 2D list of processing times for each operation.
          Dimensions: n_jobs x n_machines.
        - machines (list of list of int): A 2D list specifying the machine assignment for each operation.
          Dimensions: n_jobs x n_machines. Note machine is 1-indexed.
    Output:
        solution (dict): A dictionary containing:
            - start_times (list of list of int): A 2D list of start times for each operation.
              Dimensions: n_jobs x n_machines.
            Each start time must be a non-negative integer, and the schedule must respect the following constraint:
                (i) Non-parallel operation: Each job must be processed on only one machine at a time
                (ii) Machine exclusivity: For operations assigned to the same machine, their processing intervals must not overlap.
            The evaluation function will use the start_times to compute the makespan and verify the constraints.
    """
    start_times = [[-1 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_available = [0] * (n_machines + 1)  # 1-indexed
    job_available = [0] * n_jobs

    # Create a list of all operations with their job, machine, and time
    operations = []
    for job in range(n_jobs):
        for op in range(n_machines):
            machine = machines[job][op]
            time = times[job][op]
            operations.append((job, op, machine, time))

    # Sort operations by some heuristic (e.g., longest total job time first)
    job_total_time = [sum(times[job]) for job in range(n_jobs)]
    operations.sort(key=lambda x: (-job_total_time[x[0]], x[3]))

    # Schedule operations
    for job, op, machine, time in operations:
        # Earliest start is max of job's last operation end and machine's last operation end
        earliest_start = max(job_available[job], machine_available[machine])
        start_times[job][op] = earliest_start
        job_available[job] = earliest_start + time
        machine_available[machine] = earliest_start + time

    return {"start_times": start_times}


