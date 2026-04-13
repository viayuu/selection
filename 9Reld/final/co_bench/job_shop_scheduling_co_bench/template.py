import numpy as np
import scipy.optimize as opt
def solve(n_jobs: int, n_machines: int, times: list, machines: list) -> dict:
    """
    Solves a single job shop scheduling test case.
    Input:
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
            Each start time must be a non-negative integer, and the schedule must respect the following constraints:
                (i) Sequential processing: For each job, an operation cannot start until its preceding operation has finished.
                (ii) Machine exclusivity: For operations assigned to the same machine, their processing intervals must not overlap.
            The evaluation function will use the start_times to compute the makespan and verify the constraints.
    """
    from collections import defaultdict
    import random

    # Initialize start times and machine schedules
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_schedules = defaultdict(list)  # key: machine, value: list of (start, end, job, op)

    # Helper to check if a time slot is available on a machine
    def is_available(machine, start, duration):
        for (s, e, _, _) in machine_schedules[machine]:
            if not (start + duration <= s or start >= e):
                return False
        return True

    # Assign start times operation by operation
    for job in range(n_jobs):
        for op in range(n_machines):
            machine = machines[job][op]
            duration = times[job][op]
            # Earliest possible start time considering job precedence
            if op == 0:
                earliest_start = 0
            else:
                earliest_start = start_times[job][op-1] + times[job][op-1]
            # Find the earliest available slot on the machine
            start = earliest_start
            while True:
                if is_available(machine, start, duration):
                    break
                start += 1
            start_times[job][op] = start
            machine_schedules[machine].append((start, start + duration, job, op))
            # Sort machine schedules by start time for efficiency
            machine_schedules[machine].sort()

    return {"start_times": start_times}


