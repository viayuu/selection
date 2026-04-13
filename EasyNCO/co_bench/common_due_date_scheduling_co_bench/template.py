import math
from typing import List, Tuple, Dict

def solve(jobs: List[Tuple[int, int, int]], h: float = 0.6) -> Dict[str, List[int]]:
    """
    Solves the restricted single‐machine common due date scheduling problem.
    The problem:
       Given a list of jobs where each job is represented as a tuple (p, a, b):
         • p: processing time
         • a: earliness penalty coefficient
         • b: tardiness penalty coefficient
       and an optional parameter h (default 0.6), the common due date is computed as:
             d = floor(sum(p) * h)
       A schedule (i.e., a permutation of job indices in 1‐based numbering) is produced.
       When processing the jobs in that order, the penalty is computed by:
         • Adding a × (d − C) if a job’s completion time C is less than d,
         • Adding b × (C − d) if C is greater than d,
         • No penalty if C equals d.
       The objective is to minimize the total penalty.
    Input kwargs:
         - 'jobs' (List[Tuple[int, int, int]]): a list of tuples where each tuple represents a job with:
              • p (int): processing time,
              • a (int): earliness penalty coefficient,
              • b (int): tardiness penalty coefficient.
         - Optional: 'h' (float): the factor used to compute the common due date (default is 0.6).
    Evaluation Metric:
         The computed schedule is evaluated by accumulating processing times and applying
         the appropriate earliness/tardiness penalties with respect to the common due date.
    Returns:
         A dictionary with key 'schedule' whose value is a list of integers representing
         a valid permutation of job indices (1-based).
    """
    total_p = sum(p for p, _, _ in jobs)
    d = math.floor(total_p * h)

    n = len(jobs)
    job_indices = list(range(1, n + 1))

    # Sort jobs based on a heuristic: higher penalty coefficients first
    sorted_jobs = sorted(zip(job_indices, jobs), key=lambda x: (-x[1][1], -x[1][2], x[1][0]))
    schedule = [idx for idx, _ in sorted_jobs]

    # Local search to improve the schedule
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                new_schedule = schedule.copy()
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                # Calculate penalties
                current_penalty = 0
                new_penalty = 0
                current_time = 0
                new_time = 0

                for k in range(n):
                    # Current schedule
                    idx = schedule[k] - 1
                    p, a, b = jobs[idx]
                    current_time += p
                    if current_time < d:
                        current_penalty += a * (d - current_time)
                    elif current_time > d:
                        current_penalty += b * (current_time - d)

                    # New schedule
                    new_idx = new_schedule[k] - 1
                    p_new, a_new, b_new = jobs[new_idx]
                    new_time += p_new
                    if new_time < d:
                        new_penalty += a_new * (d - new_time)
                    elif new_time > d:
                        new_penalty += b_new * (new_time - d)

                if new_penalty < current_penalty:
                    schedule = new_schedule
                    improved = True
                    break
            if improved:
                break

    return {'schedule': schedule}

