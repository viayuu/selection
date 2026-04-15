from EasyNCO.data import *
from typing import Callable

def generate_data(problem_name: str) -> Callable:
    data_generator = {
        "TSP": TSPGenerator,
        "CVRP": CVRPGenerator,
        "KP": KPGenerator,
        "TSPImprove": TSPGenerator,
        "CVRPImprove": CVRPGenerator,
        "CVRPNLNS": CVRPGenerator,
        "ASHPP": ATSPGenerator,
        "MTSP": TSPGenerator,
        "MPDP": TSPGenerator,
        "MDVRP": TSPGenerator,
        "FMDVRP": TSPGenerator,
        "ATSP": ATSPGenerator,
        "FFSP": FFSPGenerator,
        "PCTSP": PCTSPGenerator,
        "OP": OPGenerator,
        "SOP": SOPGenerator,
        "SMTWTP": SMTWTPGenerator,
        "RCPSP": RCPSPGenerator,
        "MKP": MKPGenerator,
        "BPP": BPPGenerator,
        "MIS": MISGenerator,
        "MVRP": MVRPGenerator,
        # For LEHD and BQ-NCO
        "TSPPartialUpdater": TSPGenerator,
        "CVRPPartialUpdater": CVRPGenerator,
        "JSSP": JSSPGenerator,
        "FJSP": FJSPGenerator,
        "HCP": HCPGenerator,
        "SAT": SATGenerator,
        # For MOCO
        "MOTSP": MOTSPGenerator,
        "MOCVRP": MOCVRPGenerator,
        "MOKP": MOKPGenerator,
    }
    if problem_name not in data_generator.keys():
        raise ValueError(
            f"The given Generator name '{problem_name}' is not supported. Available data generators: {data_generator.keys()}")
    return data_generator[problem_name]
