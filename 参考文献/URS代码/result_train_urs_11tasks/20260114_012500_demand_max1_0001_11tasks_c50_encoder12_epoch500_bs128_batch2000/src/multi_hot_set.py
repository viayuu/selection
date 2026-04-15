def get_problem_attributes_set():

    # [identifier0, coordinates1, demand2, prize3, penalty4, early_time5, late_time6, service_time7, depot8, pickup9, delivery10, multi_route11, open_route12]
    problem_attributes_set = {
        'atsp':  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # identifier
        'tsp':   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # coordinates
        'op':    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,depot
        'pctsp': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'spctsp':[0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'cvrp':  [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # coordinates,demand, depot, delivery, multi route
        'pdtsp':   [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # coordinates,depot, pickup,delivery,
        'acvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # identifier,demand, depot, delivery, multi route
        'mdcvrp':[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # coordinates,demand, depot, delivery, multi route
        'pdcvrp':[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # coordinates,demand, depot,pickup,delivery,multi_route
        'apdcvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # identifier,demand, depot,pickup,delivery,multi_route
        'amdcvrp':[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], # identifier,demand, depot, delivery, multi route
        'apdtsp': [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],   # identifier0,depot, pickup,delivery,
        'opdcvrp': [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # coordinates,demand, depot,pickup,delivery,multi_route,open_route
        'aopdcvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # identifier,demand, depot,pickup,delivery,multi_route,open_route
    }
    vrpmix_variants = get_problem_list("vrpmix_list")
    for k in vrpmix_variants:
        attributes = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0] # basic_vrp: coordinates,demand,depot, deliver,multi_route
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_attributes_set[k] = attributes

    avrpmix_variants = get_problem_list("avrpmix_list")
    for k in avrpmix_variants:
        attributes = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0] # identifier,demand, depot, delivery, multi route
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_attributes_set[k] = attributes

    mdvrpmix_variants = get_problem_list("mdvrpmix_list")
    for k in mdvrpmix_variants:
        attributes = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0] # coordinates ,demand, depot, delivery, multi route
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_attributes_set[k] = attributes

    amdvrpmix_variants = get_problem_list("amdvrpmix_list")
    for k in amdvrpmix_variants:
        attributes = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]  # identifier,demand, depot, delivery, multi route
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_attributes_set[k] = attributes

    return problem_attributes_set

def get_problem_validation_set():
    problem_validation_set={
        'tsp': {
            'episodes_100': 1000,
            'filename_100': 'test_TSP100_n10000.txt',
        },
        'op': {
            'episodes_100': 1000,
            'filename_100': 'op100_n10000_seed1234.pt',
        },
        'pctsp': {
            'episodes_100': 1000,
            'filename_100': 'pctsp100_n10000_seed1234.pt',
        },
        'spctsp': {
            'episodes_100': 1000,
            'filename_100': 'spctsp100_n10000_seed1234.pt',
        },
        'atsp': {
            'episodes_100': 1000,
            'filename_100': 'test_lkh3_atsp100_nums10000_seed1234_uniform.pt',
        },
        'pdtsp': {
            'episodes_100': 1000,
            'filename_100': 'pdtsp_100.pkl',
        },
        'apdtsp': {
            'episodes_100': 1000,
            'filename_100': 'apdtsp100_n1000_seed1234.pt',
            'extra_args': {
                'solution_100': '../data/apdtsp/apdtsp100_cost_ortools20s.pt'
            },
        },
        'cvrpl': {
            'episodes_100': 1000,
            'extra_args': {
                'solution_100': '../data/cvrpl/lkh_cvrpl100_uniform.pkl'
            },
            'filename_100': 'cvrpl100_uniform.pkl',
        },
        'ocvrp': {
            'episodes_100': 1000,
            'extra_args': {
                'solution_100': '../data/ocvrp/lkh_ocvrp100_uniform.pkl',
            },
            'filename_100': 'ocvrp100_uniform.pkl',
        },
        'cvrp': {
            'episodes_100': 1000,
            'filename_100': 'cvrp100_uniform.pkl',
        },
        'cvrptw': {
            'episodes_100': 1000,
            'extra_args': {
                'solution_100': '../data/cvrptw/hgs_pyvrp_cvrptw100_1000uni.pkl',
            },
            'filename_100': 'cvrptw100_1000uni.pkl',
        },
    }


    amdvrpmix_list = get_problem_list("amdvrpmix_list")
    avrpmix_list = get_problem_list("avrpmix_list")
    vrpbp_data_set = ['cvrpbp', 'ocvrpbp', 'cvrpbpl', 'ocvrpbptw', 'cvrpbpltw', 'cvrpbptw', 'ocvrpbpl', 'ocvrpbpltw']
    mdvrpbp_data_set = ['mdcvrpbp', 'mdocvrpbp', 'mdcvrpbpl', 'mdocvrpbptw', 'mdcvrpbpltw', 'mdcvrpbptw', 'mdocvrpbpl', 'mdocvrpbpltw']
    temp_list = amdvrpmix_list+avrpmix_list+vrpbp_data_set

    for problem in temp_list:
        validation_set = {
            problem: {
                "episodes_100": 1000,
                "filename_100": f"{problem}100_n1000_seed1234.pt",
                "extra_args": {"solution_100": f"../data/{problem}/{problem}100_ortools20s.pt"}
            }
        }
        problem_validation_set.update(validation_set)

    mdvrpmix_list = get_problem_list("mdvrpmix_list")
    for mdcvrp_problem in list(set(mdvrpmix_list) - set(mdvrpbp_data_set)):
        validation_set = {
            mdcvrp_problem:{
                "episodes_100": 1000,
                "filename_100": f"{mdcvrp_problem}100_n1000_seed1234.pt",
                "extra_args": {"solution_100": f"../data/{mdcvrp_problem}/{mdcvrp_problem}100_pyvrp20s.pt"}
            }
        }
        problem_validation_set.update(validation_set)
    for mdcvrp_problem in mdvrpbp_data_set:
        validation_set = {
            mdcvrp_problem:{
                "episodes_100": 1000,
                "filename_100": f"{mdcvrp_problem}100_n1000_seed1234.pt",
            }
        }
        problem_validation_set.update(validation_set)

    vrpmix_list = get_problem_list("vrpmix_list")
    recorded_vrpmix_data_set = ['cvrpl', 'ocvrp', 'cvrp', 'cvrptw',]
    temp_list = list(set(vrpmix_list) - set(recorded_vrpmix_data_set) - set(vrpbp_data_set))
    # 12CVRP
    for cvrp_problem in temp_list:
        validation_set = {
            cvrp_problem: {
                "episodes_100": 1000,
                "filename_100": f"{cvrp_problem}100_uniform.pkl",
                "extra_args": {"solution_100": f"../data/{cvrp_problem}/or_tools_400s_{cvrp_problem}100_uniform.pkl"}
            }
        }
        problem_validation_set.update(validation_set)

    pdcvrp_list = get_problem_list("pdcvrp_list")
    for pdcvrp_problem in pdcvrp_list:
        validation_set = {
            pdcvrp_problem: {
                "episodes_100": 1000,
                "filename_100": f"{pdcvrp_problem}100_C20_n1000_seed1234.pt",
                "extra_args": {"solution_100": f"../data/{pdcvrp_problem}/{pdcvrp_problem}100_C20_cost_ortools20s.pt"}
            }
        }
        problem_validation_set.update(validation_set)


    #large_scale
    validation_set ={
        'cvrptw': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,

            'extra_args': {
                'solution_100': '../data/cvrptw/hgs_pyvrp_cvrptw100_1000uni.pkl',
                'solution_500': '../data/cvrptw/or_tools_400s_cvrptw500_128uni.pkl',
                'solution_1000': '../data/cvrptw/cvrptw1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': '../data/cvrptw/cvrptw2000_C300_n16_seed1234_pyvrp2400s.pt',
                'solution_3000': '../data/cvrptw/cvrptw3000_C300_n16_seed1234_pyvrp3600s.pt',
                'solution_4000': '../data/cvrptw/cvrptw4000_C300_n16_seed1234_pyvrp9600s.pt',
                'solution_5000': '../data/cvrptw/cvrptw5000_C300_n16_seed1234_pyvrp12000s.pt',

            },
            'filename_100': 'cvrptw100_1000uni.pkl',
            'filename_500': 'cvrptw500_128uni.pkl',
            'filename_1000': 'cvrptw1000_C200_n16_seed1234.pt',
            'filename_2000': 'cvrptw2000_C300_n16_seed1234.pt',
            'filename_3000': 'cvrptw3000_C300_n16_seed1234.pt',
            'filename_4000': 'cvrptw4000_C300_n16_seed1234.pt',
            'filename_5000': 'cvrptw5000_C300_n16_seed1234.pt',
        },

        'cvrpb': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,
            'extra_args': {
                'solution_100': '../data/cvrpb/or_tools_400s_cvrpb100_uniform.pkl',
                'solution_500': '../data/cvrpb/or_tools_400s_vrpb500_128uni.pkl',
                'solution_1000': '../data/cvrpb/cvrpb1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': '../data/cvrpb/cvrpb2000_C300_n16_seed1234_pyvrp2400s.pt',
                'solution_3000': '../data/cvrpb/cvrpb3000_C300_n16_seed1234_pyvrp3600s.pt',
                'solution_4000': '../data/cvrpb/cvrpb4000_C300_n16_seed1234_pyvrp9600s.pt',
                'solution_5000': '../data/cvrpb/cvrpb5000_C300_n16_seed1234_pyvrp12000s.pt',

            },
            'filename_100': 'cvrpb100_uniform.pkl',
            'filename_500': 'cvrpb500_128uni.pkl',
            'filename_1000': 'cvrpb1000_C200_n16_seed1234.pt',
            'filename_2000': 'cvrpb2000_C300_n16_seed1234.pt',
            'filename_3000': 'cvrpb3000_C300_n16_seed1234.pt',
            'filename_4000': 'cvrpb4000_C300_n16_seed1234.pt',
            'filename_5000': 'cvrpb5000_C300_n16_seed1234.pt',
        },

        'ocvrptw': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,

            'extra_args': {
                'solution_100': '../data/ocvrptw/or_tools_400s_ocvrptw100_uniform.pkl',
                'solution_500': '../data/ocvrptw/or_tools_400s_ocvrptw500_128uni.pkl',
                'solution_1000': '../data/ocvrptw/ocvrptw1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': '../data/ocvrptw/ocvrptw2000_C300_n16_seed1234_pyvrp2400s.pt',
                'solution_3000': '../data/ocvrptw/ocvrptw3000_C300_n16_seed1234_pyvrp3600s.pt',
                'solution_4000': '../data/ocvrptw/ocvrptw4000_C300_n16_seed1234_pyvrp9600s.pt',
                'solution_5000': '../data/ocvrptw/ocvrptw5000_C300_n16_seed1234_pyvrp12000s.pt',
            },
            'filename_100': 'ocvrptw100_uniform.pkl',
            'filename_500': 'ocvrptw500_128uni.pkl',
            'filename_1000': 'ocvrptw1000_C200_n16_seed1234.pt',
            'filename_2000': 'ocvrptw2000_C300_n16_seed1234.pt',
            'filename_3000': 'ocvrptw3000_C300_n16_seed1234.pt',
            'filename_4000': 'ocvrptw4000_C300_n16_seed1234.pt',
            'filename_5000': 'ocvrptw5000_C300_n16_seed1234.pt',
        },
    }

    problem_validation_set.update(validation_set)

    return problem_validation_set



def get_problem_list(name):
    ############################################################################################
    # Commonly adopted problem classification methods
    ############################################################################################
    # CVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    vrpmix_list = ["cvrptw", "ocvrp", "cvrpl", "cvrpb", "ocvrptw", 'ocvrpb', 'cvrpbl', 'cvrpltw', 'ocvrpbtw', 'cvrpbltw',
                   'ocvrpl', 'cvrpbtw', 'ocvrpbl', 'ocvrpltw', 'ocvrpbltw',
                   "cvrpbp", "ocvrpbp", "cvrpbpl", "ocvrpbptw", "cvrpbpltw", "cvrpbptw", "ocvrpbpl", "ocvrpbpltw","cvrp"
                   ]

    # ACVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    avrpmix_list = ["acvrptw", "aocvrp", "acvrpl", "acvrpb", "aocvrptw", 'aocvrpb', 'acvrpbl', 'acvrpltw', 'aocvrpbtw',
                    'acvrpbltw', 'aocvrpl', 'acvrpbtw', 'aocvrpbl', 'aocvrpltw', 'aocvrpbltw',
                    "acvrpbp", "aocvrpbp", "acvrpbpl", "aocvrpbptw", "acvrpbpltw", "acvrpbptw", "aocvrpbpl", "aocvrpbpltw","acvrp"]

    # MDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    mdvrpmix_list = ["mdcvrptw", "mdocvrp", "mdcvrpl", "mdcvrpb", "mdocvrptw", 'mdocvrpb', 'mdcvrpbl', 'mdcvrpltw', 'mdocvrpbtw',
                    'mdcvrpbltw', 'mdocvrpl', 'mdcvrpbtw', 'mdocvrpbl', 'mdocvrpltw', 'mdocvrpbltw',
                     "mdcvrpbp", "mdocvrpbp", "mdcvrpbpl", "mdocvrpbptw", "mdcvrpbpltw", "mdcvrpbptw", "mdocvrpbpl", "mdocvrpbpltw","mdcvrp"]

    # AMDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    amdvrpmix_list = ["amdcvrptw", "amdocvrp", "amdcvrpl", "amdcvrpb", "amdocvrptw", 'amdocvrpb', 'amdcvrpbl', 'amdcvrpltw', 'amdocvrpbtw',
                    'amdcvrpbltw', 'amdocvrpl', 'amdcvrpbtw', 'amdocvrpbl', 'amdocvrpltw', 'amdocvrpbltw',
                      "amdcvrpbp", "amdocvrpbp", "amdcvrpbpl", "amdocvrpbptw", "amdcvrpbpltw", "amdcvrpbptw", "amdocvrpbpl",
                      "amdocvrpbpltw","amdcvrp"]

    # PD problems
    pdcvrp_list = ['pdcvrp','apdcvrp','aopdcvrp','opdcvrp']

    #  remaining problem with one depot
    remain_problem_with_one_depot = ['pctsp','spctsp','op','pdtsp','apdtsp']

    # tsp
    tsp_and_atsp = ['tsp','atsp']
    ############################################################################################
    # End. A total of 107 problems
    ############################################################################################

    ############################################################################################
    # Another way to categorize problems
    ############################################################################################
    # 11 training problems
    train_problem_list = ['atsp', 'tsp', 'op', 'pctsp', 'pdtsp', 'acvrp', 'cvrp', 'cvrptw', 'cvrpb', 'ocvrp', 'ocvrptw']

    # 32 b constraints problems
    constraint_b_list = ['cvrpb', 'ocvrpb', 'cvrpbl', 'ocvrpbtw', 'cvrpbltw', 'cvrpbtw', 'ocvrpbl', 'ocvrpbltw',
     'acvrpb', 'aocvrpb', 'acvrpbl', 'aocvrpbtw', 'acvrpbltw', 'acvrpbtw', 'aocvrpbl', 'aocvrpbltw',
     'mdcvrpb', 'mdocvrpb', 'mdcvrpbl', 'mdocvrpbtw', 'mdcvrpbltw', 'mdcvrpbtw', 'mdocvrpbl', 'mdocvrpbltw',
     'amdcvrpb', 'amdocvrpb', 'amdcvrpbl', 'amdocvrpbtw', 'amdcvrpbltw', 'amdcvrpbtw', 'amdocvrpbl', 'amdocvrpbltw',]

    # A problems
    A_list = ['atsp','apdtsp','apdcvrp','aopdcvrp']
    A_list = A_list + avrpmix_list + amdvrpmix_list

    # Different problem set
    all_vrpmix_list = vrpmix_list + avrpmix_list + mdvrpmix_list+ amdvrpmix_list # 96 tasks

    single_depot_list = vrpmix_list+avrpmix_list+remain_problem_with_one_depot+pdcvrp_list

    multi_depot_list = mdvrpmix_list + amdvrpmix_list

    mapping = {
        "vrpmix_list": vrpmix_list,
        "avrpmix_list": avrpmix_list,

        "amdvrpmix_list": amdvrpmix_list,
        "mdvrpmix_list": mdvrpmix_list,

        "pdcvrp_list": pdcvrp_list,

        "single_depot_list": single_depot_list,
        "multi_depot_list": multi_depot_list,
        "all_vrpmix_list": all_vrpmix_list,

        "constraint_b_list": constraint_b_list,
        "A_list": A_list,

        'train_problem_list': train_problem_list,
    }

    return mapping[name]