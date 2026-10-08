def get_problem_attributes_set():

    # [identifier0, coordinates1, demand2, prize3, penalty4, early_time5, late_time6, service_time7, depot8, pickup9, delivery10, multi_route11, open_route12]
    problem_attributes_set = {
        'atsp':  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # identifier
        'tsp':   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # coordinates
        'op':    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,depot
        'pctsp': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'spctsp':[0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'cvrp':  [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # coordinates,demand, depot, delivery, multi route
        'sdvrp': [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # coordinates,demand, depot, delivery, multi route
        'pdp':   [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # coordinates,depot, pickup,delivery,
        'acvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # identifier,demand, depot, delivery, multi route
        'mdcvrp':[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],  # coordinates,demand, depot, delivery, multi route
        'pdcvrp':[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # coordinates,demand, depot,pickup,delivery,multi_route
        'pdcvrpl': [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # coordinates,demand, depot,pickup,delivery,multi_route
        'apdcvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # identifier,demand, depot,pickup,delivery,multi_route
        'apdcvrpl': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # identifier,demand, depot,pickup,delivery,multi_route
        'amdcvrp':[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], # identifier,demand, depot, delivery, multi route
        'pdvrp':[0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],   # coordinates,depot, pickup,delivery,multi_route11
        'apdp': [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],   # identifier0,depot, pickup,delivery,
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
    # 1. 创建一个包含所有问题配置的列表
    # 每个字典代表一个问题的配置信息
    problem_validation_set = {
        'tsp': {
            'episodes_100': 1000,
            'filename_100': 'test_TSP100_n10000.txt',

            # 'episodes_100': 10000,
            # 'filename_100': 'tsp100_test_seed1234.pkl',

            'episodes_500': 128,
            'filename_500': 'test_TSP500_n128.txt',
            'episodes_1000': 128,
            'filename_1000': 'test_TSP1000_n128.txt',
        },


        'cvrp': {
            'episodes_100': 1000,
            'filename_100': 'cvrp100_uniform.pkl',

            'episodes_500': 128,
            'filename_500': 'cvrp500_128uni.pkl',
            'extra_args': {
                'solution_500': 'data/cvrp/hgs_pyvrp_cvrp500_128uni.pkl'
            },

        },
        'op': {
            # 'episodes_100': 1000,
            # 'filename_100': 'op100_n10000_seed1234.pt',

            'episodes_100': 10000,
            'filename_100': 'op_dist100_test_seed1234.pkl',


            'episodes_500': 128,
            'filename_500': 'op_dist500_test_seed1234.pkl',
        },
        'pctsp': {
            # 'episodes_100': 1000,
            # 'filename_100': 'pctsp100_n10000_seed1234.pt',

            'episodes_100': 10000,
            'filename_100': 'pctsp100_test_seed1234.pkl',

            'episodes_500': 128,
            'filename_500': 'pctsp500_test_seed1234.pkl',
        },
        'sdvrp': {
            'episodes_100': 1000,
            'filename_100': 'sdcvrp100_test.npz',
        },
        'spctsp': {
            'episodes_100': 1000,
            'filename_100': 'spctsp100_n10000_seed1234.pt',
        },
        'cvrptw': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,

            'extra_args': {
                'solution_100': 'data/cvrptw/hgs_pyvrp_vrptw100_1000uni.pkl',
                'solution_500': 'data/cvrptw/or_tools_400s_vrptw500_128uni.pkl',
                'solution_1000': 'data/cvrptw/cvrptw1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': 'data/cvrptw/cvrptw2000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_3000': 'data/cvrptw/cvrptw3000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_4000': 'data/cvrptw/cvrptw4000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_5000': 'data/cvrptw/cvrptw5000_C300_n16_seed1234_pyvrp1200s.pt',

            },
            'filename_100': 'vrptw100_1000uni.pkl',
            'filename_500': 'vrptw500_128uni.pkl',
            'filename_1000': 'cvrptw1000_C200_n16_seed1234.pt',
            'filename_2000': 'cvrptw2000_C300_n16_seed1234.pt',
            'filename_3000': 'cvrptw3000_C300_n16_seed1234.pt',
            'filename_4000': 'cvrptw4000_C300_n16_seed1234.pt',
            'filename_5000': 'cvrptw5000_C300_n16_seed1234.pt',
        },
        'atsp': {
            'episodes_100': 1000,
            'filename_100': 'test_lkh3_atsp100_nums10000_seed1234_uniform.pt',

            'episodes_500': 128,
            'filename_500': 'test_lkh3_atsp500_nums128_seed1234_uniform.pt',
        },
        'ovrp': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,

            'extra_args': {
                'solution_100': 'data/ovrp/lkh_ovrp100_uniform.pkl',
                'solution_500': 'data/ovrp/or_tools_400s_ovrp500_128uni.pkl',
                'solution_1000': 'data/ovrp/ovrp1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': 'data/ovrp/ovrp2000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_3000': 'data/ovrp/ovrp3000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_4000': 'data/ovrp/ovrp4000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_5000': 'data/ovrp/ovrp5000_C300_n16_seed1234_pyvrp1200s.pt',
            },
            'filename_100': 'ovrp100_uniform.pkl',
            'filename_500': 'ovrp500_128uni.pkl',
            'filename_1000': 'ovrp1000_C200_n16_seed1234.pt',
            'filename_2000': 'ovrp2000_C300_n16_seed1234.pt',
            'filename_3000': 'ovrp3000_C300_n16_seed1234.pt',
            'filename_4000': 'ovrp4000_C300_n16_seed1234.pt',
            'filename_5000': 'ovrp5000_C300_n16_seed1234.pt',

        },
        'vrpl': {
            'episodes_100': 1000,
            'extra_args': {
                'solution_100': 'data/vrpl/lkh_vrpl100_uniform.pkl'
            },
            'filename_100': 'vrpl100_uniform.pkl',
        },
        'vrpb': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,

            'extra_args': {
                'solution_100': 'data/vrpb/or_tools_400s_vrpb100_uniform.pkl',
                'solution_500': 'data/vrpb/or_tools_400s_vrpb500_128uni.pkl',
                'solution_1000': 'data/vrpb/vrpb1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': 'data/vrpb/vrpb2000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_3000': 'data/vrpb/vrpb3000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_4000': 'data/vrpb/vrpb4000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_5000': 'data/vrpb/vrpb5000_C300_n16_seed1234_pyvrp1200s.pt',

            },
            'filename_100': 'vrpb100_uniform.pkl',
            'filename_500': 'vrpb500_128uni.pkl',
            'filename_1000': 'vrpb1000_C200_n16_seed1234.pt',
            'filename_2000': 'vrpb2000_C300_n16_seed1234.pt',
            'filename_3000': 'vrpb3000_C300_n16_seed1234.pt',
            'filename_4000': 'vrpb4000_C300_n16_seed1234.pt',
            'filename_5000': 'vrpb5000_C300_n16_seed1234.pt',
        },
        'ovrptw': {
            'episodes_100': 1000,
            'episodes_500': 128,
            'episodes_1000': 16,
            'episodes_2000': 16,
            'episodes_3000': 16,
            'episodes_4000': 16,
            'episodes_5000': 16,


            'extra_args': {
                'solution_100': 'data/ovrptw/or_tools_400s_ovrptw100_uniform.pkl',
                'solution_500': 'data/ovrptw/or_tools_400s_ovrptw500_128uni.pkl',
                'solution_1000': 'data/ovrptw/ovrptw1000_C200_n16_seed1234_pyvrp1200s.pt',
                'solution_2000': 'data/ovrptw/ovrptw2000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_3000': 'data/ovrptw/ovrptw3000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_4000': 'data/ovrptw/ovrptw4000_C300_n16_seed1234_pyvrp1200s.pt',
                'solution_5000': 'data/ovrptw/ovrptw5000_C300_n16_seed1234_pyvrp1200s.pt',
            },
            'filename_100': 'ovrptw100_uniform.pkl',
            'filename_500': 'ovrptw500_128uni.pkl',
            'filename_1000': 'ovrptw1000_C200_n16_seed1234.pt',
            'filename_2000': 'ovrptw2000_C300_n16_seed1234.pt',
            'filename_3000': 'ovrptw3000_C300_n16_seed1234.pt',
            'filename_4000': 'ovrptw4000_C300_n16_seed1234.pt',
            'filename_5000': 'ovrptw5000_C300_n16_seed1234.pt',
        },
        'pdp': {
            'episodes_100': 1000,
            'filename_100': 'pdp_100.pkl',

            'episodes_500': 128,
            'filename_500': 'pdp500_n128_seed1234.pt',
        },
        'acvrp': {
            'episodes_100': 1000,
            'filename_100': 'acvrp100_n1000_seed1234.pt',

            'episodes_500': 128,
            'extra_args': {
                'solution_500': 'data/acvrp/acvrp500_ortools20s.pt'
            },
            'filename_500': 'acvrp500_n128_seed1234.pt',
        },
        'hcvrp':{
            'episodes_100': 1000,
            'filename_100': 'hcvrp_100_seed24610.pkl'
        },
        'mtsp':{
            'episodes_100': 100,
            'filename_100': 'mtsp100_test_seed3333.pkl'
        },
        'mdcvrp':{
            'episodes_100': 1000,
            'extra_args': {
                'solution_100': 'data/mdcvrp/mdcvrp100_ortools20s.pt'
            },
            'filename_100': 'mdcvrp100_n1000_seed1234.pt',
        },
        'pdcvrp':{
            'episodes_100': 1000,
            'filename_100': 'pdcvrp100_C20_n1000_seed1234.pt'
        },
        'opdcvrp': {
            'episodes_100': 1000,
            'filename_100': 'opdcvrp100_C20_n1000_seed1234.pt'
        },
        'apdcvrp':{
            'episodes_100': 1000,
            'filename_100': 'apdcvrp100_C20_n1000_seed1234.pt'
        },
        'aopdcvrp': {
            'episodes_100': 1000,
            'filename_100': 'aopdcvrp100_C20_n1000_seed1234.pt'
        },


        'amdcvrp':{
            'episodes_100':  1000,
            'filename_100': 'amdcvrp100_n1000_seed1234.pt',
            'extra_args': {
                'solution_100': 'data/amdcvrp/amdcvrp100_ortools20s.pt'
            },

        },
        'pdvrp':{
            'episodes_100': 1000,
            'filename_100': 'pdp_100.pkl',
        },
        'apdp':{
            'episodes_100': 1000,
            'filename_100': 'apdp100_n1000_seed1234.pt',
            'extra_args': {
                'solution_100': 'data/apdp/apdp100_cost_ortools20s.pt'
            },
        }
    }



    vrpmix_files = {
        "ovrpb": ("ovrpb100_uniform.pkl", "data/ovrpb/or_tools_400s_ovrpb100_uniform.pkl"),
        "vrpbl": ("vrpbl100_uniform.pkl", "data/vrpbl/or_tools_400s_vrpbl100_uniform.pkl"),
        "vrpltw": ("vrpltw100_uniform.pkl", "data/vrpltw/or_tools_400s_vrpltw100_uniform.pkl"),
        "ovrpbtw": ("ovrpbtw100_uniform.pkl", "data/ovrpbtw/or_tools_400s_ovrpbtw100_uniform.pkl"),
        "vrpbltw": ("vrpbltw100_uniform.pkl", "data/vrpbltw/or_tools_400s_vrpbltw100_uniform.pkl"),
        "ovrpl": ("ovrpl100_uniform.pkl", "data/ovrpl/or_tools_400s_ovrpl100_uniform.pkl"),
        "vrpbtw": ("vrpbtw100_uniform.pkl", "data/vrpbtw/or_tools_400s_vrpbtw100_uniform.pkl"),
        "ovrpbl": ("ovrpbl100_uniform.pkl", "data/ovrpbl/or_tools_400s_ovrpbl100_uniform.pkl"),
        "ovrpltw": ("ovrpltw100_uniform.pkl", "data/ovrpltw/or_tools_400s_ovrpltw100_uniform.pkl"),
        "ovrpbltw": ("ovrpbltw100_uniform.pkl", "data/ovrpbltw/or_tools_400s_ovrpbltw100_uniform.pkl"),


        # 'aovrp': ("aovrp100_n1000_seed1234.pt", "data/aovrp/aovrp100_pyvrp20s.pt"),
        # 'avrpb': ("avrpb100_n1000_seed1234.pt", "data/avrpb/avrpb100_pyvrp20s.pt"),
        # # 'avrpl': ("avrpl100_n1000_seed1234.pt", "data/avrpl/avrpl100_pyvrp20s.pt"),
        # 'avrpl': ("avrpl100[0.5]_n1000_seed1234.pt", "data/avrpl/avrpl100[0.5]_pyvrp20s.pt"),
        # 'acvrptw': ("acvrptw100_n1000_seed1234.pt", "data/acvrptw/acvrptw100_pyvrp20s.pt")
        # # 'acvrptw': ("acvrptw100[0.5,0.1]_n1000_seed1234.pt", "data/acvrptw/acvrptw100[0.5,0.1]_100_pyvrp20s.pt")

        # or-tools 20s
        # 'acvrp':     ("acvrp100_n1000_seed1234.pt", "data/acvrp/acvrp100_ortools20s.pt"),
        'acvrptw':   ("acvrptw100_n1000_seed1234.pt", "data/acvrptw/acvrptw100_ortools20s.pt"),
        'aovrp':     ("aovrp100_n1000_seed1234.pt",   "data/aovrp/aovrp100_ortools20s.pt"),
        'aovrpb':    ("aovrpb100_n1000_seed1234.pt",  "data/aovrpb/aovrpb100_ortools20s.pt"),
        'aovrpbl':   ("aovrpbl100_n1000_seed1234.pt", "data/aovrpbl/aovrpbl100_ortools20s.pt"),
        'aovrpbltw': ("aovrpbltw100_n1000_seed1234.pt","data/aovrpbltw/aovrpbltw100_ortools20s.pt"),
        'aovrpbtw':  ("aovrpbtw100_n1000_seed1234.pt","data/aovrpbtw/aovrpbtw100_ortools20s.pt"),
        'aovrpl':    ("aovrpl100_n1000_seed1234.pt",  "data/aovrpl/aovrpl100_ortools20s.pt"),
        'aovrpltw':  ("aovrpltw100_n1000_seed1234.pt","data/aovrpltw/aovrpltw100_ortools20s.pt"),
        'aovrptw':   ("aovrptw100_n1000_seed1234.pt", "data/aovrptw/aovrptw100_ortools20s.pt"),

        'avrpb':     ("avrpb100_n1000_seed1234.pt",   "data/avrpb/avrpb100_ortools20s.pt"),
        'avrpbl':    ("avrpbl100_n1000_seed1234.pt",  "data/avrpbl/avrpbl100_ortools20s.pt"),
        'avrpbltw':  ("avrpbltw100_n1000_seed1234.pt","data/avrpbltw/avrpbltw100_ortools20s.pt"),
        'avrpbtw':   ("avrpbtw100_n1000_seed1234.pt", "data/avrpbtw/avrpbtw100_ortools20s.pt"),
        'avrpl':     ("avrpl100_n1000_seed1234.pt",   "data/avrpl/avrpl100_ortools20s.pt"),
        'avrpltw':   ("avrpltw100_n1000_seed1234.pt", "data/avrpltw/avrpltw100_ortools20s.pt"),

        #md problem
        'mdcvrptw':   ("mdcvrptw100_n1000_seed1234.pt", "data/mdcvrptw/mdcvrptw100_ortools20s.pt"),
        'mdovrp':     ("mdovrp100_n1000_seed1234.pt",   "data/mdovrp/mdovrp100_ortools20s.pt"),
        'mdovrpb':    ("mdovrpb100_n1000_seed1234.pt",  "data/mdovrpb/mdovrpb100_ortools20s.pt"),
        'mdovrpbl':   ("mdovrpbl100_n1000_seed1234.pt", "data/mdovrpbl/mdovrpbl100_ortools20s.pt"),
        'mdovrpbltw': ("mdovrpbltw100_n1000_seed1234.pt","data/mdovrpbltw/mdovrpbltw100_ortools20s.pt"),
        'mdovrpbtw':  ("mdovrpbtw100_n1000_seed1234.pt","data/mdovrpbtw/mdovrpbtw100_ortools20s.pt"),
        'mdovrpl':    ("mdovrpl100_n1000_seed1234.pt",  "data/mdovrpl/mdovrpl100_ortools20s.pt"),
        'mdovrpltw':  ("mdovrpltw100_n1000_seed1234.pt","data/mdovrpltw/mdovrpltw100_ortools20s.pt"),
        'mdovrptw':   ("mdovrptw100_n1000_seed1234.pt", "data/mdovrptw/mdovrptw100_ortools20s.pt"),

        'mdvrpb':     ("mdvrpb100_n1000_seed1234.pt",   "data/mdvrpb/mdvrpb100_ortools20s.pt"),
        'mdvrpbl':    ("mdvrpbl100_n1000_seed1234.pt",  "data/mdvrpbl/mdvrpbl100_ortools20s.pt"),
        'mdvrpbltw':  ("mdvrpbltw100_n1000_seed1234.pt","data/mdvrpbltw/mdvrpbltw100_ortools20s.pt"),
        'mdvrpbtw':   ("mdvrpbtw100_n1000_seed1234.pt", "data/mdvrpbtw/mdvrpbtw100_ortools20s.pt"),
        'mdvrpl':     ("mdvrpl100_n1000_seed1234.pt",   "data/mdvrpl/mdvrpl100_ortools20s.pt"),
        'mdvrpltw':   ("mdvrpltw100_n1000_seed1234.pt", "data/mdvrpltw/mdvrpltw100_ortools20s.pt"),
        # 'mdvrpl': ("100.npz", "data/mdvrpl/100_sol_pyvrp.npz"),



        #amd problem
        'amdcvrptw': ("amdcvrptw100_n1000_seed1234.pt", "data/amdcvrptw/amdcvrptw100_ortools20s.pt"),
        'amdovrp': ("amdovrp100_n1000_seed1234.pt", "data/amdovrp/amdovrp100_ortools20s.pt"),
        'amdovrpb': ("amdovrpb100_n1000_seed1234.pt", "data/amdovrpb/amdovrpb100_ortools20s.pt"),
        'amdovrpbl': ("amdovrpbl100_n1000_seed1234.pt", "data/amdovrpbl/amdovrpbl100_ortools20s.pt"),
        'amdovrpbltw': ("amdovrpbltw100_n1000_seed1234.pt", "data/amdovrpbltw/amdovrpbltw100_ortools20s.pt"),
        'amdovrpbtw': ("amdovrpbtw100_n1000_seed1234.pt", "data/amdovrpbtw/amdovrpbtw100_ortools20s.pt"),
        'amdovrpl': ("amdovrpl100_n1000_seed1234.pt", "data/amdovrpl/amdovrpl100_ortools20s.pt"),
        'amdovrpltw': ("amdovrpltw100_n1000_seed1234.pt", "data/amdovrpltw/amdovrpltw100_ortools20s.pt"),
        'amdovrptw': ("amdovrptw100_n1000_seed1234.pt", "data/amdovrptw/amdovrptw100_ortools20s.pt"),

        'amdvrpb': ("amdvrpb100_n1000_seed1234.pt", "data/amdvrpb/amdvrpb100_ortools20s.pt"),
        'amdvrpbl': ("amdvrpbl100_n1000_seed1234.pt", "data/amdvrpbl/amdvrpbl100_ortools20s.pt"),
        'amdvrpbltw': ("amdvrpbltw100_n1000_seed1234.pt", "data/amdvrpbltw/amdvrpbltw100_ortools20s.pt"),
        'amdvrpbtw': ("amdvrpbtw100_n1000_seed1234.pt", "data/amdvrpbtw/amdvrpbtw100_ortools20s.pt"),
        'amdvrpl': ("amdvrpl100_n1000_seed1234.pt", "data/amdvrpl/amdvrpl100_ortools20s.pt"),
        'amdvrpltw': ("amdvrpltw100_n1000_seed1234.pt", "data/amdvrpltw/amdvrpltw100_ortools20s.pt"),


        #bp_problem
        'vrpbp': ("vrpbp100_n1000_seed1234.pt", "data/vrpbp/vrpbp100_ortools20s.pt"),
        'ovrpbp': ("ovrpbp100_n1000_seed1234.pt", "data/ovrpbp/ovrpbp100_ortools20s.pt"),
        'vrpbpl': ("vrpbpl100_n1000_seed1234.pt", "data/vrpbpl/vrpbpl100_ortools20s.pt"),
        'ovrpbptw': ("ovrpbptw100_n1000_seed1234.pt", "data/ovrpbptw/ovrpbptw100_ortools20s.pt"),
        'vrpbpltw': ("vrpbpltw100_n1000_seed1234.pt", "data/vrpbpltw/vrpbpltw100_ortools20s.pt"),
        'vrpbptw': ("vrpbptw100_n1000_seed1234.pt", "data/vrpbptw/vrpbptw100_ortools20s.pt"),
        'ovrpbpl': ("ovrpbpl100_n1000_seed1234.pt", "data/ovrpbpl/ovrpbpl100_ortools20s.pt"),
        'ovrpbpltw': ("ovrpbpltw100_n1000_seed1234.pt", "data/ovrpbpltw/ovrpbpltw100_ortools20s.pt"),

        'avrpbp': ("avrpbp100_n1000_seed1234.pt", "data/avrpbp/avrpbp100_ortools20s.pt"),
        'aovrpbp': ("aovrpbp100_n1000_seed1234.pt", "data/aovrpbp/aovrpbp100_ortools20s.pt"),
        'avrpbpl': ("avrpbpl100_n1000_seed1234.pt", "data/avrpbpl/avrpbpl100_ortools20s.pt"),
        'aovrpbptw': ("aovrpbptw100_n1000_seed1234.pt", "data/aovrpbptw/aovrpbptw100_ortools20s.pt"),
        'avrpbpltw': ("avrpbpltw100_n1000_seed1234.pt", "data/avrpbpltw/avrpbpltw100_ortools20s.pt"),
        'avrpbptw': ("avrpbptw100_n1000_seed1234.pt", "data/avrpbptw/avrpbptw100_ortools20s.pt"),
        'aovrpbpl': ("aovrpbpl100_n1000_seed1234.pt", "data/aovrpbpl/aovrpbpl100_ortools20s.pt"),
        'aovrpbpltw': ("aovrpbpltw100_n1000_seed1234.pt", "data/aovrpbpltw/aovrpbpltw100_ortools20s.pt"),

        'mdvrpbp': ("mdvrpbp100_n1000_seed1234.pt", "data/mdvrpbp/mdvrpbp100_ortools20s.pt"),
        'mdovrpbp': ("mdovrpbp100_n1000_seed1234.pt", "data/mdovrpbp/mdovrpbp100_ortools20s.pt"),
        'mdvrpbpl': ("mdvrpbpl100_n1000_seed1234.pt", "data/mdvrpbpl/mdvrpbpl100_ortools20s.pt"),
        'mdovrpbptw': ("mdovrpbptw100_n1000_seed1234.pt", "data/mdovrpbptw/mdovrpbptw100_ortools20s.pt"),
        'mdvrpbpltw': ("mdvrpbpltw100_n1000_seed1234.pt", "data/mdvrpbpltw/mdvrpbpltw100_ortools20s.pt"),
        'mdvrpbptw': ("mdvrpbptw100_n1000_seed1234.pt", "data/mdvrpbptw/mdvrpbptw100_ortools20s.pt"),
        'mdovrpbpl': ("mdovrpbpl100_n1000_seed1234.pt", "data/mdovrpbpl/mdovrpbpl100_ortools20s.pt"),
        'mdovrpbpltw': ("mdovrpbpltw100_n1000_seed1234.pt", "data/mdovrpbpltw/mdovrpbpltw100_ortools20s.pt"),

        'amdvrpbp': ("amdvrpbp100_n1000_seed1234.pt", "data/amdvrpbp/amdvrpbp100_ortools20s.pt"),
        'amdovrpbp': ("amdovrpbp100_n1000_seed1234.pt", "data/amdovrpbp/amdovrpbp100_ortools20s.pt"),
        'amdvrpbpl': ("amdvrpbpl100_n1000_seed1234.pt", "data/amdvrpbpl/amdvrpbpl100_ortools20s.pt"),
        'amdovrpbptw': ("amdovrpbptw100_n1000_seed1234.pt", "data/amdovrpbptw/amdovrpbptw100_ortools20s.pt"),
        'amdvrpbpltw': ("amdvrpbpltw100_n1000_seed1234.pt", "data/amdvrpbpltw/amdvrpbpltw100_ortools20s.pt"),
        'amdvrpbptw': ("amdvrpbptw100_n1000_seed1234.pt", "data/amdvrpbptw/amdvrpbptw100_ortools20s.pt"),
        'amdovrpbpl': ("amdovrpbpl100_n1000_seed1234.pt", "data/amdovrpbpl/amdovrpbpl100_ortools20s.pt"),
        'amdovrpbpltw': ("amdovrpbpltw100_n1000_seed1234.pt", "data/amdovrpbpltw/amdovrpbpltw100_ortools20s.pt"),


    }
    vrpmix_validation_set = {
        name: {
            "episodes_100": 1000,
            "filename_100": filename,
            "extra_args": {"solution_100": solution}
        }
        for name, (filename, solution) in vrpmix_files.items()
    }
    problem_validation_set.update(vrpmix_validation_set)

    return problem_validation_set


def get_problem_list(name):
    # CVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    vrpmix_list = ["cvrptw", "ovrp", "vrpl", "vrpb", "ovrptw", 'ovrpb', 'vrpbl', 'vrpltw', 'ovrpbtw', 'vrpbltw',
                   'ovrpl', 'vrpbtw', 'ovrpbl', 'ovrpltw', 'ovrpbltw',
                   "vrpbp", "ovrpbp", "vrpbpl", "ovrpbptw", "vrpbpltw", "vrpbptw", "ovrpbpl", "ovrpbpltw","cvrp"
                   ]

    # ACVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    avrpmix_list = ["acvrptw", "aovrp", "avrpl", "avrpb", "aovrptw", 'aovrpb', 'avrpbl', 'avrpltw', 'aovrpbtw',
                    'avrpbltw', 'aovrpl', 'avrpbtw', 'aovrpbl', 'aovrpltw', 'aovrpbltw',
                    "avrpbp", "aovrpbp", "avrpbpl", "aovrpbptw", "avrpbpltw", "avrpbptw", "aovrpbpl", "aovrpbpltw","acvrp"]

    # MDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    mdvrpmix_list = ["mdcvrptw", "mdovrp", "mdvrpl", "mdvrpb", "mdovrptw", 'mdovrpb', 'mdvrpbl', 'mdvrpltw', 'mdovrpbtw',
                    'mdvrpbltw', 'mdovrpl', 'mdvrpbtw', 'mdovrpbl', 'mdovrpltw', 'mdovrpbltw',
                     "mdvrpbp", "mdovrpbp", "mdvrpbpl", "mdovrpbptw", "mdvrpbpltw", "mdvrpbptw", "mdovrpbpl", "mdovrpbpltw","mdcvrp"]

    # AMDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems
    amdvrpmix_list = ["amdcvrptw", "amdovrp", "amdvrpl", "amdvrpb", "amdovrptw", 'amdovrpb', 'amdvrpbl', 'amdvrpltw', 'amdovrpbtw',
                    'amdvrpbltw', 'amdovrpl', 'amdvrpbtw', 'amdovrpbl', 'amdovrpltw', 'amdovrpbltw',
                      "amdvrpbp", "amdovrpbp", "amdvrpbpl", "amdovrpbptw", "amdvrpbpltw", "amdvrpbptw", "amdovrpbpl",
                      "amdovrpbpltw","amdcvrp"]

    # PD problems
    pdcvrp_list = ['pdcvrp','apdcvrp','aopdcvrp','opdcvrp']

    #  remaining problem with one depot
    remain_problem_with_one_depot = ['pctsp','spctsp','op','pdp','apdp']

    # tsp
    tsp_and_atsp = ['tsp','atsp']

    # 32 b constraints problems
    constraint_b_list = ['vrpb', 'ovrpb', 'vrpbl', 'ovrpbtw', 'vrpbltw', 'vrpbtw', 'ovrpbl', 'ovrpbltw',
     'avrpb', 'aovrpb', 'avrpbl', 'aovrpbtw', 'avrpbltw', 'avrpbtw', 'aovrpbl', 'aovrpbltw',
     'mdvrpb', 'mdovrpb', 'mdvrpbl', 'mdovrpbtw', 'mdvrpbltw', 'mdvrpbtw', 'mdovrpbl', 'mdovrpbltw',
     'amdvrpb', 'amdovrpb', 'amdvrpbl', 'amdovrpbtw', 'amdvrpbltw', 'amdvrpbtw', 'amdovrpbl', 'amdovrpbltw',]

    # A problems
    A_list = ['atsp','apdp','apdcvrp','aopdcvrp']
    A_list = A_list + avrpmix_list + amdvrpmix_list


    all_vrpmix_list = vrpmix_list + avrpmix_list + mdvrpmix_list+ amdvrpmix_list

    single_depot_list = vrpmix_list+avrpmix_list+remain_problem_with_one_depot+pdcvrp_list


    multi_depot_list = mdvrpmix_list + amdvrpmix_list

    train_problem_list = ['atsp','tsp','op','pctsp','pdp','acvrp','cvrp','cvrptw','vrpb','ovrp','ovrptw']

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