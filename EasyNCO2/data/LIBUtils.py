import os
import numpy as np
import itertools


def TSPLIBWriter(directory, node_coords, int_coord_scale, ptype, par_args):
    '''
        Generates a TSPLIB-formatted file for a TSP problem.
        The file includes the problem's name, comment, type, dimension, edge weight type, and node coordinates.
        Parameters:
        - directory: The directory where the file will be saved.
        - node_coords: A list of node coordinates.
        - user_comment: A user-provided comment.
        - ptype: The type of problem, defaults to 'TSP'.
        - scale: The scaling factor for coordinates, defaults to 100000.
    '''
    to_int_coord = lambda arr: (np.array(arr) * int_coord_scale + 0.5).astype(int)

    problem_path = os.path.join(directory, 'problem.tsp')

    with open(problem_path, 'w') as f:
        f.write("\n".join(
            f"{k} : {v}" for k, v in (
                ("NAME", "problem"),
                ("COMMENT", "None"),
                ("TYPE", ptype),
                ("DIMENSION", len(node_coords)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D")
            )
        ))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join(
            f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate(itertools.chain(to_int_coord(node_coords)))
        ))
        f.write("\n")
        f.write("EOF\n")

def ATSPLIBWriter(directory, node_matrix, int_matrix_scale, ptype, par_args):
    to_int_matrix = lambda arr: (np.array(arr) * int_matrix_scale + 0.5).astype(int)

    problem_path = os.path.join(directory, 'problem.atsp')
    with open(problem_path, 'w') as f:
        f.write("\n".join(
            f"{k} : {v}" for k, v in (
                ("NAME", "problem"),
                ("COMMENT", "None"),
                ("TYPE", ptype),
                ("DIMENSION", len(node_matrix)),
                ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            )
        ))
        f.write("\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        # f.write("\n".join(
        #     f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate(itertools.chain(to_int_coord(node_coords)))
        # ))
        node_matrix = to_int_matrix(node_matrix)
        for i in range(0, len(node_matrix)):
            for j in range(0, len(node_matrix)):
                if(i == j):
                    node_matrix[i][j] = 9999
                f.write(str(node_matrix[i][j]))
                f.write("\t")
            f.write("\n")
        f.write("EOF\n")

def CVRPLIBWriter(directory, depot, loc, demand, capacity, int_coord_scale, demand_scale, ptype, pkwargs):
    '''
        Generates a TSPLIB-formatted file for a CVRP problem.
        The file includes the problem's name, comment, type, dimension, edge weight type, node coordinates, capacity, and demands.
        Parameters:
        - directory: The directory where the file will be saved.
        - depot: The coordinates of the depot.
        - loc: A list of node coordinates.
        - demand: A list of node demands.
        - capacity: The capacity of the vehicle.
        - demand_scale: The scaling factor for demands.
        - ptype: The type of problem.
        - pkwargs: Additional keyword arguments for the problem.
        - grid_size: The size of the grid for scaling coordinates.
        - name: The name of the problem, defaults to 'problem'.
        - scale: The scaling factor for coordinates, defaults to 100000.
    '''
    to_int_coord = lambda arr: (np.array(arr) * int_coord_scale + 0.5).astype(int)
    to_int_demand = lambda arr: (np.array(arr) * demand_scale + 0.5).astype(int)

    problem_path = os.path.join(directory, 'problem.vrp')
    param_path = os.path.join(directory, 'params.par')
    output_path = os.path.join(directory, 'output.tour')

    with open(problem_path, 'w') as f:
        f.write("\n".join(
            f"{k} : {v}" for k, v in (
                ("NAME", "problem"),
                ("COMMENT", "None"),
                ("TYPE", ptype),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ))
        if ptype == "CVRPTW":
            f.write("\n")
            f.write(f"SERVICE_TIME : {to_int_coord(pkwargs['service_time'])}")
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join(
            f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate(itertools.chain([to_int_coord(depot)], to_int_coord(loc)))
        ))
        if ptype == "VRPMPD":
            f.write("\n")
            f.write("PICKUP_AND_DELIVERY_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t0\t0\t10000000\t0\t{d * p}\t{d * (not p)}"
                for i, (d, p) in enumerate(zip(itertools.chain([0], demand), pkwargs['is_pickup']))
            ))
        else:
            f.write("\n")
            f.write("DEMAND_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t{d}" for i, d in enumerate(itertools.chain([0], to_int_demand(demand)))
            ))
        if ptype == "CVRPTW":
            f.write("\n")
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate(to_int_coord(pkwargs['window']))
            ))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def TSPLIBReader(filename):
    '''
        Acquire description of a TSP problem from a TSPLIB-formatted file
        Parameters:
        - filename: the name of the TSPLIB-formatted file. e.g."D:\\desktooop\\Code\\EasyNCO\\EasyNCO\\1204\\results_lkh_tsp\\0\\problem.tsp"
        Returns:
        - name: the name of the TSP problem.
        - dimension: the number of nodes in the TSP problem. (int)
        - locs: the coordinates of nodes in the TSP problem. e.g.[[31020, 8718], [85228, 81588], [28825, 6767], [9301, 86527]]
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started = False
        locs = []

        for line in f:
            loc = []
            if started:
                if line.startswith("EOF"):
                    break
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)
            if line.startswith("NAME"):
                name = line.strip().split(" ")[-1]
            if line.startswith("DIMENSION"):
                dimension = int(line.strip().split(" ")[-1])
            if line.startswith("EDGE_WEIGHT_TYPE"):
                if line.strip().split(" ")[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None
                else:
                    edge_weight_type = line.strip().split(" ")[-1]
            if line.startswith("NODE_COORD_SECTION"):
                started = True

    assert len(locs) == dimension
    return name, dimension, locs, edge_weight_type

def ATSPLIBReader(filename):
    '''
        Acquire description of a TSP problem from a TSPLIB-formatted file
        Parameters:
        - filename: the name of the TSPLIB-formatted file. e.g."D:\\desktooop\\Code\\EasyNCO\\EasyNCO\\1204\\results_lkh_tsp\\0\\problem.tsp"
        Returns:
        - name: the name of the TSP problem.
        - dimension: the number of nodes in the TSP problem. (int)
        - locs: the coordinates of nodes in the TSP problem. e.g.[[31020, 8718], [85228, 81588], [28825, 6767], [9301, 86527]]
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started = False
        locs = []

        for line in f:
            loc = []
            if started:
                if line.startswith("EOF"):
                    break
                loc.append(line.strip().split("\t"))
                locs.append(loc)
            if line.startswith("NAME"):
                name = line.strip().split(" ")[-1]
            if line.startswith("DIMENSION"):
                dimension = int(line.strip().split(" ")[-1])
            if line.startswith("EDGE_WEIGHT_SECTION"):
                started = True

    assert len(locs) == dimension
    return name, dimension, locs

def CVRPLIBReader(filename):
    '''
        Acquire description of a CVRP problem from a CVRPLIB-formatted file
        Parameters:
        - filename: the name of the CVRPLIB-formatted file. e.g."D:\\desktooop\\Code\\EasyNCO\\EasyNCO\\1204\\results_lkh_cvrp\\0\\problem.vrp"
        Returns:
        - name: the name of the CVRP problem.
        - dimension: the number of nodes in the CVRP problem. (int)
        - locs: the coordinates of nodes in the CVRP problem. e.g.[[31020, 8718], [85228, 81588], [28825, 6767], [9301, 86527]]
        - demand: A list of node demands. e.g.[0, 9, 2, 2, 5, 3, 9, 4, 8, 1, 8, 3, 7, 5, 8, 6, 3, 9, 5, 6, 8]
        - capacity: The capacity of the vehicle. (int)
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started_node = False
        started_demand = False
        locs = []
        demand = []
        for line in f:
            loc = []
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(int(line.strip().split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
            if started_node:
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)

            if line.startswith("NAME"):
                name = line.strip().split(" ")[-1]
            if line.startswith("DIMENSION"):
                dimension = float(line.strip().split(" ")[-1]) - 1 # depot is not counted
            if line.startswith("CAPACITY"):
                capacity = float(line.strip().split(" ")[-1])
            if line.startswith("NODE_COORD_SECTION"):
                started_node = True
    cost_file = filename.replace('.vrp', '.sol')
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            for line in f:
                if line.startswith("Cost"):
                    cost = float(line.split()[1])
    else:
        cost = None

    assert len(locs) == dimension + 1  # +1 for depot
    return name, dimension, locs, demand, capacity, cost


if __name__ == "__main__":

    # test  TSPLIBReader
    # filename = "D:\\desktooop\\Code\\EasyNCO\\EasyNCO\\1204\\results_lkh_tsp\\0\\problem.tsp"
    # name, dimension, locs = TSPLIBReader(filename)
    # print(name)
    # print(dimension)
    # print(locs)

    # test  CVRPLIBReader
    filename = "D:\\desktooop\\Code\\EasyNCO\\EasyNCO\\1204\\results_lkh_cvrp\\0\\problem.vrp"
    name, dimension, locs, demand, capacity = CVRPLIBReader(filename)
    print(name)
    print(dimension)
    print(locs)
    print(demand)
    print(capacity)


# 最优值 invit data_utils


tsplib_cost = {
    "a280": 2579,
    "ali535": 202339,
    "att48": 10628,
    "att532": 27686,
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    "brazil58": 25395,
    "brd14051": 469385,
    "brg180": 1950,
    "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    "dantzig42": 699,
    "dsj1000": 18660188,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    "fri26": 937,
    "gil262": 2378,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "gr48": 5046,
    "gr96": 55209,
    "gr120": 6942,
    "gr137": 69853,
    "gr202": 40160,
    "gr229": 134602,
    "gr431": 171414,
    "gr666": 294358,
    "hk48": 11461,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728,
    "pla33810": 66048945,
    "pla85900": 142382641,
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    "si175": 21407,
    "si535": 48450,
    "si1032": 92650,
    "st70": 675,
    "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556,
}


#best know solutions in
#Behnke D, Geiger M J. Test instances for the flexible job shop scheduling problem with work centers[J]. 2012.
brandimarte_cost = {
    "Mk01": 39,
    "Mk02": 26,
    "Mk03": 204,
    "Mk04": 60,
    "Mk05": 172,
    "Mk06": 58,
    "Mk07": 139,
    "Mk08": 523,
    "Mk09": 307,
    "Mk10": 197
}

#la data include edata,rdata,vdata
e_la_cost = {
    "e_la01": 609,
    "e_la02": 655,
    "e_la03": 550,
    "e_la04": 568,
    "e_la05": 503,
    "e_la06": 833,
    "e_la07": 762,
    "e_la08": 845,
    "e_la09": 878,
    "e_la10": 866,
    "e_la11": 1103,
    "e_la12": 960,
    "e_la13": 1053,
    "e_la14": 1123,
    "e_la15": 1111,
    "e_la16": 892,
    "e_la17": 707,
    "e_la18": 842,
    "e_la19": 796,
    "e_la20": 857,
    "e_la21": 1017,
    "e_la22": 882,
    "e_la23": 950,
    "e_la24": 909,
    "e_la25": 941,
    "e_la26": 1125,
    "e_la27": 1186,
    "e_la28": 1149,
    "e_la29": 1118,
    "e_la30": 1204,
    "e_la31": 1539,
    "e_la32": 1698,
    "e_la33": 1547,
    "e_la34": 1604,
    "e_la35": 1736,
    "e_la36": 1162,
    "e_la37": 1397,
    "e_la38": 1144,
    "e_la39": 1184,
    "e_la40": 1150
}

r_la_cost = {
    "r_la01": 571,
    "r_la02": 530,
    "r_la03": 478,
    "r_la04": 502,
    "r_la05": 457,
    "r_la06": 799,
    "r_la07": 750,
    "r_la08": 765,
    "r_la09": 853,
    "r_la10": 804,
    "r_la11": 1071,
    "r_la12": 936,
    "r_la13": 1038,
    "r_la14": 1070,
    "r_la15": 1090,
    "r_la16": 717,
    "r_la17": 646,
    "r_la18": 666,
    "r_la19": 700,
    "r_la20": 756,
    "r_la21": 835,
    "r_la22": 760,
    "r_la23": 842,
    "r_la24": 808,
    "r_la25": 791,
    "r_la26": 1061,
    "r_la27": 1091,
    "r_la28": 1080,
    "r_la29": 998,
    "r_la30": 1078,
    "r_la31": 1521,
    "r_la32": 1659,
    "r_la33": 1499,
    "r_la34": 1536,
    "r_la35": 1550,
    "r_la36": 1030,
    "r_la37": 1077,
    "r_la38": 962,
    "r_la39": 1024,
    "r_la40": 970
}

v_la_cost = {
    "v_la01": 570,
    "v_la02": 529,
    "v_la03": 477,
    "v_la04": 502,
    "v_la05": 457,
    "v_la06": 799,
    "v_la07": 749,
    "v_la08": 765,
    "v_la09": 853,
    "v_la10": 804,
    "v_la11": 1071,
    "v_la12": 936,
    "v_la13": 1038,
    "v_la14": 1070,
    "v_la15": 1089,
    "v_la16": 717,
    "v_la17": 646,
    "v_la18": 663,
    "v_la19": 617,
    "v_la20": 756,
    "v_la21": 806,
    "v_la22": 739,
    "v_la23": 815,
    "v_la24": 777,
    "v_la25": 756,
    "v_la26": 1054,
    "v_la27": 1085,
    "v_la28": 1070,
    "v_la29": 994,
    "v_la30": 1069,
    "v_la31": 1520,
    "v_la32": 1658,
    "v_la33": 1497,
    "v_la34": 1535,
    "v_la35": 1549,
    "v_la36": 948,
    "v_la37": 986,
    "v_la38": 943,
    "v_la39": 922,
    "v_la40": 955
}


#jssp
jssp_result = {
    "abz10x10": [1234.,  943.],
    "abz20x15": [656., 665., 678.],
    "ft10x10": [930.],
    "ft20x5": [1165.],
    "ft6x6": [55.],
    "la10x10": [945., 784., 848., 842., 902.],
    "la10x5": [666., 655., 597., 590., 593.],
    "la15x10": [1046.,  927., 1032.,  935.,  977.],
    "la15x15": [1268., 1397., 1196., 1233., 1222.],
    "la15x5": [926., 890., 863., 951., 958.],
    "la20x10": [1218., 1235., 1216., 1152., 1355.],
    "la20x5": [1222., 1039., 1150., 1292., 1207.],
    "la30x10": [1784., 1850., 1719., 1721., 1888.],
    "orb10x10": [1059.,  888., 1005., 1005.,  887., 1010.,  397.,  899.,  934.,
        944.],
    "swv20x10": [1407., 1475., 1398., 1464., 1424.],
    "swv20x15": [1671., 1594., 1752., 1655., 1743.],
    "swv50x10": [2983., 2977., 3104., 2968., 2885., 2924., 2794., 2852., 2843.,
       2823.],
    "tai100x20": [5464., 5181., 5568., 5339., 5392., 5342., 5436., 5394., 5358.,
       5183.],
    "tai15x15": [1231., 1244., 1218., 1175., 1224., 1238., 1227., 1217., 1274.,
       1241.],
    "tai20x15": [1357., 1367., 1342., 1345., 1345., 1360., 1462., 1396., 1332.,
       1348.],
    "tai20x20": [1642., 1600., 1557., 1644., 1595., 1645., 1680., 1603., 1625.,
       1584.],
    "tai30x15": [1764., 1784., 1791., 1828., 2007., 1819., 1771., 1673., 1795.,
       1670.],
    "tai30x20": [2006., 1939., 1846., 1979., 2000., 2006., 1889., 1937., 1963.,
       1923.],
    "tai50x15": [2760., 2756., 2717., 2839., 2679., 2781., 2943., 2885., 2655.,
       2723.],
    "tai50x20": [2868., 2869., 2755., 2702., 2725., 2845., 2825., 2784., 3071.,
       2995.],
    "yn20x20": [884., 904., 892., 968.]
}