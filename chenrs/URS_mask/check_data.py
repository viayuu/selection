import torch


problem_list = ['cvrptw','vrpb','ovrptw','ovrp']
problem_size_list = [1000,2000,3000,4000,5000]

for problem in problem_list:
    for problem_size in problem_size_list:
        if problem_size > 1000:
            capacity = 300
        else:
            capacity = 200
        print("#####################")
        print(f"{problem}{problem_size}")
        path =  f"/home/zhengyp/ych/AAA/data/{problem}/{problem}{problem_size}_C{capacity}_n16_seed1234_pyvrp1200s.pt"
        data = torch.load(path)
        solution = data['solution']
        cost = data['cost']
        print(f"solution.shape:{solution.shape}")
        print(f"cost:{cost}")
        print("#####################")
