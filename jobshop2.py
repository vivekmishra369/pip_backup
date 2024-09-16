import gurobipy as gp
from gurobipy import GRB

def flexible_jobshop_gurobi():
    # Data for the job shop problem
    jobs = [
        [[(3, 0), (1, 1), (5, 2)], [(2, 0), (4, 1), (6, 2)], [(2, 0), (3, 1), (1, 2)]],
        [[(2, 0), (3, 1), (4, 2)], [(1, 0), (5, 1), (4, 2)], [(2, 0), (1, 1), (4, 2)]],
        [[(2, 0), (1, 1), (4, 2)], [(2, 0), (3, 1), (4, 2)], [(3, 0), (1, 1), (5, 2)]],
    ]

    num_jobs = len(jobs)
    num_machines = 3

    # Model
    model = gp.Model('FlexibleJobShop')

    # Horizon (an upper bound for the makespan)
    # horizon = sum(max(task[0] for task in job) for job in jobs)
    horizon = sum([max([task[0] for task in job]) [0]for job in jobs])

    # Decision Variables
    x = {}
    for j in range(num_jobs):
        for t, task in enumerate(jobs[j]):
            for alt_id, (duration, machine) in enumerate(task):
                x[j, t, machine] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}_{machine}")

    # s[j,t] is the start time of task t of job j
    s = {}
    for j in range(num_jobs):
        for t in range(len(jobs[j])):
            s[j, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=horizon, name=f"s_{j}_{t}")

    # Makespan variable
    makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=horizon, name='makespan')

    # Constraints
    for j in range(num_jobs):
        for t, task in enumerate(jobs[j]):
            model.addConstr(gp.quicksum(x[j, t, machine] for _, machine in task) == 1)

    for j in range(num_jobs):
        for t in range(1, len(jobs[j])):
            task_duration = gp.quicksum(x[j, t-1, machine] * jobs[j][t-1][alt_id][0]
                                        for alt_id, (duration, machine) in enumerate(jobs[j][t-1]))
            model.addConstr(s[j, t] >= s[j, t-1] + task_duration)

    # Machine capacity constraints: no two tasks can overlap on the same machine
    big_M = horizon  # A sufficiently large number representing the horizon
    for machine in range(num_machines):
        for j1 in range(num_jobs):
            for t1, task1 in enumerate(jobs[j1]):
                for j2 in range(num_jobs):
                    if j1 == j2:
                        continue
                    for t2, task2 in enumerate(jobs[j2]):
                        for alt_id1, (duration1, machine1) in enumerate(task1):
                            for alt_id2, (duration2, machine2) in enumerate(task2):
                                if machine1 == machine2:
                                    z = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{t1}_{j2}_{t2}_{machine1}")

                                    # Case 1: Task 1 finishes before Task 2 starts
                                    model.addConstr(s[j1, t1] + duration1 <= s[j2, t2] + big_M * (1 - z))

                                    # Case 2: Task 2 finishes before Task 1 starts
                                    model.addConstr(s[j2, t2] + duration2 <= s[j1, t1] + big_M * z)

    # Minimize makespan
    for j in range(num_jobs):
        model.addConstr(makespan >= gp.quicksum(x[j, t, machine] * (s[j, t] + jobs[j][t][alt_id][0])
                                                for t in range(len(jobs[j]))
                                                for alt_id, (duration, machine) in enumerate(jobs[j][t])))
    model.setObjective(makespan, GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    # Output the results
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal makespan: {makespan.x}\n")
        for j in range(num_jobs):
            print(f"Job {j}:")
            for t, task in enumerate(jobs[j]):
                start_time = s[j, t].x
                assigned_machine = None
                task_duration = None
                for alt_id, (duration, machine) in enumerate(task):
                    if x[j, t, machine].x > 0.5:
                        assigned_machine = machine
                        task_duration = duration
                        break
                print(f"  Task {t}: Start = {start_time}, Machine = {assigned_machine}, Duration = {task_duration}")
    else:
        print("No optimal solution found.")

flexible_jobshop_gurobi()
