# -*- coding: utf-8 -*-

import numpy as np
import os

def read_tsp_file(tsp_file):
    lines = map(lambda x:x.strip(), open(tsp_file).readlines())
    start_idx = [i for i,s in enumerate(lines) if s == "NODE_COORD_SECTION"][0]
    points = [map(float, s.split()[1:]) for s in lines[(start_idx+1):-1] ]
    return np.asarray(points, dtype=float)


def length(x, y):
    return np.linalg.norm(np.asarray(x) - np.asarray(y))

def get_path_distance(tsp_points, tsp_path):
    distance = 0
    path_len = tsp_points.shape[0]
    for i in range(path_len):
        distance += length(tsp_points[tsp_path[i]], tsp_points[tsp_path[(i+1)%path_len]])
    return distance

def read_sol_file(sol_file):
    lines = map(lambda x:x.strip(), open(sol_file).readlines())
    sol = [idx for s in lines[1:] for idx in map(int, s.split())]
    return np.asarray(sol, dtype=int)

concord_tl = \
"""NAME: gen%(dim)d
TYPE: TSP
COMMENT: gen_tsp
DIMENSION: %(dim)d
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
%(points)s
EOF
"""
def save_condord_tsp(tsp_file, points):
    if 'int' not in str(points.dtype):
        points = np.asarray(points/np.max(points) * 1e8, dtype=np.int64)
    dim = points.shape[0]
    points = "\n".join([str(i+1) +" " + str(points[i])[1:-1] for i in range(dim)])
    open(tsp_file, 'w').write(concord_tl % {"dim": dim, "points": points})

def solve(points, tmp_dir="/tmp/concord"):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tsp_file_name = "gentsp%d.tsp" % points.shape[0]
    solution_file_name = "gentsp%d.sol" % points.shape[0]
    tsp_file = os.path.join(tmp_dir, tsp_file_name)
    save_condord_tsp(tsp_file, points)
    # 给定足够的时间，对int问题concorde总是给出最优解？
    os.system('cd '+tmp_dir+" && concorde " + tsp_file_name + "> /dev/null 2>/dev/null")
    sol = read_sol_file(os.path.join(tmp_dir, solution_file_name))
    return sol

if __name__ == "__main__":
    pass
    # points = read_tsp_file('tmp/gentsp60.tsp')
    # ssol = read_sol_file('tmp/succ.sol')
    # fsol = read_sol_file('tmp/fail.sol')
    # print get_path_distance(points, ssol)
    # print get_path_distance(points, fsol)

    # for i in range(10):
    #     print i
    #     points = np.random.rand(20,2)
        # print solve(points)