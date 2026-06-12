def plus_minus(iter, u):
    l1, l2 = [], []
    for el in iter:
        l1.append(el + u)
        l2.append(el - u)
    return l1, l2

def combine_concat(l1, l2):
    result = []
    subsets_l1 = get_subsets(l1)
    subsets_l2 = get_subsets(l2)
    for el in subsets_l1:
        for el2 in subsets_l2:
            result.append(el + el2)
    return result

def get_subsets(l1):
    result = []
    stop = len(l1)
    for i in range(stop - 1, -1, -1): # i = 3
        for k in range(abs(i-(stop-1)) + 1): # 3 - 4 = 1 --> [0, 1]
            sub_left = l1[k:i+k+1] # von 0 bis 3 und von 1 bis 3, komplett
            concat_single = "".join(sub_left)
            result.append(concat_single)
    return result

import random
import string
random.seed(42)

def id_generator(size=4, chars=string.ascii_letters + string.digits):
    return "".join(random.choice(chars) for _ in range(size))

def init_rnd_dict(n):
    rand_dict = {}
    for i in range(n):
        key = id_generator()
        value = random.random() * 100
        rand_dict[key] = value
    return rand_dict

def quadriere_dict(dictionary : dict):
    dict_copy = dictionary.copy()
    for k in dict_copy.keys():
        # print(v)
        # print(type(v))
    
        try:
            if isinstance(dict_copy[k], float):
                dict_copy[k]**=2
            else:
                raise Exception("Value was not a float")
        except Exception as e:
            print(e.args)
    return dict_copy

def selection_sort(arr):
    arr = arr.copy()
    end = len(arr)
    
    for i in range(end):
        max_val = max(arr[0:end-i])
        max_val_idx = list.index(arr, max_val)
        
        temp = arr[end-i-1]
        arr[end-i-1] = max_val
        arr[max_val_idx] = temp
    
    return arr


        
import numpy as np

def calc_angle(v1, v2):
    product1 = np.inner(v1, v2)
    length_mult = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle1_rad = np.arccos(product1/length_mult)
    angle1 = np.rad2deg(angle1_rad)
    return angle1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_vecs(v1, v2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    base = [0, 0, 0]
    # Plot both vectors
    
    # * (asterix) for unpacking positional arguments, replaces 0, 0, 0, etc. here
    # opposite = *kargs --> packs arguments into tuple
    # or **kwargs --> packs arguments into a dict
    ax.quiver(*base, *v1, color='r', label='v1')
    ax.quiver(*base, *v2, color='b', label='v2')
    # Set axis limits to fit both vectors
    all_coords = np.array([v1, v2, base])
    max_range = np.max(np.abs(all_coords)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    # Add legend
    ax.legend()
    plt.show()
    
def find_orth(v1):
    # 2 * x + 5 * y - 3 * z = 0?
    v2 = [1, 0, 0] if not np.allclose(np.cross(v1, [1, 0, 0]), 0) else [0, 1, 0] # nehme einen random vektor, wenn der parallel ist, einen
    return np.cross(v1, v2)

    

if __name__ == "__main__":
    # l1, l2 = plus_minus([1,2,3,4], 10.)
    # print(l1, l2)
    # combinations = combine_concat(["h", "e", "l", "l", "o"], ["w", "r", "d"])
    # print(combinations)
    
    # dict1 = init_rnd_dict(20)
    # print(dict1)
    # dict_quadriert = quadriere_dict(dict1)
    # print(dict_quadriert)
    
    # arr = [8,7,6,5,4,3,3,2,1]
    # arr_sorted = selection_sort(arr)
    # print(arr_sorted)
    
    v1 = [3, -2, 2]
    v2 = [1, 2, -1.5]
    
    # print(calc_angle(v1, v2))
    # print(calc_angle([2, 0, 5, 1], [-1, 2, 0, 1]))
    
    # plot_vecs(v1, v2)
    
    v3 = [2, 5, -3]
    orth_vec = find_orth(v3)
    plot_vecs(v3,orth_vec)

    

            
        
    