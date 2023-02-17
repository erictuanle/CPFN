import numpy as np

from SPFN import plane_fitter, sphere_fitter, cylinder_fitter, cone_fitter

primitive_name_to_id_dict = {}

def primitive_name_to_id(name):
    return primitive_name_to_id_dict[name]

def get_n_registered_primitives():
    return len(primitive_name_to_id_dict)

def register_primitives(primitive_name_list):
    # Must be called once before everything
    global primitive_name_to_id_dict
    primitive_name_to_id_dict = {}
    for idx, name in enumerate(primitive_name_list):
        primitive_name_to_id_dict[name] = idx
    print('Registered ' + ','.join(primitive_name_list))

def create_primitive_from_dict(d):
    if d['type'] == 'plane':
        return plane_fitter.create_primitive_from_dict(d)
    elif d['type'] == 'sphere':
        return sphere_fitter.create_primitive_from_dict(d)
    elif d['type'] == 'cylinder':
        return cylinder_fitter.create_primitive_from_dict(d)
    elif d['type'] == 'cone':
        return cone_fitter.create_primitive_from_dict(d)
    else:
        raise NotImplementedError