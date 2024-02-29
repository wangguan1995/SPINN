import argparse
import os
import time

import jax
import numpy as np
import optax
from networks.hessian_vector_products import *
from tqdm import trange
from utils.data_generators import generate_test_data, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *
from utils.vorticity import vorx, vory, vorz
from utils.visualizer import show_solution
import flax
import pickle
import matplotlib.pyplot as plt


def save_weights(weights, filename):
    bytes_output=flax.serialization.to_bytes(target=weights)
    pickle.dump(bytes_output,open(filename,"wb"))


def load_weights(weights, filename):
    pkl_file=pickle.load(open(filename,"rb"))
    tained_weights=flax.serialization.from_bytes(target=weights,encoded_bytes=pkl_file)
    return tained_weights


def load_vtu_from_mesh(dir):
    import meshio
    mesh = meshio.read(dir)
    n = mesh.points.shape[0]
    input_keys = ("x", "y", "z")
    mesh.points = mesh.points.astype(np.float32)
    input_dict = {}

    for i, key in enumerate(input_keys):
        mesh.points[:, i] = mesh.points[:, i]
        input_dict[key] = mesh.points[:, i].reshape(n, 1)

    output_keys = ("u", "v", "w", "p", "e", "c_p", "vorticity_vector")

    for i, key in enumerate(output_keys):
        input_dict[key] = mesh.point_data[key].reshape(n, -1)

    return input_dict, mesh


def plot_u(time, x_pos, epoch, var, plane_normal):
    ic_mesh = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000+100*time}.vtu"
    input_dict, _ = load_vtu_from_mesh(ic_mesh)
    var_ref, var_pred = {}, {}
    var_ref["ux_ref"] = input_dict["u"].reshape(32,32,32)[jnp.newaxis, :]
    var_ref["uy_ref"] = input_dict["v"].reshape(32,32,32)[jnp.newaxis, :]
    var_ref["uz_ref"] = input_dict["w"].reshape(32,32,32)[jnp.newaxis, :]
    var_ref["wx_ref"] = input_dict["vorticity_vector"][:,0].reshape(32,32,32)[jnp.newaxis, :]
    var_ref["wy_ref"] = input_dict["vorticity_vector"][:,1].reshape(32,32,32)[jnp.newaxis, :]
    var_ref["wz_ref"] = input_dict["vorticity_vector"][:,2].reshape(32,32,32)[jnp.newaxis, :]

    var_pred[f"{var}_pred"] = np.load(f"./{var}_{epoch}.npy")
    if plane_normal == 'x':
        ref = var_ref[var + "_ref"][0][x_pos]
        pred = var_pred[var + "_pred"][time][x_pos]
    elif plane_normal == 'y':
        ref = var_ref[var + "_ref"][0][:][x_pos]
        pred = var_pred[var + "_pred"][time][:][x_pos]
    elif plane_normal == 'z':
        ref = var_ref[var + "_ref"][0][:][:][x_pos]
        pred = var_pred[var + "_pred"][time][:][:][x_pos]
    else:
        raise
        

    os.makedirs(os.path.join("./", f'vis/'), exist_ok=True)
    fig = plt.figure(figsize=(14, 5))

    # reference
    ax1 = fig.add_subplot(131)
    
    ax1.imshow(ref, cmap='jet', vmin=jnp.min(ref), vmax=jnp.max(ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Reference {var}(t={time:d} / 32, x={x_pos:d}) / 32', fontsize=15)

    # predicted
    ax1 = fig.add_subplot(132)
    
    ax1.imshow(pred, cmap='jet', vmin=jnp.min(pred), vmax=jnp.max(pred))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Predicted {var}(t={time:d} / 32, x={x_pos:d} / 32)', fontsize=15)
    
    # error
    ax1 = fig.add_subplot(133)
    err = np.abs(ref - pred)
    ax1.imshow(err, cmap='jet', vmin=jnp.min(err), vmax=jnp.max(err))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Error {var}(t={time:d} / 32, x={x_pos:d} / 32)', fontsize=15)
    plt.show()
    return err, np.abs(ref)


def generate_train_data_cylinder(step_idx):
    ic_mesh = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000 + 100*step_idx}.vtu"
    print("loading ic mesh : ", ic_mesh)
    input_dict, mesh = load_vtu_from_mesh(ic_mesh)
    # box_size = 32 * 32 * 32 
    nc = 32 
    time_step = 5
    # [pde] points
    # 时间步长是0.01s
    tc = jnp.linspace(0.01,0.33,time_step).reshape(-1, 1)
    xc = np.unique(input_dict['x']).reshape(-1, 1) - input_dict['x'].min()
    yc = np.unique(input_dict['y']).reshape(-1, 1) - input_dict['y'].min()
    zc = np.unique(input_dict['z']).reshape(-1, 1) - input_dict['z'].min()

    # [initial] points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    zi = zc
    
    if step_idx == 0: #causal pinn
        wi_x = input_dict["vorticity_vector"][:,0].reshape(32,32,32)[jnp.newaxis, :]
        wi_y = input_dict["vorticity_vector"][:,1].reshape(32,32,32)[jnp.newaxis, :]
        wi_z = input_dict["vorticity_vector"][:,2].reshape(32,32,32)[jnp.newaxis, :]
        wi = [wi_x, wi_y, wi_z]
        ui_x = input_dict["u"].reshape(32,32,32)[jnp.newaxis, :]
        ui_y = input_dict["v"].reshape(32,32,32)[jnp.newaxis, :]
        ui_z = input_dict["w"].reshape(32,32,32)[jnp.newaxis, :]
        ui = [ui_x, ui_y, ui_z]
    else:
        epoch = 10000
        wi_x = np.load(f"./wx_{epoch}.npy")
        wi_y = np.load(f"./wy_{epoch}.npy")
        wi_z = np.load(f"./wz_{epoch}.npy")
        wi = [wi_x, wi_y, wi_z]
        
        ui_x = np.load(f"./ux_{epoch}.npy")
        ui_y = np.load(f"./ux_{epoch}.npy")
        ui_z = np.load(f"./ux_{epoch}.npy")
        ui = [ui_x, ui_y, ui_z]

    # [boundary] points
    x_max = jnp.array([[input_dict['x'].max() - input_dict['x'].min()]])
    y_max = jnp.array([[input_dict['y'].max() - input_dict['y'].min()]])
    z_max = jnp.array([[input_dict['z'].max() - input_dict['z'].min()]])

    tb = [tc, tc, tc, tc, tc, tc]
    xb = [jnp.array([[0.]]), x_max, xc, xc, xc, xc]
    yb = [yc, yc, jnp.array([[0.]]), y_max, yc, yc]
    zb = [zc, zc, zc, zc, jnp.array([[0.]]), z_max]
    def slice_wb(axis):
        velocity_key = ['u', 'v', 'w']
        key = velocity_key[axis]
        wb_xmin = []
        wb_xmax = []
        wb_ymin = []
        wb_ymax = []
        wb_zmin = []
        wb_zmax = []
        ub_xmin = []
        ub_xmax = []
        ub_ymin = []
        ub_ymax = []
        ub_zmin = []
        ub_zmax = []
        for i, t in enumerate(tc):
            mesh_file = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000 + 100*step_idx + 100 * (i+1)}.vtu"
            input_dict, mesh = load_vtu_from_mesh(mesh_file)
            wt = input_dict["vorticity_vector"][:,axis].reshape(32,32,32)[jnp.newaxis, :]
            ut = input_dict[key].reshape(32,32,32)[jnp.newaxis, :]
            wb_xmin.append(wt[:, :1,  :,  :]) # slice box
            wb_xmax.append(wt[:,-1:,  :,  :]) # slice box
            wb_ymin.append(wt[:,  :, :1,  :]) # slice box
            wb_ymax.append(wt[:,  :,-1:,  :]) # slice box
            wb_zmin.append(wt[:,  :,  :, :1]) # slice box
            wb_zmax.append(wt[:,  :,  :,-1:]) # slice box
            ub_xmin.append(ut[:, :1,  :,  :]) # slice box
            ub_xmax.append(ut[:,-1:,  :,  :]) # slice box
            ub_ymin.append(ut[:,  :, :1,  :]) # slice box
            ub_ymax.append(ut[:,  :,-1:,  :]) # slice box
            ub_zmin.append(ut[:,  :,  :, :1]) # slice box
            ub_zmax.append(ut[:,  :,  :,-1:]) # slice box
        wb_xmin = jnp.concatenate(wb_xmin, axis=0)
        wb_xmax = jnp.concatenate(wb_xmax, axis=0)
        wb_ymin = jnp.concatenate(wb_ymin, axis=0)
        wb_ymax = jnp.concatenate(wb_ymax, axis=0)
        wb_zmin = jnp.concatenate(wb_zmin, axis=0)
        wb_zmax = jnp.concatenate(wb_zmax, axis=0)
        ub_xmin = jnp.concatenate(ub_xmin, axis=0)
        ub_xmax = jnp.concatenate(ub_xmax, axis=0)
        ub_ymin = jnp.concatenate(ub_ymin, axis=0)
        ub_ymax = jnp.concatenate(ub_ymax, axis=0)
        ub_zmin = jnp.concatenate(ub_zmin, axis=0)
        ub_zmax = jnp.concatenate(ub_zmax, axis=0)
        return [wb_xmin, wb_xmax, wb_ymin, wb_ymax, wb_zmin, wb_zmax], [ub_xmin, ub_xmax, ub_ymin, ub_ymax, ub_zmin, ub_zmax]
    slice_x = slice_wb(0)
    slice_y = slice_wb(1)
    slice_z = slice_wb(2)
    wb = [[slice_x[0][i], slice_y[0][i], slice_z[0][i]] for i in range(6)]
    ub = [[slice_x[1][i], slice_y[1][i], slice_z[1][i]] for i in range(6)]
    return tc, xc, yc, zc, None, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb, ub


def generate_test_data_cylinder(step_idx):
    ic_mesh = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000 + 100*step_idx}.vtu"
    print("loading ic mesh : ", ic_mesh)
    input_dict, mesh = load_vtu_from_mesh(ic_mesh)
    # box_size = 20 * 20 * 20 
    nc = 20 
    time_step = 5
    # [test] points
    # 时间步长是0.01s
    t_vec = jnp.linspace(0.01,0.33,time_step).reshape(-1, 1)
    x_vec = np.unique(input_dict['x'])[10:10+nc].reshape(-1, 1) - input_dict['x'].min()
    y_vec = np.unique(input_dict['y'])[10:10+nc].reshape(-1, 1) - input_dict['y'].min()
    z_vec = np.unique(input_dict['z'])[10:10+nc].reshape(-1, 1) - input_dict['z'].min()
    wi_x = []
    wi_y = []
    wi_z = []

    for i, t in enumerate(t_vec):
        mesh_file = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000 + 100*step_idx + 100 * (i+1)}.vtu"
        input_dict, mesh = load_vtu_from_mesh(mesh_file)
        wi_x.append((input_dict["vorticity_vector"][:,0].reshape(32,32,32)[jnp.newaxis, :])[:,10:10+nc,10:10+nc,10:10+nc])
        wi_y.append((input_dict["vorticity_vector"][:,1].reshape(32,32,32)[jnp.newaxis, :])[:,10:10+nc,10:10+nc,10:10+nc])
        wi_z.append((input_dict["vorticity_vector"][:,2].reshape(32,32,32)[jnp.newaxis, :])[:,10:10+nc,10:10+nc,10:10+nc])

    wi_x = jnp.concatenate(wi_x, axis=0)
    wi_y = jnp.concatenate(wi_y, axis=0)
    wi_z = jnp.concatenate(wi_z, axis=0)
    w_gt = [wi_x, wi_y, wi_z]
    return t_vec, x_vec, y_vec, z_vec, w_gt


def visu(apply_fn, params, e):
    ic_mesh = "./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_400000.vtu"
    input_dict, mesh = load_vtu_from_mesh(ic_mesh)
    t = jnp.linspace(0.01,0.33,32).reshape(-1, 1)
    x = np.unique(input_dict['x']).reshape(-1, 1) - input_dict['x'].min()
    y = np.unique(input_dict['y']).reshape(-1, 1) - input_dict['y'].min()
    z = np.unique(input_dict['z']).reshape(-1, 1) - input_dict['z'].min()
    # calculate u
    ux, uy, uz = apply_fn(params, t, x, y, z)
    # calculate w (3D vorticity vector)
    wx = vorx(apply_fn, params, t, x, y, z)
    wy = vory(apply_fn, params, t, x, y, z)
    wz = vorz(apply_fn, params, t, x, y, z)
    jnp.save(f"ux_{e}", ux)
    jnp.save(f"uy_{e}", uy)
    jnp.save(f"uz_{e}", uz)
    jnp.save(f"wx_{e}", wx)
    jnp.save(f"wy_{e}", wy)
    jnp.save(f"wz_{e}", wz)
