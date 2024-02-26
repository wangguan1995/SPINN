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
    if plane_normal is 'x':
        ref = var_ref[var + "_ref"][0][x_pos]
        pred = var_pred[var + "_pred"][time][x_pos]
    elif plane_normal is 'y':
        ref = var_ref[var + "_ref"][0][:][x_pos]
        pred = var_pred[var + "_pred"][time][:][x_pos]
    elif plane_normal is 'z':
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

    # [pde] points
    # 时间步长是0.01s
    tc = jnp.linspace(0.01,0.33,32).reshape(-1, 1)
    xc = np.unique(input_dict['x']).reshape(-1, 1) - input_dict['x'].min()
    yc = np.unique(input_dict['y']).reshape(-1, 1) - input_dict['y'].min()
    zc = np.unique(input_dict['z']).reshape(-1, 1) - input_dict['z'].min()

    # [initial] points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    zi = zc
    wi_x = input_dict["vorticity_vector"][:,0].reshape(32,32,32)[jnp.newaxis, :]
    wi_y = input_dict["vorticity_vector"][:,1].reshape(32,32,32)[jnp.newaxis, :]
    wi_z = input_dict["vorticity_vector"][:,2].reshape(32,32,32)[jnp.newaxis, :]
    wi = [wi_x, wi_y, wi_z]
    ui_x = input_dict["u"].reshape(32,32,32)[jnp.newaxis, :]
    ui_y = input_dict["v"].reshape(32,32,32)[jnp.newaxis, :]
    ui_z = input_dict["w"].reshape(32,32,32)[jnp.newaxis, :]
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
  # box_size = 32 * 32 * 32 
  nc = 20 

  # [test] points
  # 时间步长是0.01s
  t_vec = jnp.linspace(0.01,0.33,20).reshape(-1, 1)
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

@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, nu, lbda_c, lbda_ic, *train_data):
    def residual_loss(params, t, x, y, z, f):
        # calculate u
        ux, uy, uz = apply_fn(params, t, x, y, z)
        # calculate w (3D vorticity vector)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)
        # tangent vector dx/dx
        vec_t = jnp.ones(t.shape)
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        vec_z = jnp.ones(z.shape)

        # x-component
        wx_t = jvp(lambda t: vorx(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wx_x, wx_xx = hvp_fwdfwd(lambda x: vorx(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wx_y, wx_yy = hvp_fwdfwd(lambda y: vorx(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wx_z, wx_zz = hvp_fwdfwd(lambda z: vorx(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        ux_x = jvp(lambda x: apply_fn(params, t, x, y, z)[0], (x,), (vec_x,))[1]
        ux_y = jvp(lambda y: apply_fn(params, t, x, y, z)[0], (y,), (vec_y,))[1]
        ux_z = jvp(lambda z: apply_fn(params, t, x, y, z)[0], (z,), (vec_z,))[1]

        # no source term
        loss_x = jnp.mean((wx_t + ux*wx_x + uy*wx_y + uz*wx_z - \
             (wx*ux_x + wy*ux_y + wz*ux_z) - \
                nu*(wx_xx + wx_yy + wx_zz)))

        # y-component
        wy_t = jvp(lambda t: vory(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wy_x, wy_xx = hvp_fwdfwd(lambda x: vory(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wy_y, wy_yy = hvp_fwdfwd(lambda y: vory(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wy_z, wy_zz = hvp_fwdfwd(lambda z: vory(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        uy_x = jvp(lambda x: apply_fn(params, t, x, y, z)[1], (x,), (vec_x,))[1]
        uy_y = jvp(lambda y: apply_fn(params, t, x, y, z)[1], (y,), (vec_y,))[1]
        uy_z = jvp(lambda z: apply_fn(params, t, x, y, z)[1], (z,), (vec_z,))[1]

        loss_y = jnp.mean((wy_t + ux*wy_x + uy*wy_y + uz*wy_z - \
             (wx*uy_x + wy*uy_y + wz*uy_z) - \
                nu*(wy_xx + wy_yy + wy_zz)))

        # z-component
        wz_t = jvp(lambda t: vorz(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wz_x, wz_xx = hvp_fwdfwd(lambda x: vorz(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wz_y, wz_yy = hvp_fwdfwd(lambda y: vorz(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wz_z, wz_zz = hvp_fwdfwd(lambda z: vorz(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        uz_x = jvp(lambda x: apply_fn(params, t, x, y, z)[2], (x,), (vec_x,))[1]
        uz_y = jvp(lambda y: apply_fn(params, t, x, y, z)[2], (y,), (vec_y,))[1]
        uz_z = jvp(lambda z: apply_fn(params, t, x, y, z)[2], (z,), (vec_z,))[1]

        loss_z = jnp.mean((wz_t + ux*wz_x + uy*wz_y + uz*wz_z - \
             (wx*uz_x + wy*uz_y + wz*uz_z) - \
                nu*(wz_xx + wz_yy + wz_zz)))

        loss_c = jnp.mean((ux_x + uy_y + uz_z)**2)

        return loss_x + loss_y + loss_z + lbda_c*loss_c

    def initial_loss(params, t, x, y, z, w, u):
        ux, uy, uz = apply_fn(params, t, x, y, z)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)
        loss = jnp.mean((wx - w[0])**2) + jnp.mean((wy - w[1])**2) + jnp.mean((wz - w[2])**2)
        loss += jnp.mean((ux - u[0])**2) + jnp.mean((uy - u[1])**2) + jnp.mean((uz - u[2])**2)
        return loss

    def boundary_loss(params, t, x, y, z, w):
        loss = 0.
        for i in range(6):
            wx = vorx(apply_fn, params, t[i], x[i], y[i], z[i])
            wy = vory(apply_fn, params, t[i], x[i], y[i], z[i])
            wz = vorz(apply_fn, params, t[i], x[i], y[i], z[i])
            loss += (1/6.) * jnp.mean((wx - w[i][0])**2) + jnp.mean((wy - w[i][1])**2) + jnp.mean((wz - w[i][2])**2)
        return loss

    # unpack data
    tc, xc, yc, zc, fc, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb = train_data

    # isolate loss func from redundant arguments
    loss_fn = lambda params: residual_loss(params, tc, xc, yc, zc, fc) + \
                        lbda_ic*initial_loss(params, ti, xi, yi, zi, wi, ui) + \
                        boundary_loss(params, tb, xb, yb, zb, wb)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--debug', type=str, default='false', help='debugging purpose')
    parser.add_argument('--equation', type=str, default='navier_stokes4d', help='equation to solve')
    
    # pde settings
    parser.add_argument('--nc', type=int, default=32, help='the number of collocation points')
    parser.add_argument('--nc_test', type=int, default=20, help = 'the number of collocation points')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')
    parser.add_argument('--mlp', type=str, default='modified_mlp', help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=5, help='the number of layer')
    parser.add_argument('--features', type=int, default=64, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of a approximated tensor')
    parser.add_argument('--out_dim', type=int, default=3, help='size of model output')
    parser.add_argument('--nu', type=float, default=(1/3900), help='viscosity')
    parser.add_argument('--lbda_c', type=int, default=100, help='None')
    parser.add_argument('--lbda_ic', type=int, default=10, help='None')

    # log settings
    parser.add_argument('--log_iter', type=int, default=1000, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=10000, help='plot result every...')

    # time marching
    parser.add_argument('--step_idx', type=int, default=0, help='step index for time marching')

    args = parser.parse_args()

    # random key
    key = jax.random.PRNGKey(args.seed)

    # make & init model forward function
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_networks(args, subkey)

    # count total params
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # name model
    name = name_model(args)

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)

    # make dir
    os.makedirs(result_dir, exist_ok=True)

    # optimizer
    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)

    # dataset
    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data_cylinder(args.step_idx)
    tc, xc, yc, zc, _, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb = train_data

    test_data = generate_test_data_cylinder(args.step_idx)
    t, x, y, z, w_gt = test_data

    # evaluation function
    eval_fn = setup_eval_function(args.model, args.equation)

    # save training configuration
    save_config(args, result_dir)

    # log
    logs = []
    if os.path.exists(os.path.join(result_dir, 'log (loss, error).csv')):
        os.remove(os.path.join(result_dir, 'log (loss, error).csv'))
    if os.path.exists(os.path.join(result_dir, 'best_error.csv')):
        os.remove(os.path.join(result_dir, 'best_error.csv'))
    best = 100000.

    print("compiling...")

    # start training
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # exclude compiling time
            start = time.time()
        if e % 100 == 0:
            # sample new input data
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        loss, gradient = apply_model_spinn(apply_fn, params, args.nu, args.lbda_c, args.lbda_ic, *train_data)
        params, state = update_model(optim, gradient, params, state)

        if e % 10 == 0:
            if loss < best:
                best = loss
                best_error = eval_fn(apply_fn, params, *test_data)

        # log
        if e % args.log_iter == 0:
            optim = optax.adam(learning_rate=args.lr)
            state = optim.init(params)
            error = eval_fn(apply_fn, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, best error {best_error:.8f}')
            with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                f.write(f'{loss}, {error}, {best_error}\n')
        # lr
        if e % 100 == 0:
          args.lr = args.lr * 0.1

        # visualization
        if e % args.plot_iter == 0:
          ic_mesh = f"./data/Box_X=3D_0.036/DNS_Box_X=3D_0.036_{400000 + 100*step_idx}.vtu"
          print("loading ic mesh : ", ic_mesh)
          input_dict, mesh = load_vtu_from_mesh(ic_mesh)
          tc = jnp.linspace(0.01,0.33,32).reshape(-1, 1)
          xc = np.unique(input_dict['x']).reshape(-1, 1) - input_dict['x'].min()
          yc = np.unique(input_dict['y']).reshape(-1, 1) - input_dict['y'].min()
          zc = np.unique(input_dict['z']).reshape(-1, 1) - input_dict['z'].min()
          ux, uy, uz = apply_fn(params, tc, xc, yc, zc)
          jnp.save("./ux.pt", ux)
          jnp.save("./uy.pt", uy)
          jnp.save("./uz.pt", uz)

    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    jnp.save(os.path.join(result_dir, 'params.npy'), params)
        
    # save runtime
    runtime = np.array([runtime])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')

    # save total error
    with open(os.path.join(result_dir, 'best_error.csv'), 'a') as f:
        f.write(f'best error: {best_error}\n')