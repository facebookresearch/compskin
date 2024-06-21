# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy as sp
import torch
import time
import igl
import rig.riglogic as rl


def buildTR(device):
    # fmt: off
    ebase = torch.tensor([[[0, 0,  0, 0],
                           [0, 0, -1, 0],
                           [0, 1,  0, 0]],

                          [[ 0, 0, 1, 0],
                           [ 0, 0, 0, 0],
                           [-1, 0, 0, 0]],

                          [[0, -1, 0, 0],
                           [1,  0, 0, 0],
                           [0,  0, 0, 0]],

                          [[0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],

                          [[0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]],

                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]], dtype=torch.float32).to(device)
    return ebase.reshape(6, 1, 1, 3, 4)
    # fmt: on


def add_homog_coordinate(M, dim):
    x = list(M.shape)
    x[dim] = 1
    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


def compBX(Wn, Brt, TR, n_bs, P):
    # calculates Linear Blend Skinning
    # Wn ∈ PxN   (numBones x numVertices)
    # Brt - 6 degree of freedom  per blendshape per bone  (6, n_bs, numBones, 1, 1)
    # TR  - 6 base matrices (n_bx, numBones, 3, 4)[6]  one per degree of freedom these are used to convert Brt to B
    # rest_pose  ∈ nx4
    # X :  rest_pose.p * weight
    #         vertex...vertex
    #            0      N
    #         ┌           ┐
    # bone0  x│┌───┐      │
    #        y││  →│...   │
    #        z││w*p│      │
    #        w│└───┘      │
    # bone1  x│           │
    #        y│           │
    #        z│           │
    #        w│           │
    #         ┆           ┆
    # boneP  w│           │
    #         └           ┘
    X = (Wn.unsqueeze(2) * rest_pose).permute(0, 2, 1).reshape(4 * P, -1)
    B = Brt[0, ...] * TR[0]
    for i in range(1, 6):
        B += Brt[i, ...] * TR[i]
    B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)
    # B current bone transforms
    #               bone 0... bone N
    #                0123     0123
    #              ┌               ┐
    # blendshape0 0│┌────┐   ┌────┐│
    #             1││TM  │...│TM  ││
    #             2│└────┘   └────┘│
    # blendshape1 0│┌────┐   ┌────┐│
    #             1││TM  │...│TM  ││
    #             2│└────┘   └────┘│
    #              │               │
    #              ┆               ┆
    #              └               ┘
    return B @ X, B, X


def train(num_iter, power, alpha, beta=None, normalizeW=False):
    global Brt, W

    st = time.time()
    for i in range(num_iter):
        if normalizeW:
            Wn = W / W.sum(axis=0)
        else:
            Wn = W

        BX, _, _ = compBX(Wn, Brt, TR, n_bs, P)
        weighed_error = BX - A

        if beta is not None:
            weighed_error[:, salient_verts] *= beta

        loss = weighed_error.pow(power).mean().pow(2 / power)
        if alpha is not None:
            # add Laplacian regularization term
            loss += alpha * (L @ (BX).transpose(0, 1)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Wcutoff = torch.topk(W, max_influences + 1, dim=0).values[-1, :]
            Wmask = W > Wcutoff
            Wpruned = Wmask * W
            W.copy_(Wpruned)
            W.clamp_(min=0)

            # Bdecider = Brt.abs()
            # Bcutoff = torch.topk(Bdecider.flatten(), total_nnz_Brt).values[-1]
            # Bmask = Bdecider >= Bcutoff
            # Bpruned = Bmask * Brt
            # Brt.copy_(Bpruned)

        if i % 200 == 0:
            BX, _, _ = compBX(Wn, Brt, TR, n_bs, P)
            trunc_err = (BX - A).abs().max().item()

            if device == "cuda":
                torch.cuda.synchronize()

            print(
                f"{i:05d}({time.time() - st:.3f}) {loss.item():.5e} {trunc_err:.5e} {(Brt.abs() > 1e-4).count_nonzero().item()} {(W.abs() > 1e-4).count_nonzero().item()}"
            )
            loss_list.append(loss.item())
            abserr_list.append(trunc_err)

            st = time.time()


def npf(T):
    return T.detach().cpu().numpy()


def generateXforms(weights, shapeXforms):
    # weights ... (num_shapes, 1), output of riglogic
    # shapeXforms ... (3*num_shapes, 4*num_proxy_bones) matrix
    # returns: (num_proxy_bones, 3, 4) skinning transforms, input to skinCluster

    nShapes = weights.shape[0]
    nBones = shapeXforms.shape[1] // 4
    Z = weights.reshape(1, 1, nShapes) * np.dstack([np.eye(3)] * nShapes)
    # Z:
    # ┌      ┐┌      ┐┌      ┐
    # │w₁   0││w₂   0││w₃   0│
    # │  w₁  ││  w₂  ││  w₃  │  ───▶ axis 2
    # │0   w₁││0   w₂││0   w₃│
    # └      ┘└      ┘└      ┘
    #
    # Z.transpose(0, 2, 1).reshape(3, -1)
    # ┌                  ┐
    # │w₁0 0 w₂0 0 w₃0 0 │
    # │0 w₁0 0 w₂0 0 w₃0 │
    # │0 0 w₁0 0 w₂0 0 w₃│
    # └                  ┘
    # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms
    #   weighted sum of blendshape transfomrs (3, 4 * num_bones)
    #
    # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array([np.eye(3, 4)] * nBones).transpose(1, 0, 2).reshape(3, -1)
    #   add 1 to diagonals for every transform (befor was 0)
    res = Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array(
        [np.eye(3, 4)] * nBones
    ).transpose(1, 0, 2).reshape(3, -1)
    return res


seed = 12345
P = 200  # number of bones
max_influences = 32  # number of weights per vertex
# total_nnz_Brt = 6000  # number of non-zero values into Brt matrix
init_weight = 1e-3
power = 12  # metric power
alpha = 5
beta = None

torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
npb = np.load(f"in/proteus.npz", allow_pickle=True)
n_bs = npb["deltas"].shape[0]
deltas = npb["deltas"].transpose(1, 0, 2).reshape(-1, n_bs * 3).transpose()

# A is blendshapes matrix. (goal of optimization)
# shape: (num_blendShapes * 3 (x, y, z), num vertices)
A = torch.from_numpy(deltas).float().to(device)
N = A.shape[1]  # number of vertices

quads = npb["rest_faces"]
rest = npb["rest_verts"]

adj = igl.adjacency_matrix(quads)
adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
# Rigidness Laplacian regularization.
# ⎧-1        if k = i
# ⎨1/|N(i)|, if k ∈ N(i)
# ⎩0         otherwise.
# Where N(i) denotes all the 1-ring neighbours of i
Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
L = torch.from_numpy((Lg).todense()).float().to(device).to_sparse()

loss_list, abserr_list = [], []

TR = buildTR(device)
# Brt - 6 degree of freedom  per blendshape per bone  (6, numBlendshapes, numBones, 1, 1)
Brt = (init_weight * torch.randn((6, n_bs, P, 1, 1))).clone().float().to(device).requires_grad_()

rest_centered = rest - rest.mean(axis=0)
# rest_pose  nx4 aray of vertices (with forth column 1)
rest_pose = torch.from_numpy(add_homog_coordinate(rest_centered, 1)).float().to(device)
# W PxN (numBonex x numVertices) weights one per vertex per bone
W = (1e-8 * torch.randn(P, N)).clone().float().to(device).requires_grad_()

param_list = [Brt, W]

optimizer = torch.optim.Adam(param_list, lr=1e-3, betas=(0.9, 0.9))
train(10000, power=power, alpha=alpha, beta=beta, normalizeW=False)

optimizer = torch.optim.Adam(param_list, lr=1e-3, betas=(0.9, 0.9))
train(10000, power=power, alpha=alpha, beta=beta, normalizeW=True)

optimizer.param_groups[0]["lr"] = 1e-4
train(480000, power=power, alpha=alpha, beta=beta, normalizeW=True)

Wn = W / W.sum(axis=0)
print(Wn.min().item(), Wn.max().item())
BX, _, _ = compBX(Wn, Brt, TR, n_bs, P)
orig_deltas = npf(A.transpose(1, 0).reshape(-1, n_bs, 3))
our_deltas = npf(BX.transpose(1, 0).reshape(-1, n_bs, 3))

maxDelta = np.abs(orig_deltas - our_deltas).max()
meanDelta = np.abs(orig_deltas - our_deltas).mean()
print(f"maxDelta {maxDelta}")
print(f"meanDelta {meanDelta}")

inbetween_dict = npb["inbetween_info"].item()
corrective_dict = npb["combination_info"].item()

test_anim = np.load("in/test_anim.npz")
# anim_weights num_frames x num_blendshapes
# one weight per blendshape per frame
anim_weights = rl.compute_rig_logic(
    torch.from_numpy(test_anim["weights"][:, :72]).float(), inbetween_dict, corrective_dict
).numpy()

num_frames = anim_weights.shape[0]
print(num_frames, anim_weights.shape[1])

_, B, _ = compBX(Wn, Brt, TR, n_bs, P)

shapeXforms = B.detach().cpu().numpy()

np.savez(
    "out/result.npz",
    rest=npf(rest_pose[:, :3]),
    quads=quads,
    weights=npf(Wn).transpose(),
    restXform=np.array([np.eye(3, 4)] * P),
    shapeXform=shapeXforms,
)

for i in range(num_frames):
    T = generateXforms(anim_weights[i, :], shapeXforms)
    X = npf((Wn.unsqueeze(2) * rest_pose).permute(0, 2, 1).reshape(4 * P, -1))
    anim_verts = T @ X
    igl.write_obj(f"out/anim_frame{i:05d}.obj", anim_verts.transpose(), quads)
