import os
import torch
import torch.nn as nn

from flow_net_dev.network import Graph
import pickle

SAVE_PATH = "state.pkl"

g = Graph.input_output_graph(num_inputs=2, num_outputs=1)
for _ in range(15):
    g.add_random_vertex()

step = 0

if os.path.exists(SAVE_PATH):
    print(f"loading from {SAVE_PATH}")
    with open(SAVE_PATH, "rb") as f:
        obj = pickle.load(f)
    with torch.no_grad():
        param = nn.Parameter(obj["param"])
    step = obj["step"]
    opt = torch.optim.Adam([param], lr=0.01)
    opt.load_state_dict(obj["opt"])
    g = obj["g"]
else:
    param = nn.Parameter(torch.rand(g.num_edges))
    opt = torch.optim.Adam([param], lr=0.01)

lhs = g.create_constraint_lhs()

while True:
    loss = 0
    for a in [0, 1]:
        for b in [0, 1]:
            inputs = torch.tensor([a, b], dtype=torch.float)
            target = a ^ b
            out = g.read_outputs_for_inputs(
                lhs=lhs,
                inputs=inputs,
                capacities=param.abs(),
            )
            loss = loss + (out - target).pow(2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step}: loss={loss}")
    step += 1
    if step % 10 == 0:
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(
                dict(
                    param=param.detach(),
                    g=g,
                    step=step,
                    opt=opt.state_dict(),
                ),
                f,
            )
