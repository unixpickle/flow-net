import torch
import torch.nn as nn

from flow_net_dev.network import Graph

g = Graph.input_output_graph(num_inputs=2, num_outputs=1)
for _ in range(20):
    g.add_random_vertex()
lhs = g.create_constraint_lhs()

param = nn.Parameter(torch.rand(g.num_edges))
opt = torch.optim.Adam([param], lr=0.01)
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
                num_outputs=1,
            )
            loss = loss + (out - target).pow(2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"loss={loss}")
