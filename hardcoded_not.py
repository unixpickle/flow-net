import torch
import torch.nn as nn

from flow_net_dev.network import Graph, Vertex, DirectedEdge

g = Graph.input_output_graph(num_inputs=1, num_outputs=1)

# first vertex that either feeds into output or into rest of NOT gate
v1 = Vertex()
g.add_vertex(v1)
g.add_edge(g.source, v1)
g.add_edge(v1, g.outputs[0])

# gate with constraint
v2 = Vertex()
g.add_vertex(v2)
g.add_edge(v1, v2)
g.add_edge(g.inputs[0], v2)
g.equal_edges.add((DirectedEdge(v1, v2), DirectedEdge(g.inputs[0], v2)))

# # output replicator to flush extra output to sink
v3 = Vertex()
g.add_vertex(v2)
g.add_edge(v2, v3)
g.add_edge(g.source, v3)
g.add_edge(v3, g.sink)
g.equal_edges.add((DirectedEdge(v2, v3), DirectedEdge(g.source, v3)))

param = nn.Parameter(torch.zeros(g.num_edges))
with torch.no_grad():
    param[g.edge_ids[DirectedEdge(g.source, v1)]] = 1.0
    param[g.edge_ids[DirectedEdge(v1, g.outputs[0])]] = 1.0
    param[g.edge_ids[DirectedEdge(v1, v2)]] = 1.0
    param[g.edge_ids[DirectedEdge(g.inputs[0], v2)]] = 1.0
    param[g.edge_ids[DirectedEdge(v2, v3)]] = 1.0
    param[g.edge_ids[DirectedEdge(g.source, v3)]] = 1.0
    param[g.edge_ids[DirectedEdge(v3, g.sink)]] = 2.0
    param[g.edge_ids[DirectedEdge(g.outputs[0], g.sink)]] = 1.0

lhs = g.create_constraint_lhs()


def compute_loss():
    loss = 0
    for input in [0, 1]:
        out = g.read_outputs_for_inputs(
            lhs=lhs,
            inputs=torch.tensor([input], dtype=torch.float),
            capacities=param.abs(),
        )
        loss = loss + (out - (1 - input)).pow(2)
        break
    return loss


with torch.no_grad():
    print("initial loss", compute_loss())

    param.copy_(torch.rand_like(param))

opt = torch.optim.Adam([param], lr=0.01)
step = 0
while True:
    loss = compute_loss()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step}: loss={loss}")
    step += 1
