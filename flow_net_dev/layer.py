import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Graph, DirectedEdge


class GraphLayer(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.g = Graph.input_output_graph(num_inputs, num_outputs)
        for i, o in enumerate(self.g.outputs):
            for inp in self.g.inputs:
                self.g.add_edge(inp, o)

            # Constrain some arbitrary pairs of input edges to be equal.
            in_idx = round(i * num_inputs / num_outputs)
            self.g.equal_edges.add(
                (
                    DirectedEdge(self.g.inputs[in_idx % len(self.g.inputs)], o),
                    DirectedEdge(self.g.inputs[(in_idx + 1) % len(self.g.inputs)], o),
                )
            )
        self.param = nn.Parameter(torch.randn(self.g.num_edges))
        self.register_buffer("lhs", self.g.create_constraint_lhs())

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = [
            self.g.read_outputs_for_inputs(
                lhs=self.lhs,
                inputs=x,
                capacities=F.softplus(self.param),
            )
            for x in xs
        ]
        return torch.stack(outs)
