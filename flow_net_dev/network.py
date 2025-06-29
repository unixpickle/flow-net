import random
from dataclasses import dataclass

import numpy as np
import torch

from .lp import make_lp_layer


class Vertex:
    pass


@dataclass
class DirectedEdge:
    v1: Vertex
    v2: Vertex

    def __hash__(self) -> int:
        return hash((self.v1, self.v2))


@dataclass
class Graph:
    source: Vertex
    sink: Vertex

    vs: set[Vertex]
    neighbors: dict[Vertex, set[Vertex]]

    edge_ids: dict[DirectedEdge, int]

    equal_edges: set[tuple[DirectedEdge, DirectedEdge]]

    @property
    def num_edges(self) -> int:
        return len(self.edge_ids)

    @classmethod
    def input_output_graph(cls, num_inputs: int, num_outputs: int) -> "Graph":
        """
        The first num_inputs edges are edges from source to the
        input vertices.

        The next num_outputs edges are edges from outputs to the sink.
        """
        source = Vertex()
        sink = Vertex()
        inputs = [Vertex() for _ in range(num_inputs)]
        outputs = [Vertex() for _ in range(num_outputs)]
        neighbors = {source: set(inputs), sink: set(outputs)}
        edge_ids = {}
        for inp in inputs:
            neighbors[inp] = {source}
            edge_ids[DirectedEdge(source, inp)] = len(edge_ids)
        for out in outputs:
            neighbors[out] = {sink}
            edge_ids[DirectedEdge(out, sink)] = len(edge_ids)

        # Create the opposite edges, though they aren't particularly useful.
        for inp in inputs:
            edge_ids[DirectedEdge(inp, source)] = len(edge_ids)
        for out in outputs:
            edge_ids[DirectedEdge(sink, out)] = len(edge_ids)

        return cls(
            source=source,
            sink=sink,
            vs=set(inputs) | set(outputs) | {source, sink},
            neighbors=neighbors,
            edge_ids=edge_ids,
            equal_edges=set(),
        )

    def add_vertex(self, v: Vertex):
        self.vs.add(v)

    def add_edge(self, v1: Vertex, v2: Vertex):
        self.neighbors[v1] = self.neighbors.get(v1, set()) | {v2}
        self.neighbors[v2] = self.neighbors.get(v2, set()) | {v1}
        self.edge_ids[DirectedEdge(v1, v2)] = len(self.edge_ids)
        self.edge_ids[DirectedEdge(v2, v1)] = len(self.edge_ids)

    def add_random_vertex(self, force_equal_prob: float = 0.5):
        v = Vertex()
        neighbors = random.sample(list(self.vs), 3)
        self.add_vertex(v)
        for neighbor in neighbors:
            self.add_edge(v, neighbor)
        if random.random() < force_equal_prob:
            self.equal_edges.add(
                (DirectedEdge(neighbors[0], v), DirectedEdge(neighbors[1], v))
            )

    def create_constraint_lhs(self) -> torch.Tensor:
        num_vars = (
            len(self.edge_ids) * 2 + 2
        )  # source and sink get an extra var, edges get slack vars
        source_var = num_vars - 2
        sink_var = num_vars - 1

        num_constraints = len(self.vs) + len(self.edge_ids) + len(self.equal_edges)
        result = torch.zeros(num_constraints, num_vars)

        # All vertex flows (plus source/sink slack) equal zero.
        for i, v in enumerate(self.vs):
            for neighbor in self.neighbors.get(v, []):
                outgoing = DirectedEdge(v, neighbor)
                incoming = DirectedEdge(neighbor, v)
                result[i, self.edge_ids[outgoing]] = 1
                result[i, self.edge_ids[incoming]] = -1
            if v == self.source:
                result[i, source_var] = -1
            elif v == self.sink:
                result[i, sink_var] = 1

        # Each edge + its slack variable equals capacity
        for i in range(len(self.edge_ids)):
            result[i + len(self.vs), i] = 1
            result[i + len(self.vs), i + len(self.edge_ids)] = 1

        # edge1 - edge2 = 0 for force-equal edges
        for i, (edge1, edge2) in enumerate(self.equal_edges):
            row = len(self.vs) + len(self.edge_ids) + i
            result[row, self.edge_ids[edge1]] = 1
            result[row, self.edge_ids[edge2]] = -1

        return result

    def create_constraint_rhs(self, capacities: torch.Tensor) -> torch.Tensor:
        assert capacities.shape == (len(self.edge_ids),)
        return torch.cat(
            [
                torch.zeros(
                    len(self.vs), dtype=capacities.dtype, device=capacities.device
                ),
                capacities,
                torch.zeros(
                    len(self.equal_edges),
                    dtype=capacities.dtype,
                    device=capacities.device,
                ),
            ]
        )

    def read_outputs_for_inputs(
        self,
        lhs: torch.Tensor,
        inputs: torch.Tensor,
        capacities: torch.Tensor,
        num_outputs: int,
    ) -> torch.Tensor:
        c = torch.zeros(lhs.shape[1], device=lhs.device, dtype=lhs.dtype)
        c[-1] = 1  # minimize the sink flow => maximize the source flow

        use_capacities = capacities.clone()
        use_capacities[: len(inputs)] = inputs
        rhs = self.create_constraint_rhs(use_capacities)

        lp_layer = make_lp_layer(n=lhs.shape[1], m=lhs.shape[0])

        c = torch.zeros(lhs.shape[1], device=lhs.device, dtype=lhs.dtype)
        c[-1] = 1.0

        (x_star,) = lp_layer(lhs, rhs, c)
        return x_star[inputs.shape[0] : inputs.shape[0] + num_outputs]
