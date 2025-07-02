# flow-net

This was an experiment in creating a different kind of neural network based on constrained maximum flow. The observation is that you can create logic gates out of vertices and edges in a flow graph, provided the extra ability to constrain two edges to have equal flow.

In practice, these networks seem difficult to train. For example, I can hard-code a logical NOT operation, but if I randomize the edge capacities in this network, gradient descent finds a local optimum instead of recovering the correct edge capacities.
