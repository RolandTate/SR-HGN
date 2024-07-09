import torch.nn as nn
import dgl


class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: dgl.DGLHeteroGraph, h: dict, relation_embedding: dict,
                relation_src_node_transformation_layers: nn.ModuleDict,
                relation_transformation_weight: nn.ParameterDict):
        """
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph, The Heterogeneous Graph.
        input_src: dict[tuple, Tensor], Input source node features {relation_type: features, }
        input_dst: dict[tuple, Tensor], Input destination node features {relation_type: features, }
        relation_embedding: dict[etype, Tensor], Input relation features {etype: feature}
        node_transformation_weight: nn.ParameterDict, weights {ntype, (inp_dim, hidden_dim)}
        relation_transformation_weight: nn.ParameterDict, weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs, dict[tuple, Tensor]  Output representations for every relation -> {(stype, etype, dtype): features}.
        """
        # dictionary, {(srctype, etype, dsttype): representations}
        outputs = dict()
        dst_nodes_after_transformation = dict()
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            dst_representation, dst_after_transformation = self.mods[etype](rel_graph,
                                                  (h[stype],
                                                   h[dtype]),
                                                  relation_src_node_transformation_layers[etype],
                                                  relation_embedding[etype],
                                                  relation_transformation_weight[etype])


            # dst_representation (dst_nodes, hid_dim)
            outputs[(stype, etype, dtype)] = dst_representation
            dst_nodes_after_transformation[(stype, etype, dtype)] = dst_after_transformation

        return outputs, dst_nodes_after_transformation
