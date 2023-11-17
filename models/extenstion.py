import torch
import os

from ExplanationEvaluation.models.GNN_paper import NodeGCN as GNN_NodeGCN
from ExplanationEvaluation.models.GNN_paper import GraphGCN as GNN_GraphGCN
from ExplanationEvaluation.models.PG_paper import NodeGCN as PG_NodeGCN
from ExplanationEvaluation.models.PG_paper import GraphGCN as PG_GraphGCN

def batch_forward(model, x,edge_index, edge_weights=None,sizes_lists=None):
    """
    Given a GNN model for graph classification, output the batch predictions.
    """
    input_lin = model.embedding(x, edge_index, edge_weights) # batch*node, embedding

    if sizes_lists is None:
        final = model.lin(input_lin)
        return final
    else:
        # Split the tensor according to sizes
        split_tensors = torch.split(input_lin, sizes_lists)







# Max pooling for each matrix in the split tensors
pooled_outputs = [torch.max(tensor, dim=0)[0] for tensor in split_tensors]

# Stack the results
final_tensor = torch.stack(pooled_outputs)

print(final_tensor.size())  # This should be [N, F] where N is the size of the sizes_vector.

