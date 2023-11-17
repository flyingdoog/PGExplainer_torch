import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge
# from ExplanationEvaluation.models.extenstion import batch_forward
class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, graphs, features, task, device='cpu',epochs=30, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task,device)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = self.model_to_explain.embedding_size * 2
   
        self.device = device

    def _create_explainer_input(self, pair, embeds):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(),device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)

    def train(self, indices = None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))


        
        batch_edgelists = []
        edgelists = []
        adjs = []
        offset = 0
        full_rows = []
        full_cols = []
        batch_features = []
        offsets = []
        node_sizes = []
        node_indicator = []
#         batch_size = 1000
        
        self.reg_coefs[0]/=len(self.graphs)

        for i in range(len(self.graphs)):
            feature = self.features[i]
            edge_list = self.graphs[i]
            nmb_node = feature.shape[0]
            node_sizes.append(nmb_node)
            
            node_indicator.append(torch.tensor([i]*nmb_node))
            
            edgelists.append(edge_list+offset)
            offsets.append(offset)
            batch_features.append(feature)

            # get the index for fully connected graphs. 
            _temp_arange_nmb_node = torch.arange(nmb_node)
            _temp_matrix_nmb_node = _temp_arange_nmb_node.repeat(nmb_node,1)
            _full_col = _temp_matrix_nmb_node.view(-1)+offset
            _full_row = _temp_matrix_nmb_node.transpose(1,0).reshape(-1)+offset
            full_rows.append(_full_row)
            full_cols.append(_full_col)

            offset+=int(nmb_node)
    
        batch_features_tensor = torch.concat(batch_features,0)
        batch_edge_list = torch.concat(edgelists,-1)
        batch_full_rows = torch.concat(full_rows)
        batch_full_cols = torch.concat(full_cols)
        batch_graphs = torch.stack([batch_full_rows,batch_full_rows])
        
        all_one_edge_weights = torch.ones(batch_edge_list.size(1)).to(self.device)
        node_indicator_tensor = torch.concat(node_indicator,-1).to(self.device)
        
        with torch.no_grad():
            embeds = self.model_to_explain.embedding(batch_features_tensor,batch_edge_list,all_one_edge_weights)
            original_pred = self.model_to_explain(batch_features_tensor, 
                                              batch_edge_list,
                                              batch=node_indicator_tensor, 
                                              edge_weights=all_one_edge_weights)        
        # Start training loop
        for e in tqdm(range(0, self.epochs)):

            optimizer.zero_grad()
            t = temp_schedule(e)
            input_expl = self._create_explainer_input(batch_edge_list, embeds).unsqueeze(0)
            sampling_weights = self.explainer_model(input_expl)
            mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

            masked_pred = self.model_to_explain(batch_features_tensor,
                                                batch_edge_list, 
                                                batch=node_indicator_tensor,
                                                edge_weights=mask) # [batch,2]
            
            target = torch.argmax(original_pred,-1)
            loss = self._loss(masked_pred,target, mask, self.reg_coefs)

            loss.backward()
            optimizer.step()
        

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index = int(index)
        feats = self.features[index].clone().detach()
        graph = self.graphs[index].clone().detach()
        all_one_edge_weights = torch.ones(graph.size(1)).to(self.device)
        embeds = self.model_to_explain.embedding(feats, graph,all_one_edge_weights).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights
