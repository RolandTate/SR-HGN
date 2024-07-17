import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """

        :param node_hidden_dim: int, node hidden feature size
        :param relation_hidden_dim: int,relation hidden feature size
        :param num_heads: int, number of heads in Multi-Head Attention
        :param dropout: float, dropout rate, defaults: 0.0
        :param negative_slope: float, negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)


    def forward(self, dst_node_features: list, raw_dst_node_features: list, dst_relation_embeddings: list,
                dst_relation_embedding_transformation_weight: list, residual_weight: nn.parameter):
        """
        :param dst_node_features: list, [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        :param dst_relation_embeddings: list, [each shape is (n_heads * relation_hidden_dim)]
        :param dst_node_feature_transformation_weight: list, [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        :param dst_relation_embedding_transformation_weight:  list, [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]
        :return: dst_node_relation_fusion_feature: Tensor of the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # 将 dst_node_features 和 raw_dst_node_features 进行平均值聚合
            raw_dst_node_features_avg = torch.stack(raw_dst_node_features, dim=0).mean(dim=0)

            # # (num_dst_relations, n_heads, relation_hidden_dim)
            # dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
            #                                                                               self.num_heads,
            #                                                                               self.relation_hidden_dim)
            # # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            # dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
            #                                                            dim=0).reshape(len(dst_node_features),
            #                                                                           self.num_heads,
            #                                                                           self.relation_hidden_dim,
            #                                                                           self.node_hidden_dim)
            # # shape (num_dst_relations, n_heads, hidden_dim)
            # dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
            #                                        dst_relation_embedding_transformation_weight)

            # 将 dst_node_features 和对应的 dst_relation_embeddings 相乘
            dst_node_features_transformed = []
            for i in range(len(dst_node_features)):
                transformed_feature = dst_node_features[i].reshape(-1, self.num_heads, self.node_hidden_dim)
                # transformed_feature = transformed_feature * dst_relation_embeddings[i].unsqueeze(0)
                dst_node_features_transformed.append(transformed_feature)

            # 平均值聚合
            dst_node_features_avg = torch.stack(dst_node_features_transformed, dim=0).mean(dim=0)
            dst_node_features_avg = dst_node_features_avg.reshape(-1, self.num_heads * self.node_hidden_dim)

            dst_node_relation_fusion_feature = self.dropout(dst_node_features_avg)

            # alpha = torch.sigmoid(residual_weight)
            # dst_node_relation_fusion_feature = dst_node_relation_fusion_feature * alpha + raw_dst_node_features_avg * (
            #             1 - alpha)

        return dst_node_relation_fusion_feature
