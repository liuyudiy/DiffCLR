import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

from utils.graph import BatchMatGraph
from utils.basic import timestep_embedding


class TokenEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_value):
        super(TokenEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed_token = []
        for i, item in enumerate(embed_value):
            if isinstance(item, int):
                # TODO: should we use sparse gradient?
                # Keep in mind that only a limited number of optimizers support sparse gradients:
                # currently it’s optim.SGD(CUDA and CPU), optim.SparseAdam(CUDA and CPU) and optim.Adagrad(CPU)
                # See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                item = torch.nn.Embedding(item, embed_dim)
            elif isinstance(item, torch.nn.Embedding):
                assert item.embedding_dim == embed_dim
            self.add_module(f'embed_token_{i}', item)
            self.embed_token.append(item)

    def forward(self, node_type, node_id) -> torch.FloatTensor:
        # Node type embedding as a base
        embedding = torch.zeros((len(node_id), self.embed_dim), device=node_id.device) # (BN,D)
        for i, embed in enumerate(self.embed_token):
            mask = node_type == i
            # Add token embedding
            # TODO: check whether the in-place operation works with gradients
            embedding[mask] += embed(node_id[mask])
        return embedding # (BN, D)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size 
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        # Scaled Dot-Product Attention.
        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias
        
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class KGReasoning(nn.Module):
    def __init__(self, num_nodes: int, relation_cnt: int, args):
        super(KGReasoning, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.hidden_t_dim = args.hidden_t_dim
        self.num_nodes = num_nodes
        self.num_heads = args.num_heads
        self.relation_cnt = relation_cnt
        self.loss_type = args.loss
        self.smoothing = args.smoothing
        self.alpha = args.alpha

        # Check all appearances of token_embed before changing the scheme!
        self.embed_value = [self.num_nodes, 1, relation_cnt * 2]
        self.embed_type = torch.nn.Embedding(len(self.embed_value), self.hidden_dim)
        self.embed_degree = torch.nn.Embedding(args.hop, self.hidden_dim)

        self.token_embed = TokenEmbedding(self.hidden_dim, embed_value=self.embed_value)
        time_embed_dim = args.hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(args.hidden_t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.hidden_dim)
        )
        self.register_buffer("position_ids", torch.arange(args.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, self.hidden_dim)
        
        self.attn_bias_embed = nn.Embedding(300, self.num_heads, padding_idx=1) 
        with torch.no_grad():
            self.attn_bias_embed.weight[1] = torch.full((self.num_heads,), float('-inf'))
        self.encode_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=args.hidden_dim,
                ffn_size=args.dim_feedforward,
                dropout_rate=args.dropout,
                attention_dropout_rate=args.dropout,
                head_size=args.num_heads
            )
            for _ in range(args.num_layers)
        ])
        self.Dropout = nn.Dropout(args.dropout)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.pred_ent_proj = torch.nn.Linear(self.hidden_dim, self.num_nodes)
        # self.pred_ent_proj = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 2*self.hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(2*self.hidden_dim, self.num_nodes)
        # )
        # self.token_embed.embed_token

        self.lm_head = [0] * len(self.embed_value)
        for i, item in enumerate(self.embed_value):
            if isinstance(item, int):
                self.lm_head[i] = nn.Linear(self.hidden_dim, item)
        with torch.no_grad():
            for i in range(len(self.embed_value)):
                self.lm_head[i].weight = self.token_embed.embed_token[i].weight

    def get_logits(self, node_type, hidden_repr):
        logits = [0] * len(self.embed_value)
        for i, item in enumerate(self.embed_value):
            if isinstance(item, int):
                logits[i] = torch.zeros((hidden_repr.shape[0], item), device=hidden_repr.device)
        for i in range(len(self.embed_value)):
            mask = node_type == i
            lm_head = self.lm_head[i].to(hidden_repr.device)
            logits[i][mask] += lm_head(hidden_repr[mask])

        return logits


    def query_embedding(self, input_x, attn_bias_type):
        rel_pos_bias = self.attn_bias_embed(attn_bias_type) 
        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)
        attn_bias = rel_pos_bias
        for layer in self.encode_layers:
            input_x = layer(input_x, attn_bias)
        feat = self.LayerNorm(input_x) 
        feat = feat.view(-1, self.hidden_dim) 
        return feat 

    def forward(self, input_x, timesteps, attn_bias_type=None, pred_type=None, node_type=None, node_degree=None, isdiffu=False):
        if isdiffu:
            emb_x = input_x  # (B,N,D)
            emb_p = self.embed_type(node_type).view(emb_x.shape[0], emb_x.shape[1], -1)
            emb_d = self.embed_degree(node_degree).view(emb_x.shape[0], emb_x.shape[1], -1)
            emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim)) 
            seq_length = input_x.size(1)
            # position_ids = self.position_ids[:, : seq_length ] 
            # emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_p + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
            # emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
            # emb_inputs = emb_x + emb_p + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
            emb_inputs = emb_x + emb_p + emb_d + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
            
            emb_inputs = self.Dropout(self.LayerNorm(emb_inputs))

            rel_pos_bias = self.attn_bias_embed(attn_bias_type) 
            rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)
            attn_bias = rel_pos_bias
            for layer in self.encode_layers:
                hidden_states = layer(emb_inputs, attn_bias)
                emb_inputs = hidden_states
            
            output = self.LayerNorm(hidden_states)
        else:
            output = input_x

        mask = pred_type == 0
        feat = input_x.view(-1, self.hidden_dim)
        feat_entity = feat[mask]
        score = self.pred_ent_proj(feat_entity) 
        return output, score
    

    def answer_queries(self, feat, data: BatchMatGraph): 
        device = data.x.device
        relabel_arr = torch.empty(data.x.shape, dtype=torch.long, device=device)
        # Currently supports query type 0 (entities) only
        mask = data.pred_type == 0 
        mask_cnt = torch.count_nonzero(mask).item()

        # relabel all the nodes
        relabel_arr[mask] = torch.arange(mask_cnt, device=device)

        if min(data.joint_nodes.shape) != 0:
            sfm = torch.nn.Softmax(dim=1)
            q_mask = mask[data.x_query]
            jq_mask = mask[data.joint_nodes]
            uq_mask = mask[data.union_query]
            p_mask = mask[data.pos_x]

            x_pred = self.pred_ent_proj(feat[mask])
            x_pred = x_pred.double()
            jq = data.joint_nodes[jq_mask]
            uq = data.union_query[uq_mask]
            assert sum(jq) == sum(data.joint_nodes)
            assert sum(uq) == sum(data.union_query)

            relabeled_jq_even = relabel_arr[jq[::2]]
            relabeled_jq_odd = relabel_arr[jq[1::2]]
            relabeled_uq = relabel_arr[uq]

            x_pred[relabeled_jq_even] = sfm(x_pred[relabeled_jq_even])
            x_pred[relabeled_jq_odd] = sfm(x_pred[relabeled_jq_odd])
            q_score = None
            if data.x_ans is not None:
                # q_score = torch.max(x_pred[relabeled_jq_even, data.x_ans], x_pred[relabeled_jq_odd, data.x_ans])
                e_score = x_pred[relabeled_jq_even, data.x_ans]
                o_score = x_pred[relabeled_jq_odd, data.x_ans]

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            if data.x_ans is not None:
                x_pred[relabeled_jq_even, data.x_ans] = e_score
                x_pred[relabeled_jq_odd, data.x_ans] = o_score

            # Using rank as score
            eind = torch.argsort(x_pred[relabeled_jq_even], dim=1)
            fi = torch.arange(x_pred[relabeled_jq_even].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_even].shape[0], 1)
            x_pred[relabeled_jq_even] = torch.scatter(x_pred[relabeled_jq_even], 1, eind, fi)

            oind = torch.argsort(x_pred[relabeled_jq_odd], dim=1)
            fi2 = torch.arange(x_pred[relabeled_jq_odd].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_odd].shape[0], 1)
            x_pred[relabeled_jq_odd] = torch.scatter(x_pred[relabeled_jq_odd], 1, oind, fi2)

            q_pred = torch.max(x_pred[relabeled_jq_even], x_pred[relabeled_jq_odd])
            e_pred = x_pred[relabeled_jq_even]
            o_pred = x_pred[relabeled_jq_odd]
        else:
            # q_mask and p_mask: queries on entities (should all be True)
            q_mask = mask[data.x_query]
            p_mask = mask[data.pos_x] 

            # predict for all the nodes
            x_pred = self.pred_ent_proj(feat[mask]) 
            # relabel the query
            relabeled_query = relabel_arr[data.x_query[q_mask]] 

            # If we are training, we have to make sure that answers are not masked
            q_score = None
            if data.x_ans is not None:
                q_score = x_pred[relabeled_query, data.x_ans[q_mask]] 

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            q_pred = x_pred[relabeled_query]

            # Add back those to be predicted so that we know the scores of the x_ans
            if q_score is not None:
                q_pred[torch.arange(q_mask.shape[0], device=device), data.x_ans[q_mask]] = q_score
        return q_pred, None, None

