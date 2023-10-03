import copy
import torch
import torch.nn as nn

from trafficgen.utils.model_utils import CG_stacked
from .blocks import MLP, pos2posemb

copy_func = copy.deepcopy

class DETRAgentQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.use_attr_gmm = cfg.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_ENABLE
        self.attr_gmm_k = cfg.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_K
        self.use_background = cfg.LOSS.DETR.PRED_BACKGROUND
        self.model_cfg = cfg.MODEL.SCENE.INIT_CFG

        self.full_cfg = cfg
        self.hidden_dim = self.model_cfg['hidden_dim']
        self.motion_cfg = cfg.MODEL.MOTION

        self._init_encoder()
        self._init_decoder()

    def _init_encoder(self):
        self.CG_line = CG_stacked(5, self.hidden_dim)
        self.line_encode = MLP([4, 256, 512, self.hidden_dim])
        self.type_embedding = nn.Embedding(20, self.hidden_dim)
        self.traf_embedding = nn.Embedding(4, self.hidden_dim)

    def _init_decoder(self):
        mlp_dim = self.full_cfg.MODEL.SCENE.INIT_CFG.DECODER.MLP_DIM

        dcfg = self.model_cfg.DECODER
        self.dtype = dcfg.TYPE
        d_model = self.model_cfg.hidden_dim * 2

        layer_cfg = {'d_model': d_model, 'nhead': dcfg.NHEAD, 'dim_feedforward': dcfg.FF_DIM, 'dropout': dcfg.DROPOUT, 'activation': dcfg.ACTIVATION, 'batch_first': True}
        decoder_layer = nn.TransformerDecoderLayer(**layer_cfg)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dcfg.NLAYER)

        self.actor_query = nn.Parameter(torch.randn(1, dcfg.QUERY_NUM, d_model))

        self.speed_head = MLP([d_model, dcfg.MLP_DIM, 1])
        self.vel_heading_head = MLP([d_model, dcfg.MLP_DIM, 1])

        if self.use_attr_gmm:
            self.pos_head = MLP([d_model, dcfg.MLP_DIM, self.attr_gmm_k*(1+5)])
            self.bbox_head = MLP([d_model, dcfg.MLP_DIM, self.attr_gmm_k*(1+5)])
            self.heading_head = MLP([d_model, dcfg.MLP_DIM, self.attr_gmm_k*(1+2)])
        else:
            self.pos_head = MLP([d_model, dcfg.MLP_DIM, 2])
            self.bbox_head = MLP([d_model, dcfg.MLP_DIM, 2])
            self.heading_head = MLP([d_model, dcfg.MLP_DIM, 1])

        if self.use_background:
            self.background_head = MLP([d_model, dcfg.MLP_DIM, 1])

        if self.motion_cfg.ENABLE:
            self._init_motion_decoder(d_model, dcfg)

        query_dim = self.model_cfg.ATTR_QUERY.POS_ENCODING_DIM
        self.query_embedding_layer = nn.Sequential(
            nn.Linear(query_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.query_mask_head = MLP([d_model, mlp_dim*2, mlp_dim])
        self.memory_mask_head = MLP([d_model, mlp_dim*2, mlp_dim])


    def _init_motion_decoder(self, d_model, dcfg):
        self.m_K = self.motion_cfg.K
        self.m_dim = 2 * self.motion_cfg.STEP

        self.motion_prob_head = MLP([d_model, dcfg.MLP_DIM, dcfg.MLP_DIM // 2, self.m_K])
        self.motion_head = MLP([d_model, dcfg.MLP_DIM, dcfg.MLP_DIM // 2, self.m_K * self.m_dim])

        if self.motion_cfg.PRED_HEADING_VEL:
            self.angle_head = MLP([d_model, dcfg.MLP_DIM, dcfg.MLP_DIM // 2, self.m_K * self.motion_cfg.STEP])
            self.vel_head = MLP([d_model, dcfg.MLP_DIM, dcfg.MLP_DIM // 2, self.m_K * self.m_dim])

    def _map_lane_encode(self, lane_inp):
        polyline = lane_inp[..., :4]
        polyline_type = lane_inp[..., 4].to(int)
        polyline_traf = lane_inp[..., 5].to(int)

        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)

        # agent features
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed

        return line_enc

    def _map_feature_extract(self, line_enc, line_mask, context_agent):
        # map information fusion with CG block
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        # map context feature
        feature = torch.cat([line_enc, context_line.unsqueeze(1).repeat(1, line_enc.shape[1], 1)], dim=-1)

        return feature, context_line

    def _motion_predict(self, result, agent_feat):
        b = agent_feat.shape[0]
        pred_len = self.motion_cfg.STEP

        motion_prob = self.motion_prob_head(agent_feat)
        result['motion_prob'] = motion_prob

        motion_pred = self.motion_head(agent_feat).view(b, -1, self.m_K, pred_len, 2)
        
        if self.motion_cfg.CUMSUM:
            motion_pred = motion_pred.cumsum(dim=-2)
        
        result['pred_motion'] = motion_pred

        if self.motion_cfg.PRED_HEADING_VEL:
            future_heading_pred = self.angle_head(agent_feat).view(b, -1, self.m_K, pred_len, 1)
            future_vel_pred = self.vel_head(agent_feat).view(b, -1, self.m_K, pred_len, 2)
            if self.motion_cfg.CUMSUM:
                future_heading_pred = future_heading_pred.cumsum(dim=-2)
            
            result['pred_future_heading'] = future_heading_pred
            result['pred_future_vel'] = future_vel_pred

    def _output_to_dist(self, para, n):
        if n == 2:
            loc, tril, diag = para[..., :2], para[..., 2], para[..., 3:]

            sigma_1 = torch.exp(diag[..., 0])
            sigma_2 = torch.exp(diag[..., 1])
            rho = torch.tanh(tril)

            cov = torch.stack([sigma_1**2, rho * sigma_1 * sigma_2, rho * sigma_1 * sigma_2, sigma_2**2],
                                dim=-1).view(*loc.shape[:-1], 2, 2)

            distri = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, covariance_matrix=cov)

            return distri

        if n == 1:
            loc, scale = para[..., 0], para[..., 1]
            scale = torch.exp(scale)
            distri = torch.distributions.Normal(loc, scale)

            return distri

    def _output_to_attrs(self, agent_feat):
        result = {}
        result['pred_speed'] = self.speed_head(agent_feat)
        result['pred_vel_heading'] = self.vel_heading_head(agent_feat)
        
        if not self.use_attr_gmm:
            result['pred_pos'] = self.pos_head(agent_feat)
            result['pred_bbox'] = self.bbox_head(agent_feat)
            result['pred_heading'] = self.heading_head(agent_feat)
        else:
            pos_out = self.pos_head(agent_feat).view([*agent_feat.shape[:-1], self.attr_gmm_k, -1])
            pos_weight_logit = pos_out[..., 0]
            pos_param = pos_out[..., 1:]
            pos_distri = self._output_to_dist(pos_param, 2)
            pos_weight = torch.distributions.Categorical(logits=pos_weight_logit)
            result['pred_pos'] = (torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, pos_distri))

            # bbox distribution： 2 dimension length width
            bbox_out = self.bbox_head(agent_feat).view([*agent_feat.shape[:-1], self.attr_gmm_k, -1])
            bbox_weight_logit = bbox_out[..., 0]
            bbox_param = bbox_out[..., 1:]

            bbox_distri = self._output_to_dist(bbox_param, 2)
            bbox_weight = torch.distributions.Categorical(logits=bbox_weight_logit)
            result['pred_bbox'] = (torch.distributions.mixture_same_family.MixtureSameFamily(bbox_weight, bbox_distri))

            # heading distribution: 1 dimension,range(-pi/2,pi/2)
            heading_out = self.heading_head(agent_feat).view([*agent_feat.shape[:-1], self.attr_gmm_k, -1])
            heading_weight_logit = heading_out[..., 0]
            heading_param = heading_out[..., 1:]
            heading_distri = self._output_to_dist(heading_param, 1)
            heading_weight = torch.distributions.Categorical(logits=heading_weight_logit)
            result['pred_heading'] = (torch.distributions.mixture_same_family.MixtureSameFamily(heading_weight, heading_distri))

        return result

    def forward(self, data):
        attr_cfg = self.model_cfg.ATTR_QUERY
        pos_enc_dim = attr_cfg.POS_ENCODING_DIM
        
        # Map Encoder
        b = data['lane_inp'].shape[0]
        device = data['lane_inp'].device
        line_enc = self._map_lane_encode(data['lane_inp'].float())
        empty_context = torch.ones([b, line_enc.shape[-1]]).to(device)
        line_enc, _ = self._map_feature_extract(line_enc, data['lane_mask'], empty_context)
        line_enc = line_enc[:, :data['center_mask'].shape[1]]

        # Agent Query
        attr_query_input = data['text']
        attr_dim = attr_query_input.shape[-1]
        attr_query_encoding = pos2posemb(attr_query_input, pos_enc_dim//attr_dim)

        attr_query_encoding = self.query_embedding_layer(attr_query_encoding)
        learnable_query_embedding = self.actor_query.repeat(b, 1, 1)
        query_encoding = learnable_query_embedding + attr_query_encoding

        # Generative Transformer
        agent_feat = self.decoder(tgt=query_encoding, memory=line_enc, tgt_key_padding_mask=~data['agent_mask'], memory_key_padding_mask=~data['center_mask'])

        # Position MLP + Map Mask MLP
        query_mask = self.query_mask_head(agent_feat)
        memory_mask = self.memory_mask_head(line_enc)
        
        pred_logits = torch.einsum('bqk,bmk->bqm', query_mask, memory_mask)

        if self.use_background:
            background_logits = self.background_head(agent_feat)
            pred_logits = torch.cat([pred_logits, background_logits], dim=-1)

        # Attribute MLP
        result = self._output_to_attrs(agent_feat)
        result['pred_logits'] = pred_logits
        
        # Motion MLP
        if self.motion_cfg.ENABLE:
            self._motion_predict(result, agent_feat)

        return result