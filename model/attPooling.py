"""
Multi-Head Attention Pooling for Graph Representation

Implements global graph pooling using multi-head attention with:
- Learnable class token (like Vision Transformer)
- Edge-aware attention masking
- Multiple attention heads for different aspects
- Layer normalization and dropout

Author: Enhanced for LENS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling
    
    Uses a learnable class token that attends to all nodes
    to produce a global graph representation.
    
    Architecture:
        1. Prepend learnable class token to node features
        2. Multi-head self-attention across [class_token, nodes]
        3. Extract class token output as graph representation
    
    Edge-aware: Attention scores are masked by learned edge weights
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2, use_edge_masking=True):
        """
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.2)
            use_edge_masking: Use edge weights to mask attention (default: True)
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_masking = use_edge_masking
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable class token (CLS token like in BERT/ViT)
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Initialize weights
        self._init_weights()
        
        print(f"[MultiHeadAttentionPooling] Initialized:")
        print(f"  • Hidden dim: {hidden_dim}")
        print(f"  • Num heads: {num_heads}")
        print(f"  • Head dim: {self.head_dim}")
        print(f"  • Dropout: {dropout}")
        print(f"  • Edge masking: {use_edge_masking}")
    
    def _init_weights(self):
        """Initialize attention weights"""
        # Xavier initialization for Q, K, V
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # Initialize biases to zero
        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)
        nn.init.constant_(self.output_proj.bias, 0)
        
        # Initialize class token
        nn.init.normal_(self.class_token, mean=0.0, std=0.02)
    
    def create_attention_mask(self, edge_weights, adj_matrix, masks=None):
        """
        Create attention mask from edge weights
        
        Mask structure:
            [CLS, N1, N2, ..., Nn]
        CLS: Attends to all valid nodes (based on masks)
        Ni:  Attends to CLS + neighbors (based on edge_weights)
        
        Args:
            edge_weights: Edge gates [B, N, N]
            adj_matrix: Original adjacency [B, N, N]
            masks: Node masks [B, N] (optional)
        
        Returns:
            attn_mask: Attention mask [B, 1+N, 1+N]
        """
        batch_size, num_nodes = edge_weights.shape[:2]
        device = edge_weights.device
        
        if not self.use_edge_masking:
            # No edge masking - allow all attention
            return None
        
        # Initialize mask: [B, 1+N, 1+N]
        attn_mask = torch.zeros(batch_size, 1 + num_nodes, 1 + num_nodes, device=device)
        
        # Part 1: CLS token attention (first row)
        # CLS can attend to itself
        attn_mask[:, 0, 0] = 1.0
        
        # CLS attends to valid nodes
        if masks is not None:
            attn_mask[:, 0, 1:] = masks  # [B, N]
        else:
            attn_mask[:, 0, 1:] = 1.0  # Attend to all nodes
        
        # Part 2: Node attention (remaining rows)
        # Nodes attend to CLS token (first column)
        attn_mask[:, 1:, 0] = 1.0
        
        # Nodes attend to neighbors via edge weights
        # Use edge_weights directly as attention strengths
        attn_mask[:, 1:, 1:] = edge_weights
        
        # Add self-loops (nodes attend to themselves)
        eye = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        attn_mask[:, 1:, 1:] = attn_mask[:, 1:, 1:] + eye
        attn_mask = torch.clamp(attn_mask, max=1.0)
        
        # Apply node masks if provided
        if masks is not None:
            # Mask out invalid nodes in key dimension
            node_mask_key = masks.unsqueeze(1)  # [B, 1, N]
            attn_mask[:, :, 1:] = attn_mask[:, :, 1:] * node_mask_key
            
            # Mask out invalid nodes in query dimension
            node_mask_query = masks.unsqueeze(2)  # [B, N, 1]
            attn_mask[:, 1:, :] = attn_mask[:, 1:, :] * node_mask_query
        
        return attn_mask
    
    def forward(self, node_feat, edge_weights, adj_matrix, masks=None):
        """
        Forward pass: pool graph into single representation
        
        Args:
            node_feat: Node features [B, N, D]
            edge_weights: Edge gates [B, N, N]
            adj_matrix: Original adjacency [B, N, N]
            masks: Node validity masks [B, N] (optional)
        
        Returns:
            graph_rep: Global graph representation [B, D]
        """
        batch_size, num_nodes, hidden_dim = node_feat.shape
        device = node_feat.device
        
        # 1. Prepend class token
        class_token = self.class_token.expand(batch_size, -1, -1)  # [B, 1, D]
        x = torch.cat([class_token, node_feat], dim=1)  # [B, 1+N, D]
        
        # 2. Create attention mask
        attn_mask = self.create_attention_mask(edge_weights, adj_matrix, masks)
        
        # 3. Multi-head attention
        # Compute Q, K, V
        Q = self.query(x)  # [B, 1+N, D]
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head: [B, 1+N, D] → [B, H, 1+N, D/H]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: [B, H, 1+N, 1+N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attn_mask is not None:
            # Expand mask for all heads: [B, 1+N, 1+N] → [B, H, 1+N, 1+N]
            attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Mask: set masked positions to large negative value
            scores = scores.masked_fill(attn_mask_expanded == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1+N, 1+N]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, 1+N, D/H]
        
        # Concatenate heads: [B, H, 1+N, D/H] → [B, 1+N, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, hidden_dim)
        
        # Output projection
        output = self.output_proj(attn_output)  # [B, 1+N, D]
        output = self.dropout(output)
        
        # Residual connection + layer norm
        output = self.norm(output + x)
        
        # 4. Extract class token as graph representation
        graph_rep = output[:, 0, :]  # [B, D]
        
        return graph_rep
    
    def get_attention_weights(self, node_feat, edge_weights, adj_matrix, masks=None):
        """
        Get attention weights for visualization
        
        Returns:
            attn_weights: [B, H, 1+N, 1+N] attention weights
        """
        batch_size, num_nodes, hidden_dim = node_feat.shape
        
        # Prepend class token
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, node_feat], dim=1)
        
        # Create mask
        attn_mask = self.create_attention_mask(edge_weights, adj_matrix, masks)
        
        # Compute Q, K
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(attn_mask_expanded == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights
    
    def __repr__(self):
        return (f"MultiHeadAttentionPooling(\n"
                f"  hidden_dim={self.hidden_dim},\n"
                f"  num_heads={self.num_heads},\n"
                f"  head_dim={self.head_dim},\n"
                f"  edge_masking={self.use_edge_masking}\n"
                f")")
