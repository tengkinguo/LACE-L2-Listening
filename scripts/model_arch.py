# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:26:04 2025

@author: T
"""
import torch
import torch.nn as nn

class LACEModel(nn.Module):
    """
    LACE: Lexico-phonological Approach to Cognitive Load Estimation.
    
    This architecture supports both the streamlined 'E1_LACE' (Lexical + Phonological + History)
    and the full 'E2_Full' (which adds User/Country/Client context embeddings).
    """
    def __init__(self, sizes, hparams, config):
        super().__init__()
        self.sizes = sizes
        self.config = config
        d_model = hparams["d_model"]

        # ------------------------------------------------------------------
        # 1. Sequence Modeling (Transformer Pathway)
        # ------------------------------------------------------------------
        self.q_embedding = nn.Embedding(sizes["num_questions"], d_model, padding_idx=0)
        self.r_embedding = nn.Embedding(3, d_model, padding_idx=0) # 0:pad, 1:wrong, 2:correct
        self.position_embedding = nn.Embedding(hparams["max_seq_len"], d_model)

        if config.get("use_hist_pos_phoneme"):
            self.pos_embedding = nn.Embedding(sizes["num_pos_tags"], d_model, padding_idx=0)
            self.phoneme_embedding = nn.Embedding(sizes["num_phonemes"], d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=hparams["n_head"],
            dim_feedforward=hparams["ff_dim"],
            dropout=hparams["dropout_rate"],
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_layers"])

        # ------------------------------------------------------------------
        # 2. Feature Fusion Pathways
        # ------------------------------------------------------------------
        fusion_dim = d_model # Start with Transformer output dimension

        # Pathway A: Lexical Features (Frequency, Length)
        if config.get("use_lexical"):
            self.lexical_mlp = nn.Sequential(
                nn.Linear(2, d_model), 
                nn.ReLU(),
                nn.Dropout(hparams["dropout_rate"]),
            )
            fusion_dim += d_model

        # Pathway B: Phonological Features (Syllables, Challenging Flag)
        if config.get("use_phoneme"):
            self.phoneme_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Dropout(hparams["dropout_rate"]),
            )
            fusion_dim += d_model

        # Pathway C: Previous Interaction State (Token + Label)
        if config.get("use_prev_interaction"):
            self.prev_label_embedding = nn.Embedding(3, d_model, padding_idx=0)
            fusion_dim += d_model

        # Pathway D: Context Features (User, Client, Country, Session, Exercise)
        # Only used in E2_Full configuration
        if config.get("use_context"):
            self.user_embedding = nn.Embedding(sizes["num_users"], d_model)
            self.client_embedding = nn.Embedding(sizes["num_clients"], d_model)
            self.country_embedding = nn.Embedding(sizes["num_countries"], d_model)
            self.session_embedding = nn.Embedding(sizes["num_sessions"], d_model)
            self.exercise_embedding = nn.Embedding(sizes["num_exercises"], d_model)

            self.context_mlp = nn.Sequential(
                nn.Linear(d_model * 5, d_model), # 5 context embeddings concatenated
                nn.ReLU(),
                nn.Dropout(hparams["dropout_rate"]),
            )
            fusion_dim += d_model

        # ------------------------------------------------------------------
        # 3. Final Classification Head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(hparams["dropout_rate"]),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch, return_features=False):
        B, L = batch["q_seq"].size()
        device = batch["q_seq"].device

        # --- A. Sequence Encoding ---
        x = self.q_embedding(batch["q_seq"].long()) + \
            self.r_embedding(batch["r_seq"].long() + 1) + \
            self.position_embedding(torch.arange(L, device=device).unsqueeze(0))

        if self.config.get("use_hist_pos_phoneme"):
            x += self.pos_embedding(batch["pos_seq"].long())
            # Sum phoneme embeddings for each token (bag-of-phonemes)
            x += self.phoneme_embedding(batch["phoneme_seq"].long()).sum(dim=2)

        mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
        h = self.transformer(x, mask)
        h_last = h[:, -1, :] # Take the last hidden state

        # --- B. Feature Aggregation ---
        features = {"hist": h_last}
        fusion_list = [h_last]

        if self.config.get("use_lexical"):
            lex = self.lexical_mlp(batch["next_lexical"].float())
            fusion_list.append(lex)
            features["lex"] = lex

        if self.config.get("use_phoneme"):
            pho = self.phoneme_mlp(batch["next_phoneme"].float())
            fusion_list.append(pho)
            features["pho"] = pho

        if self.config.get("use_prev_interaction"):
            prev_ids = batch["next_prev_interaction"].long()
            # Combine Prev Token Embedding + Prev Label Embedding
            prev_feat = self.q_embedding(prev_ids[:, 0]) + self.prev_label_embedding(prev_ids[:, 1] + 1)
            fusion_list.append(prev_feat)
            features["prev"] = prev_feat
        
        if self.config.get("use_context"):
            # next_context shape: [B, 5] -> [User, Client, Country, Session, Exercise]
            ctx_ids = batch["next_context"].long()
            
            u_emb = self.user_embedding(ctx_ids[:, 0])
            cli_emb = self.client_embedding(ctx_ids[:, 1])
            ctr_emb = self.country_embedding(ctx_ids[:, 2])
            sess_emb = self.session_embedding(ctx_ids[:, 3])
            ex_emb = self.exercise_embedding(ctx_ids[:, 4])
            
            ctx_feat = self.context_mlp(
                torch.cat([u_emb, cli_emb, ctr_emb, sess_emb, ex_emb], dim=1)
            )
            fusion_list.append(ctx_feat)
            features["ctx"] = ctx_feat

        # --- C. Prediction ---
        logits = self.classifier(torch.cat(fusion_list, dim=1)).squeeze(-1)

        if return_features:
            return logits, features
        return logits