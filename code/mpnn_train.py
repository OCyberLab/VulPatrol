# mpnn_train.py

import os
import json
import glob
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import degree
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna


#  HYPERPARAMETERS & DEFAULTS 

DEFAULT_NODE_VOCAB_SIZE    = 1000    # (set this to the actual size of your node‐type vocabulary)
DEFAULT_EDGE_VOCAB_SIZE    = 10      # (set this to the actual size of your edge‐type vocabulary)

DEFAULT_EMBED_DIM          = 200     # node‐embedding dim and edge‐embedding dim
DEFAULT_HIDDEN_DIM         = 200     # hidden size for message‐passing and GRU
DEFAULT_READ_HIDDEN_DIM    = 512     # hidden size for the "readout" feed‐forward layer

DEFAULT_T_PASSES           = 5       # number of message‐passing steps (we will sweep 3..8)
DEFAULT_DROPOUT_RATE       = 0.2     # dropout on hidden states between layers
DEFAULT_BATCH_SIZE         = 128     # batch size
DEFAULT_LR                 = 1e-4    # Adam learning rate
DEFAULT_NUM_EPOCHS         = 300     # max epochs
DEFAULT_PATIENCE           = 10      # early stopping patience on F1
DEFAULT_DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  1) PyG DATASET: Load JSON graph from ./graphs/

class JSONGraphDataset(Dataset):
    """
    Each JSON file in `root_dir` should have the structure:
      {
        "nodes": [ {"id": 0, "feat_id": 42}, {"id":1, "feat_id":17}, … ],
        "edges": [ {"source":0, "target":1, "etype_id":3}, … ],
        "label": 0   ︙ or 1
      }
    We convert it to a PyG Data object with:
      - data.x: [num_nodes, 1] (long) = the feat_id per node
      - data.edge_index: [2, num_edges] (long)
      - data.edge_attr: [num_edges, 1] (long) = the etype_id
      - data.y: [1] (float) = 0.0 or 1.0
    """
    def __init__(self, root_dir: str):
        super().__init__(root=root_dir)
        self.json_files = sorted(glob.glob(os.path.join(root_dir, "*.json")))

    def len(self) -> int:
        return len(self.json_files)

    def get(self, idx: int) -> Data:
        path = self.json_files[idx]
        with open(path, "r") as f:
            js = json.load(f)

        # --- 1a) Node features as integer feat_id ---
        node_feat_ids = [node["feat_id"] for node in js["nodes"]]
        x = torch.tensor(node_feat_ids, dtype=torch.long).view(-1, 1)
        # shape: [num_nodes, 1]

        # --- 1b) Edge index and edge_type IDs ---
        src_list, tgt_list, etype_list = [], [], []
        for e in js["edges"]:
            src_list.append(e["source"])
            tgt_list.append(e["target"])
            etype_list.append(e["etype_id"])
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_attr  = torch.tensor(etype_list, dtype=torch.long).view(-1, 1)
        # edge_index: [2, num_edges], edge_attr: [num_edges, 1]

        # --- 1c) Graph‐level label (0.0 or 1.0) ---
        y = torch.tensor([js["label"]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data




class MPNNLayer(MessagePassing):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        weight_sharing: bool = True
    ):
        """
        A single MPNN layer that does one round of:
          1) m_v = sum_{w in N(v)} A(h_w, e_{w->v})
          2) h_v' = GRU(h_v, m_v)
        
        """
        super().__init__(aggr="add")  # sum‐aggregation

        # --- 1) 
        self.node_embedding = nn.Embedding(node_vocab_size, embed_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size, embed_dim)

        # If embed_dim != hidden_dim, we project up to hidden_dim before GRU
        self.input_proj = nn.Identity()
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # --- 2) Message function A
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- 3) GRU cell
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # --- 4) Dropout on hidden states before passing to next layer ---
        self.dropout = nn.Dropout(p=dropout)

     
        self.weight_sharing = weight_sharing

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, 1] long = feat_id. We embed → [num_nodes, embed_dim], then maybe project → [num_nodes, hidden_dim].
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 1] long = etype_id

        Returns: h_final: [num_nodes, hidden_dim] after one round of message passing.
        """
        # 1) initial node embedding: [num_nodes, embed_dim]
        h0_embed = self.node_embedding(x.view(-1))  # [n, embed_dim]
        # 2) project if needed to hidden_dim
        h = self.input_proj(h0_embed)             # [n, hidden_dim]

        # 3) store original h^0 for readout 

        # 4) store edge embeddings up front for efficiency
        e_emb = self.edge_embedding(edge_attr.view(-1))  # [num_edges, embed_dim]

        # 5) message passing: we compute m_v 
        h = self.propagate(edge_index, x_hidden=h, edge_emb=e_emb)

        # 6) apply dropout on the updated hidden states
        h = self.dropout(h)
        return h

    def message(self, x_hidden_j: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        """
        x_hidden_j: [num_edges, hidden_dim] = h_w^{(t-1)} for each source w
        edge_emb:     [num_edges, embed_dim]   = e_{w->v} embedding
        We need to compute A( [h_w, e_{w->v}] ) → hidden_dim
        """
        cat = torch.cat([x_hidden_j, edge_emb], dim=1)  # [num_edges, hidden_dim + embed_dim]
        msg = self.msg_mlp(cat)                         # [num_edges, hidden_dim]
        return msg

    def update(self, aggr_out: torch.Tensor, x_hidden: torch.Tensor) -> torch.Tensor:
        """
        aggr_out: [num_nodes, hidden_dim] = sum of messages m_v
        x_hidden: [num_nodes, hidden_dim] = previous h_v^{(t-1)}
        We do: h_v^{(t)} = GRUCell(m_v, h_v)
        """
        new_h = self.gru(aggr_out, x_hidden)  # [num_nodes, hidden_dim]
        return new_h


class MPNNClassifier(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        read_hidden_dim: int,
        num_message_passes: int,
        dropout: float
    ):
        """
        - node_vocab_size, edge_vocab_size: cardinalities for embeddings
        - embed_dim: node‐embedding dim (also edge‐embedding dim)
        - hidden_dim: hidden size for MPNN (GRU hidden size)
        - read_hidden_dim: hidden size for the readout MLP
        - num_message_passes: how many times to apply the same MPNNLayer (weight sharing)
        - dropout: dropout rate on graph states between passes
        """
        super().__init__()

        self.num_message_passes = num_message_passes

        # One shared MPNNLayer (we will call it T times)
        self.mpnn_layer = MPNNLayer(
            node_vocab_size=node_vocab_size,
            edge_vocab_size=edge_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            weight_sharing=True
        )

        
        
        
        self.node_embedding_for_read = self.mpnn_layer.node_embedding
        self.input_proj_for_read  = self.mpnn_layer.input_proj

        # i(·): MLP from h^T → scalar logit
        self.read_i = nn.Sequential(
            nn.Linear(hidden_dim, read_hidden_dim),
            nn.ReLU(),
            nn.Linear(read_hidden_dim, 1)
        )
        # j(·): MLP from h^0 → scalar logit
        self.read_j = nn.Sequential(
            nn.Linear(hidden_dim, read_hidden_dim),
            nn.ReLU(),
            nn.Linear(read_hidden_dim, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        data.x: [num_nodes_total, 1] long = feat_id
        data.edge_index: [2, num_edges_total]
        data.edge_attr: [num_edges_total, 1] long = etype_id
        data.batch: [num_nodes_total] which graph each node belongs to

        Returns:
          graph_logits: [num_graphs_in_batch] unbounded (to use BCEWithLogitsLoss)
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_nodes = x.shape[0]

        # 1) Compute h^0: embed x and project
        h0_emb = self.node_embedding_for_read(x.view(-1))    # [n_total, embed_dim]
        h0     = self.input_proj_for_read(h0_emb)            # [n_total, hidden_dim]

        # 2) Initialize h = h^0
        h = h0.clone()

        # 3) Run T message‐passing steps (weight sharing)
        for _ in range(self.num_message_passes):
            h = self.mpnn_layer.forward(x, edge_index, edge_attr)

        hT = h  # [n_total, hidden_dim]

        # 4) Readout per node: R_v = σ( i(hT_v) + j(h0_v) )
        i_score = self.read_i(hT).view(-1)  # [n_total]
        j_score = self.read_j(h0).view(-1)  # [n_total]
        node_logits = i_score + j_score     # [n_total]
        node_probs  = torch.sigmoid(node_logits)

        # 5) Sum over nodes for each graph: R_G = sum_{v in graph} R_v
        graph_scores = global_add_pool(node_probs.unsqueeze(1), batch).view(-1)
        # shape: [batch_size, 1] → .view(-1) → [batch_size]

        return graph_scores  # unbounded logits (BCEWithLogitsLoss expects that)


# train/val/test (8:1:1)

def stratified_split(
    dataset: JSONGraphDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[JSONGraphDataset, JSONGraphDataset, JSONGraphDataset]:
    """
    Return (train_ds, val_ds, test_ds) each as JSONGraphDataset but with a subset of files.
    We do stratified sampling on the labels (0/1) to maintain the same positive/negative ratio.
    """
    # 1) Collect indices by label
    labels = []
    for path in dataset.json_files:
        with open(path, "r") as f:
            js = json.load(f)
            labels.append(int(js["label"]))
    labels = np.array(labels)

    # 2) Indices of class 0 and class 1
    idx0 = np.where(labels == 0)[0].tolist()
    idx1 = np.where(labels == 1)[0].tolist()

    random.seed(seed)
    random.shuffle(idx0)
    random.shuffle(idx1)

    def split_indices(idxs):
        n_total = len(idxs)
        n_val   = int(n_total * val_ratio)
        n_test  = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test
        return idxs[:n_train], idxs[n_train:n_train+n_val], idxs[n_train+n_val:]

    train0, val0, test0 = split_indices(idx0)
    train1, val1, test1 = split_indices(idx1)

    train_idxs = train0 + train1
    val_idxs   = val0 + val1
    test_idxs  = test0 + test1

    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    def subset_dataset(original: JSONGraphDataset, indices: list):
        sub = JSONGraphDataset(root_dir=original.root)
        sub.json_files = [original.json_files[i] for i in indices]
        return sub

    return (
        subset_dataset(dataset, train_idxs),
        subset_dataset(dataset, val_idxs),
        subset_dataset(dataset, test_idxs)
    )


#  4) TRAIN/EVAL FUNCTIONS (compute F1)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Run one epoch of training. Returns average training loss over all graphs.
    """
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_data)             # [batch_size]
        labels = batch_data.y.view(-1).to(device)  # [batch_size]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs
        total_graphs += batch_data.num_graphs

    return total_loss / total_graphs


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Compute loss + F1‐score on the dataset given by loader.
    Returns (avg_loss, f1_score).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_labels = []

    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)            # [batch_size]
            labels = batch_data.y.view(-1).to(device)  # [batch_size]

            loss = criterion(logits, labels)
            total_loss += loss.item() * batch_data.num_graphs
            total_graphs += batch_data.num_graphs

            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())

    avg_loss = total_loss / total_graphs

    # Convert logits → binary preds with threshold 0.0 (since BCEWithLogits)
    preds = [1 if logit > 0.0 else 0 for logit in all_logits]
    f1 = f1_score(all_labels, preds, zero_division=0)

    return avg_loss, f1


#  5) MAIN TRAINING + EARLY STOPPING + OPTIONAL BAYESIAN OPTIMIZATION
# 
def objective(trial: optuna.Trial) -> float:
    """
      to maximize validation F1. We sweep over:
       - T in [3, 8]
       - dropout in [0.0, 0.5]
       - ? embed_dim in [128, 256], hidden_dim in [128, 256]
      
    """
    # 5.1) Sample hyperparameters
    T_passes  = trial.suggest_int("T_passes", 3, 8)
    dropout   = trial.suggest_float("dropout", 0.0, 0.5)
    embed_dim = trial.suggest_categorical("embed_dim", [128, 200, 256])
    hidden_dim= trial.suggest_categorical("hidden_dim",[128, 200, 256])
    read_dim  = trial.suggest_categorical("read_hidden_dim",[256,512,1024])
    lr        = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_sz  = trial.suggest_categorical("batch_size",[64, 128, 256])

    # Early exit if dims do not match for GRU (we require hidden_dim == projected embed_dim)
    if embed_dim != hidden_dim:
        # We could project embed→hidden, but to keep code simple (no re‐wiring), skip bad combos
        return 0.0

    # 5.2) Build fresh model with these HPs
    model = MPNNClassifier(
        node_vocab_size=DEFAULT_NODE_VOCAB_SIZE,
        edge_vocab_size=DEFAULT_EDGE_VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        read_hidden_dim=read_dim,
        num_message_passes=T_passes,
        dropout=dropout
    ).to(DEFAULT_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 5.3) Load & split dataset
    full_ds = JSONGraphDataset(root_dir="./graphs")
    train_ds, val_ds, _ = stratified_split(full_ds, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False)

    # 5.4) Training loop with early stopping on val‐F1
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, DEFAULT_NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEFAULT_DEVICE)
        val_loss, val_f1 = evaluate(model, val_loader, DEFAULT_DEVICE)

        # Report intermediate value to Optuna (for pruning, if enabled)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping logic
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save checkpoint of best model within this trial
            torch.save(model.state_dict(), "temp_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_PATIENCE:
                break

    # 5.5) Load best model for this trial & return best_val_f1
    model.load_state_dict(torch.load("temp_best_model.pt"))
    return best_val_f1


def main(do_hpo: bool = True):
    # 5a) If we want to do Bayesian HPO, run Optuna study to maximize F1
    if do_hpo:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, timeout=3600)  # e.g. 20 trials or 1 hour

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (val‐F1): {trial.value:.4f}")
        print("  Params: ")
        for key, val in trial.params.items():
            print(f"    {key}: {val}")

        # Once we have best hyperparams, we fall through to final training & test
        best_params = trial.params
    else:
        # If no HPO, use default hyperparameters
        best_params = {
            "T_passes": DEFAULT_T_PASSES,
            "dropout": DEFAULT_DROPOUT_RATE,
            "embed_dim": DEFAULT_EMBED_DIM,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "read_hidden_dim": DEFAULT_READ_HIDDEN_DIM,
            "lr": DEFAULT_LR,
            "batch_size": DEFAULT_BATCH_SIZE
        }

    # 5b) Build final model using best_params
    model = MPNNClassifier(
        node_vocab_size=DEFAULT_NODE_VOCAB_SIZE,
        edge_vocab_size=DEFAULT_EDGE_VOCAB_SIZE,
        embed_dim=best_params["embed_dim"],
        hidden_dim=best_params["hidden_dim"],
        read_hidden_dim=best_params["read_hidden_dim"],
        num_message_passes=best_params["T_passes"],
        dropout=best_params["dropout"]
    ).to(DEFAULT_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    # 5c) Reload full dataset, split 8:1:1 (train/val/test)
    full_ds = JSONGraphDataset(root_dir="./graphs")
    train_ds, val_ds, test_ds = stratified_split(full_ds, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=best_params["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=best_params["batch_size"], shuffle=False)

    # 5d) Train on train+val combined (optional), or just re‐train using train & validate
    #     Here we re‐train from scratch using train (for simplicity), but you can combine train+val if desired.
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, DEFAULT_NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEFAULT_DEVICE)
        val_loss, val_f1 = evaluate(model, val_loader, DEFAULT_DEVICE)
        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val F1 {val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "final_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_PATIENCE:
                print("Early stopping on final training.")
                break

    # 5e) Load best model and evaluate on test set
    model.load_state_dict(torch.load("final_best_model.pt"))
    test_loss, test_f1 = evaluate(model, test_loader, DEFAULT_DEVICE)
    print(f"Final Test Loss: {test_loss:.4f} | Final Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    # Set do_hpo=True to run Bayesian optimization first, or False to skip HPO and use defaults.
    main(do_hpo=True)
