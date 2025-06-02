# Copyright adapted for Joern-based usage
# Inst2Vec Joern Graph Encoder

import pickle
import numpy as np
import networkx as nx
from typing import Optional

# Load dictionary and embedding table
DICTIONARY = "inst2vec_augmented_dictionary.pickle"
AUGMENTED_INST2VEC_EMBEDDINGS = "inst2vec_augmented_embeddings.pickle"

def simple_inst2vec_tokenize(code: str) -> str:
    """A simplified tokenizer that replaces identifiers and constants with placeholders."""
    import re
    code = re.sub(r'\b[0-9]+(\.[0-9]+)?\b', '!IMMEDIATE', code)
    code = re.sub(r'[%$a-zA-Z_][a-zA-Z0-9_]*', '!IDENTIFIER', code)
    return code


class Inst2VecJoernEncoder:
    """An encoder for Joern Code Property Graphs using inst2vec."""

    def __init__(self):
        with open(DICTIONARY, "rb") as f:
            self.dictionary = pickle.load(f)

        with open(AUGMENTED_INST2VEC_EMBEDDINGS, "rb") as f:
            self.node_text_embeddings = pickle.load(f)

        self.unk_idx = self.dictionary.get("!UNK", 0)
        self.identifier_idx = self.dictionary.get("!IDENTIFIER", 1)
        self.immediate_idx = self.dictionary.get("!IMMEDIATE", 2)

        # A basic type vocabulary
        self.type_vocab = {
            "int": 1,
            "float": 2,
            "char *": 3,
            "void": 4,
            "unknown": 0
        }

    def encode(self, graph: nx.DiGraph, source_code: Optional[str] = None) -> nx.DiGraph:
        """Annotate Joern graph nodes with inst2vec tokens, embeddings, and type indices."""
        for node_id, node in graph.nodes(data=True):
            code = node.get("code") or node.get("label") or ""
            type_name = node.get("typeFullName", "unknown").lower()

            # Tokenize code
            tokenized = simple_inst2vec_tokenize(code)
            embedding_idx = self.dictionary.get(tokenized, self.unk_idx)

            # Type embedding
            type_idx = self.type_vocab.get(type_name, self.type_vocab["unknown"])

            # Annotate node
            node["inst2vec_token"] = tokenized
            node["inst2vec_embedding"] = embedding_idx
            node["type_embedding"] = type_idx

        graph.graph["inst2vec_annotated"] = True
        return graph

    @property
    def embeddings_tables(self) -> list[np.ndarray]:
        """Return the inst2vec and selector embedding tables."""
        node_selector = np.vstack([[1, 0], [0, 1]]).astype(np.float64)
        return [self.node_text_embeddings, node_selector]
