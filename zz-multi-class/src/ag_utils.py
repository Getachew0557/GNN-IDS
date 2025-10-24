"""
Attack Graph Utilities for parsing and processing attack graphs
"""

import torch
import re
import networkx as nx
from typing import Dict, List, Tuple, Any

class Dictionary:
    """
    Vocabulary dictionary for mapping words to indices and vice versa
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def remove_word(self, word: str) -> None:
        """Remove a word from the dictionary"""
        if word in self.word2idx:
            idx = self.word2idx[word]
            del self.word2idx[word]
            del self.idx2word[idx]
            # Reindex remaining words
            new_idx2word = {}
            new_word2idx = {}
            new_idx = 0
            for old_idx, w in sorted(self.idx2word.items()):
                new_idx2word[new_idx] = w
                new_word2idx[w] = new_idx
                new_idx += 1
            self.idx2word = new_idx2word
            self.word2idx = new_word2idx
            self.idx = new_idx
    
    def __len__(self) -> int:
        return len(self.word2idx)


class Corpus:
    """
    Corpus for processing attack graph node features and properties
    """
    def __init__(self, node_dict: Dict):
        self.dictionary = Dictionary()
        self.num_tokens = 0
        self.node_dict = node_dict
        self._build_vocabulary()
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary from node dictionary"""
        tokens = 0
        for node_id, node in self.node_dict.items():
            words = [node['predicate']] + node['attributes']
            tokens += len(words)
            for word in words: 
                self.dictionary.add_word(word)  
        self.num_tokens = tokens
    
    def get_node_features(self) -> torch.Tensor:
        """Convert node dictionary to feature tensor using one-hot encoding"""
        node_features = torch.zeros(len(self.node_dict), len(self.dictionary))
        
        for idx, node in self.node_dict.items():
            words = [node['predicate']] + node['attributes']
            for word in words:
                if word in self.dictionary.word2idx:
                    node_features[idx][self.dictionary.word2idx[word]] = 1
        
        return node_features
    
    def get_node_types(self) -> List[str]:
        """Extract node types (shapes) from node dictionary"""
        return [node['shape'] for idx, node in sorted(self.node_dict.items())]
    
    def get_action_nodes(self) -> Dict[int, Dict]:
        """Extract action nodes (diamond shape) from node dictionary"""
        return {idx: node for idx, node in self.node_dict.items() 
                if node['shape'] == 'diamond'}
    
    def get_num_tokens(self) -> int:
        return self.num_tokens


def parse_ag_file(attack_graph_path: str) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    Parse attack graph DOT file and extract nodes, edges, and properties
    
    Args:
        attack_graph_path: Path to the attack graph DOT file
        
    Returns:
        Tuple of (nodes, edges, node_properties)
    """
    try:
        with open(attack_graph_path, 'r') as file:
            dot_contents = file.read()

        # Regex patterns for parsing DOT format
        node_pattern = r'(\w+)\s*\[.*?\];'
        edge_pattern = r'(\w+)\s*->\s*(\w+).*?;'
        node_properties_pattern = r'\[(.+)\]'

        # Extract components
        nodes = re.findall(node_pattern, dot_contents)
        edges = re.findall(edge_pattern, dot_contents)
        node_properties = re.findall(node_properties_pattern, dot_contents)

        return nodes, edges, node_properties
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Attack graph file not found: {attack_graph_path}")
    except Exception as e:
        raise Exception(f"Error parsing attack graph file: {str(e)}")


def parse_node_properties(nodes: List[str], node_properties: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Parse node properties and create structured node dictionary
    
    Args:
        nodes: List of node identifiers
        node_properties: List of node property strings
        
    Returns:
        Dictionary mapping node indices to their properties
    """
    node_dict = {}
    
    for item in node_properties:
        # Extract label
        label_match = re.search('label="(.*)"', item)
        if not label_match:
            continue
            
        property_list = label_match.group(1).split(':')
        if len(property_list) < 3:
            continue
            
        node_id = nodes.index(property_list[0])
        node_prop = property_list[1]
        node_compromise_prob = float(property_list[2])

        # Parse predicate and attributes
        pattern = r'(.+)\((.*)\)'
        resp = re.findall(pattern, node_prop)
        if not resp:
            continue
            
        predicate = resp[0][0].strip()
        attr = resp[0][1].strip()
        
        # Process attributes
        if ',' in attr:
            attributes = [a.strip() for a in attr.split(',')]
        else:
            attributes = attr.split()

        # Clean attribute values
        attributes = [a.strip("'").strip('"') for a in attributes]

        node_dict[node_id] = {
            'predicate': predicate, 
            'attributes': attributes, 
            'possibility': node_compromise_prob
        }

        # Extract shape
        shape_match = re.search('shape=(.*)', item)
        node_shape = shape_match.group(1) if shape_match else 'ellipse'
        node_dict[node_id]['shape'] = node_shape

    return node_dict


def create_networkx_graph(nodes: List[str], edges: List[Tuple[str, str]]) -> nx.DiGraph:
    """
    Create NetworkX graph from parsed nodes and edges
    
    Args:
        nodes: List of node identifiers
        edges: List of edge tuples (source, target)
        
    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from([(nodes.index(src), nodes.index(tgt)) for src, tgt in edges])
    return G


def get_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate comprehensive graph statistics
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary containing graph statistics
    """
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_directed': G.is_directed(),
        'is_connected': nx.is_weakly_connected(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'diameter': nx.diameter(G) if nx.is_weakly_connected(G) else float('inf')
    }