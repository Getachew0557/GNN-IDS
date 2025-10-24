"""
Tests for attack graph utilities.
"""

import pytest
import torch
import tempfile
import os
from src.ag_utils import Dictionary, Corpus, parse_ag_file, parse_node_properties


class TestDictionary:
    """Test cases for Dictionary class."""
    
    def test_initialization(self):
        """Test dictionary initialization."""
        dictionary = Dictionary()
        assert len(dictionary) == 0
        assert dictionary.idx == 0
    
    def test_add_word(self):
        """Test adding words to dictionary."""
        dictionary = Dictionary()
        dictionary.add_word("test")
        dictionary.add_word("attack")
        
        assert len(dictionary) == 2
        assert "test" in dictionary.word2idx
        assert "attack" in dictionary.word2idx
        assert dictionary.word2idx["test"] == 0
        assert dictionary.word2idx["attack"] == 1
    
    def test_duplicate_word(self):
        """Test adding duplicate words."""
        dictionary = Dictionary()
        dictionary.add_word("test")
        dictionary.add_word("test")  # Duplicate
        
        assert len(dictionary) == 1  # Should not add duplicate
    
    def test_bidirectional_mapping(self):
        """Test word to index and index to word mappings."""
        dictionary = Dictionary()
        dictionary.add_word("test")
        dictionary.add_word("attack")
        
        assert dictionary.idx2word[0] == "test"
        assert dictionary.idx2word[1] == "attack"
        assert dictionary.word2idx["test"] == 0
        assert dictionary.word2idx["attack"] == 1


class TestCorpus:
    """Test cases for Corpus class."""
    
    @pytest.fixture
    def sample_node_dict(self):
        """Sample node dictionary for testing."""
        return {
            0: {'predicate': 'execCode', 'attributes': ['workStation', 'root'], 'shape': 'diamond'},
            1: {'predicate': 'accessFile', 'attributes': ['workStation', 'write', '/usr/share'], 'shape': 'ellipse'}
        }
    
    def test_corpus_initialization(self, sample_node_dict):
        """Test corpus initialization with node dictionary."""
        corpus = Corpus(sample_node_dict)
        
        assert len(corpus.dictionary) > 0
        assert corpus.num_tokens == 7  # 2 predicates + 5 attributes
    
    def test_node_features(self, sample_node_dict):
        """Test node feature generation."""
        corpus = Corpus(sample_node_dict)
        features = corpus.get_node_features()
        
        assert features.shape == (2, len(corpus.dictionary))
        assert torch.is_tensor(features)
        # Features should be one-hot encoded
        assert torch.all(features.sum(dim=1) == torch.tensor([3., 4.]))  # 3 and 4 words per node
    
    def test_action_nodes(self, sample_node_dict):
        """Test action node filtering."""
        corpus = Corpus(sample_node_dict)
        action_nodes = corpus.get_action_nodes()
        
        assert 0 in action_nodes  # diamond shape
        assert 1 not in action_nodes  # ellipse shape


class TestAttackGraphParsing:
    """Test cases for attack graph parsing utilities."""
    
    @pytest.fixture
    def sample_dot_file(self):
        """Create a sample DOT file for testing."""
        dot_content = '''digraph G {
    1 [label="1:execCode(workStation,root):0",shape=diamond];
    2 [label="2:accessFile(workStation,write,/usr/share):0",shape=ellipse];
    1 -> 2;
}'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_parse_ag_file(self, sample_dot_file):
        """Test parsing attack graph file."""
        nodes, edges, node_properties = parse_ag_file(sample_dot_file)
        
        assert len(nodes) == 2
        assert len(edges) == 1
        assert len(node_properties) == 2
        assert ('1', '2') in edges
    
    def test_parse_node_properties(self, sample_dot_file):
        """Test node property parsing."""
        nodes, edges, node_properties = parse_ag_file(sample_dot_file)
        node_dict = parse_node_properties(nodes, node_properties)
        
        assert len(node_dict) == 2
        assert 0 in node_dict  # Node ID 1 becomes index 0
        assert node_dict[0]['predicate'] == 'execCode'
        assert node_dict[0]['shape'] == 'diamond'
        assert 'workStation' in node_dict[0]['attributes']
        assert node_dict[0]['possibility'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])