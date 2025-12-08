"""
Test suite for CIO (Chief Investment Office) functions.
Tests the AI Search functionality for investment research and CIO views.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from cio_functions import CIOSearchFunctions, search_cio


class TestCIOSearchFunctions:
    """Test cases for CIO Search Functions."""
    
    @patch.dict(os.environ, {
        'AI_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AI_SEARCH_INDEX_NAME': 'test-cio-index',
        'AI_SEARCH_SEMANTIC_CONFIG_NAME': 'test-semantic-config',
        'AI_SEARCH_VECTOR_FIELD_NAME': 'testVector'
    })
    @patch('cio_functions.SearchClient')
    @patch('cio_functions.DefaultAzureCredential')
    def test_cio_search_functions_init(self, mock_credential, mock_search_client):
        """Test CIOSearchFunctions initialization."""
        cio_search = CIOSearchFunctions()
        
        # Verify environment variables are read correctly
        assert cio_search.search_endpoint == 'https://test-search.search.windows.net'
        assert cio_search.search_index_name == 'test-cio-index'
        assert cio_search.semantic_configuration_name == 'test-semantic-config'
        
        # Verify SearchClient is initialized with correct parameters
        mock_search_client.assert_called_once()
        mock_credential.assert_called_once()

    @patch.dict(os.environ, {})
    def test_missing_environment_variables(self):
        """Test that missing environment variables raise ValueError."""
        with pytest.raises(ValueError, match="AI_SEARCH_ENDPOINT and AI_SEARCH_INDEX_NAME environment variables are required"):
            CIOSearchFunctions()

    @patch.dict(os.environ, {
        'AI_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AI_SEARCH_INDEX_NAME': 'test-cio-index',
        'AI_SEARCH_VECTOR_FIELD_NAME': 'testVector'
    })
    @patch('cio_functions.SearchClient')
    @patch('cio_functions.DefaultAzureCredential')
    def test_search_cio_success(self, mock_credential, mock_search_client):
        """Test successful search_cio function call."""
        # Mock search results
        mock_result = {
            'content': 'Investment outlook for Q4 2024 shows positive trends...',
            'title': 'Q4 2024 Investment Outlook',
            'parent_id': '123',
            'chunk_id': '456',
            'testVector': [0.1, 0.2, 0.3]
        }
        
        mock_search_client.return_value.search.return_value = [mock_result]
        
        cio_search = CIOSearchFunctions()
        result = cio_search.search_cio("Q4 investment outlook")
        
        # Verify result is valid JSON
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, list)
        assert len(parsed_result) == 1
        
        # Verify internal fields are removed
        assert 'parent_id' not in parsed_result[0]
        assert 'chunk_id' not in parsed_result[0]
        assert 'testVector' not in parsed_result[0]
        
        # Verify content is preserved
        assert parsed_result[0]['content'] == 'Investment outlook for Q4 2024 shows positive trends...'
        assert parsed_result[0]['title'] == 'Q4 2024 Investment Outlook'

    @patch.dict(os.environ, {
        'AI_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AI_SEARCH_INDEX_NAME': 'test-cio-index'
    })
    @patch('cio_functions.SearchClient')
    @patch('cio_functions.DefaultAzureCredential')
    def test_search_cio_exception_handling(self, mock_credential, mock_search_client):
        """Test search_cio exception handling."""
        # Mock search client to raise exception
        mock_search_client.return_value.search.side_effect = Exception("Search service unavailable")
        
        cio_search = CIOSearchFunctions()
        result = cio_search.search_cio("test query")
        
        # Verify error response
        parsed_result = json.loads(result)
        assert 'error' in parsed_result
        assert 'Search failed' in parsed_result['error']
        assert parsed_result['query'] == 'test query'
        assert parsed_result['results'] == []

    @patch.dict(os.environ, {
        'AI_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AI_SEARCH_INDEX_NAME': 'test-cio-index'
    })
    @patch('cio_functions.SearchClient')
    @patch('cio_functions.DefaultAzureCredential')
    def test_search_cio_wrapper_function(self, mock_credential, mock_search_client):
        """Test the wrapper function for agent execution."""
        # Mock search results
        mock_result = {
            'content': 'Market analysis shows strong performance...',
            'title': 'Market Analysis Report'
        }
        
        mock_search_client.return_value.search.return_value = [mock_result]
        
        # Test wrapper function
        result = search_cio("market analysis")
        
        # Verify result is valid JSON string
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, list)
        assert len(parsed_result) == 1
        assert parsed_result[0]['content'] == 'Market analysis shows strong performance...'

    def test_cio_functions_schema(self):
        """Test that cio_functions has the correct schema."""
        from cio_functions import cio_functions
        
        assert isinstance(cio_functions, list)
        assert len(cio_functions) == 1
        
        function_def = cio_functions[0]
        assert function_def['type'] == 'function'
        assert function_def['function']['name'] == 'search_cio'
        assert 'description' in function_def['function']
        assert 'parameters' in function_def['function']
        
        # Check parameters schema
        params = function_def['function']['parameters']
        assert params['type'] == 'object'
        assert 'query' in params['properties']
        assert params['required'] == ['query']

    @patch.dict(os.environ, {
        'AI_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AI_SEARCH_INDEX_NAME': 'test-cio-index'
    })
    @patch('cio_functions.SearchClient')
    @patch('cio_functions.DefaultAzureCredential')
    def test_search_parameters(self, mock_credential, mock_search_client):
        """Test that search is called with correct parameters."""
        mock_search_client.return_value.search.return_value = []
        
        cio_search = CIOSearchFunctions()
        cio_search.search_cio("investment recommendations")
        
        # Verify search was called with correct parameters
        mock_search_client.return_value.search.assert_called_once()
        call_args = mock_search_client.return_value.search.call_args
        
        assert call_args[1]['search_text'] == "investment recommendations"
        assert call_args[1]['query_type'] == "semantic"
        assert call_args[1]['query_answer'] == "extractive"
        assert call_args[1]['top'] == 3
        assert call_args[1]['query_answer_count'] == 3
        assert call_args[1]['include_total_count'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
