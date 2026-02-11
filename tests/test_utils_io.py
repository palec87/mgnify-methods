import json
import os
import tempfile
import tempfile

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from mgnify_methods.utils import io as io_module


def test_process_analysis_metadata():
    """Test process_analysis_metadata function with real taxonomy data files."""
    import os
    
    # Load actual taxonomy abundance files to get real sample IDs
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sola_file = os.path.join(test_dir, 'SRP237882_taxonomy_abundances_SSU_v5.0.tsv')
    osd_file = os.path.join(test_dir, 'ERP124424_taxonomy_abundances_SSU_v5.0.tsv')
    
    # Read the TSV files to extract sample column names (run IDs)
    sola_df = pd.read_csv(sola_file, sep='\t', nrows=1)
    osd_df = pd.read_csv(osd_file, sep='\t', nrows=1)
    
    # Get sample IDs (all columns except the first which is #SampleID/taxonomy)
    sola_samples = [col for col in sola_df.columns if col != '#SampleID'][:5]  # Take first 5 for testing
    osd_samples = [col for col in osd_df.columns if col != '#SampleID'][:3]    # Take first 3 for testing
    
    # Create mock metadata using real sample IDs from the files
    mock_df1 = pd.DataFrame({
        'relationships.run.data.id': sola_samples,
        'attributes.analysis-summary': [f'summary_{i}' for i in range(len(sola_samples))],
        'relationships.study.data.id': ['MGYS00006680'] * len(sola_samples),
        'relationships.sample.data.id': [f'SAMPLE_{i}' for i in range(len(sola_samples))],
    })
    
    mock_df2 = pd.DataFrame({
        'relationships.run.data.id': osd_samples,
        'attributes.analysis-summary': [f'summary_{i}' for i in range(len(osd_samples))],
        'relationships.study.data.id': ['MGYS00006608'] * len(osd_samples),
        'relationships.sample.data.id': [f'SAMPLE_{i}' for i in range(len(osd_samples))],
    })
    
    # Dictionary of studies (structure: {study_name: [analysisId, path]})
    ds_dict = {
        'Sola': ['MGYS00006680', 'path/to/sola.tsv'],
        'OSD2018': ['MGYS00006608', 'path/to/osd.tsv'],
    }
    
    # Mock the fetch_analysis_metadata function
    with patch('mgnify_methods.utils.io.fetch_analysis_metadata') as mock_fetch:
        # Configure mock to return different dataframes for different analysisIds
        def mock_fetch_side_effect(cache_folder, analysisId):
            if analysisId == 'MGYS00006680':
                return mock_df1.copy()
            elif analysisId == 'MGYS00006608':
                return mock_df2.copy()
            else:
                raise ValueError(f"Unexpected analysisId: {analysisId}")
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        # Call the function
        result = io_module.process_analysis_metadata('/fake/cache', ds_dict)
        
        # Assertions
        # 1. Check that fetch_analysis_metadata was called twice
        assert mock_fetch.call_count == 2
        
        # 2. Check the result shape (should have total rows from both studies)
        expected_total = len(sola_samples) + len(osd_samples)
        assert result.shape[0] == expected_total, f"Expected {expected_total} rows, got {result.shape[0]}"
        
        # 3. Check that study_tag column was added
        assert 'study_tag' in result.columns
        
        # 5. Verify study tags are correct for sample runs
        for sample_id in sola_samples:
            assert result.loc[
                result['relationships.run.data.id'] == sample_id,
                'study_tag'
            ].iloc[0] == 'Sola', f"Sample {sample_id} should have study_tag 'Sola'"
        
        for sample_id in osd_samples:
            assert result.loc[
                result['relationships.run.data.id'] == sample_id,
                'study_tag',
            ].iloc[0] == 'OSD2018', f"Sample {sample_id} should have study_tag 'OSD2018'"
        
        # 6. Check that all expected run IDs are present
        expected_run_ids = sola_samples + osd_samples
        assert sorted(result['relationships.run.data.id'].tolist()) == sorted(expected_run_ids)
        
        # 7. Verify other columns are preserved
        assert 'attributes.analysis-summary' in result.columns
        assert 'relationships.study.data.id' in result.columns
        assert 'relationships.sample.data.id' in result.columns
        assert 'relationships.run.data.id' in result.columns


def test_import_taxonomy_summary():
    """Test import_taxonomy_summary with real data file."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = test_dir
    path = 'SRP237882_taxonomy_abundances_SSU_v5.0.tsv'
    
    # Call the function
    result = io_module.import_taxonomy_summary(data_folder, path)
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == '#SampleID'
    assert 'sk__Archaea' in result.index
    # Check that it has sample columns
    assert result.shape[1] > 0
    # Check that values are numeric
    assert result.iloc[0, 0] >= 0


def test_fetch_analysis_metadata_cached():
    """Test fetch_analysis_metadata when file exists in cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock cached file
        analysisId = 'MGYS00006680'
        cache_file = os.path.join(tmp_dir, f'{analysisId}_analysis_meta.csv')
        
        # Create test data and save to cache
        test_data = pd.DataFrame({
            'relationships.run.data.id': ['RUN001', 'RUN002'],
            'attributes.analysis-summary': ['summary1', 'summary2'],
        })
        test_data.to_csv(cache_file, index=False)
        
        # Call the function
        result = io_module.fetch_analysis_metadata(tmp_dir, analysisId)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 2
        assert 'relationships.run.data.id' in result.columns
        assert list(result['relationships.run.data.id']) == ['RUN001', 'RUN002']


def test_fetch_analysis_metadata_download():
    """Test fetch_analysis_metadata when file doesn't exist (downloads from API)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysisId = 'MGYS00006680'
        
        # Mock the API session
        mock_data = [
            {'relationships': {'run': {'data': {'id': 'RUN001'}}}},
            {'relationships': {'run': {'data': {'id': 'RUN002'}}}},
        ]
        
        with patch('mgnify_methods.utils.io.APISession') as mock_session:
            # Configure the mock
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            mock_session_instance.iterate.return_value = [
                MagicMock(json=item) for item in mock_data
            ]
            
            # Call the function
            result = io_module.fetch_analysis_metadata(tmp_dir, analysisId)
            
            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] == 2
            # Check that the file was saved
            cache_file = os.path.join(tmp_dir, f'{analysisId}_analysis_meta.csv')
            assert os.path.exists(cache_file)


def test_fetch_samples_metadata_cached():
    """Test fetch_samples_metadata when file exists in cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysisId = 'MGYS00006680'
        cache_file = os.path.join(tmp_dir, f'{analysisId}_samples_meta.csv')
        
        # Create test data and save to cache
        test_data = pd.DataFrame({
            'id': ['SAMPLE001', 'SAMPLE002'],
            'sample-name': ['Sample 1', 'Sample 2'],
        })
        test_data.to_csv(cache_file, index=False)
        
        # Call the function
        result = io_module.fetch_samples_metadata(tmp_dir, analysisId)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 2
        assert list(result['id']) == ['SAMPLE001', 'SAMPLE002']


def test_process_samples_metadata():
    """Test process_samples_metadata function."""
    # Create mock data
    mock_df1 = pd.DataFrame({
        'id': ['SAMPLE001', 'SAMPLE002'],
        'sample-name': ['Sample 1', 'Sample 2'],
        'geographic location (latitude)': ['Location 1', 'Location 2'],
        'geographic location (longitude)': ['Location 1', 'Location 2'],
        'environment-biome': ['Biome 1', 'Biome 2'],
        'environment-feature': ['Feature 1', 'Feature 2'],
        'environment-material': ['Material 1', 'Material 2'],
    })
    
    mock_df2 = pd.DataFrame({
        'id': ['SAMPLE003', 'SAMPLE004'],
        'sample-name': ['Sample 3', 'Sample 4'],
        'sample-name': ['Sample 1', 'Sample 2'],
        'geographic location (latitude)': ['Location 1', 'Location 2'],
        'geographic location (longitude)': ['Location 1', 'Location 2'],
        'environment-biome': ['Biome 1', 'Biome 2'],
        # 'environment-feature': ['Feature 1', 'Feature 2'],
        # 'environment-material': ['Material 1', 'Material 2'],
    })
    
    ds_dict = {
        'Sola': ['MGYS00006680', 'path/to/sola.tsv'],
        'OSD2018': ['MGYS00006608', 'path/to/osd.tsv'],
    }
    
    with patch('mgnify_methods.utils.io.fetch_samples_metadata') as mock_fetch:
        def mock_fetch_side_effect(cache_folder, analysisId):
            if analysisId == 'MGYS00006680':
                return mock_df1.copy()
            elif analysisId == 'MGYS00006608':
                return mock_df2.copy()
            else:
                raise ValueError(f"Unexpected analysisId: {analysisId}")
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        # Call the function
        result = io_module.process_samples_metadata('/fake/cache', ds_dict)
        
        # Assertions
        assert mock_fetch.call_count == 2
        assert result.shape[0] == 4
        assert 'study_tag' in result.columns
        assert result[result['sample_id'] == 'SAMPLE001']['study_tag'].iloc[0] == 'Sola'
        assert result[result['sample_id'] == 'SAMPLE004']['study_tag'].iloc[0] == 'OSD2018'


def test_extract_feature_with_duplicate_index():
    factors_df = pd.DataFrame(index=["S1", "S2"])
    samples_meta = pd.DataFrame(
        {
            "season": ["winter", "summer"],
        },
        index=["S1", "S2"],
    )

    samples_meta = pd.concat([samples_meta, samples_meta])
    result = io_module.extract_feature(factors_df, "season", samples_meta=samples_meta)

    assert list(result["season"]) == ["winter", "summer"]


def test_process_analysis_metadata_adds_study_tag(monkeypatch):
    mock_df1 = pd.DataFrame({"relationships.run.data.id": ["RUN1"]})
    mock_df2 = pd.DataFrame({"relationships.run.data.id": ["RUN2"]})

    def mock_fetch(cache_folder, analysisId):
        return mock_df1.copy() if analysisId == "A" else mock_df2.copy()

    monkeypatch.setattr(io_module, "fetch_analysis_metadata", mock_fetch)

    ds_dict = {"StudyA": ["A", "path"], "StudyB": ["B", "path"]}
    result = io_module.process_analysis_metadata("/tmp", ds_dict)

    assert "study_tag" in result.columns
    assert result["study_tag"].tolist() == ["StudyA", "StudyB"]


def test_filter_tax_summary_and_integrity():
    df = pd.DataFrame(
        {
            "RUN1": [1, 1, 2],
            "RUN2": [1, 1, 3],
            "EXTRA": [5, 0, 0],
        },
        index=["sk__Archaea", "sk__Eukaryota", "sk__Bacteria"],
    )
    samples_meta = pd.DataFrame(
        {"relationships.run.data.id": ["RUN1", "RUN2"]},
        index=["RUN1", "RUN2"],
    )

    filtered = io_module.filter_tax_summary(df, samples_meta)

    assert list(filtered.columns) == ["RUN1", "RUN2"]
    io_module.assert_taxonomy_integrity(filtered, samples_meta)


def test_filter_number_reads():
    sample_total_dict = {
        "A": {"S1": (100, 0.1), "S2": (10, 0.2)},
        "B": {"S3": (5, 0.3)},
    }
    to_drop = io_module.filter_number_reads(sample_total_dict, 20)

    assert set(to_drop) == {"S2", "S3"}


def test_save_config():
    """Test save_config function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {
            'parameter1': 'value1',
            'parameter2': 42,
            'nested': {'key': 'value'},
        }
        
        # Call the function
        result_path = io_module.save_config(config, tmp_dir, 'test_config.json')
        
        # Assertions
        assert os.path.exists(result_path)
        assert result_path == os.path.join(tmp_dir, 'test_config.json')
        
        # Read back and verify
        with open(result_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['parameter1'] == 'value1'
        assert loaded_config['parameter2'] == 42
        assert loaded_config['nested']['key'] == 'value'


def test_extract_first_date():
    """Test extract_first_date function."""
    # Test valid date
    assert io_module.extract_first_date('2024-01-15') == '2024-01-15'
    assert io_module.extract_first_date('Sample collected on 2024-01-15 at site A') == '2024-01-15'
    assert io_module.extract_first_date('2024-01-15/2024-01-20') == '2024-01-15'
    
    # Test invalid/missing date
    assert io_module.extract_first_date('no date here') is None
    assert io_module.extract_first_date(None) is None
    assert io_module.extract_first_date(pd.NA) is None
    assert io_module.extract_first_date('') is None


def test_process_collection_date():
    """Test process_collection_date function."""
    # Create test data
    test_data = pd.DataFrame({
        'collection_date': [
            '2024-01-15',
            '2024-02-20',
            '2024-03-10',
            'invalid_date',
            None,
        ],
        'other_column': ['A', 'B', 'C', 'D', 'E'],
    })
    
    # Call the function
    result, new_columns = io_module.process_collection_date(test_data)
    
    # Assertions
    assert 'year' in result.columns
    assert 'month' in result.columns
    assert 'month name' in result.columns
    assert 'day' in result.columns
    
    # Check that invalid dates were dropped
    assert result.shape[0] == 3  # Only 3 valid dates
    
    # Check extracted values
    assert result['year'].iloc[0] == 2024
    assert result['month'].iloc[0] == 1
    assert result['month name'].iloc[0] == 'Jan'
    assert result['day'].iloc[0] == 15
    
    assert result['month'].iloc[1] == 2
    assert result['month name'].iloc[1] == 'Feb'
    
    # Check new_columns list
    assert 'year' in new_columns
    assert 'month' in new_columns
    assert 'month name' in new_columns
    assert 'day' in new_columns
