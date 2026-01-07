"""
MGnify API interaction module.

Provides functions to retrieve data and metadata from the MGnify API.
"""
import os
import pandas as pd
from tqdm import tqdm
from jsonapi_client import Session as APISession


# ---------------------------
# Download helpers
# ---------------------------
def retrieve_summary_old(studyId: str, matching_string: str = 'Taxonomic assignments SSU') -> None:
    """
    Retrieve summary data for a given analysis ID and save it to a file. Matching strings 
    are substrings of for instance:
    - Phylum level taxonomies SSU
    - Taxonomic assignments SSU
    - Phylum level taxonomies LSU
    - Taxonomic assignments LSU
    - Phylum level taxonomies ITSoneDB
    - Taxonomic assignments ITSoneDB
    - Phylum level taxonomies UNITE
    - Taxonomic assignments UNITE

    Example usage:
    retrieve_summary('MGYS00006680', matching_string='Taxonomic assignments SSU')

    Args:
        studyId (str): The ID of the analysis to retrieve. studyId is the MGnify study ID, used
            also to save the output .tsv file.
        matching_string (str): The string to match in the download description label.
    
    Returns:
        None
    """
    from urllib.request import urlretrieve

    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
        for download in session.iterate(f"studies/{studyId}/downloads"):
            if download.description.label == matching_string:
                print(f"Downloading {download.alias}...")
                urlretrieve(download.links.self.url, f'{studyId}.tsv')


def retrieve_summary(studyId: str, matching_string: str = 'Taxonomic assignments SSU', out_dir: str = '.') -> str:
    """
    Retrieve summary data for a given MGnify study and save it to a TSV file.

    Parameters
    ----------
    studyId : str
        MGnify study ID, e.g., 'MGYS00006680'.
    matching_string : str
        Label to match in download description (e.g., 'Taxonomic assignments SSU').
    out_dir : str
        Output directory to store the TSV file.

    Returns
    -------
    str
        Path to the downloaded TSV file.
    """
    from urllib.request import urlretrieve


    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, f'{studyId}.tsv')

    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
        for download in session.iterate(f"studies/{studyId}/downloads"):
            if download.description.label == matching_string:
                urlretrieve(download.links.self.url, tsv_path)
                return tsv_path

    raise FileNotFoundError(f"No download matched '{matching_string}' for study {studyId}")


# function to get metadata for MGnify studies
def get_mgnify_metadata(study_id):
    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:

        samples = map(lambda r: r.json, session.iterate(f'studies/{study_id}/samples?page_size=1000'))

        sample_list = []
        for sample_json in tqdm(samples):
            # Flatten sample-metadata list into a dictionary
            # 1. Extract sample-metadata (allowing None)
            metadata_fields = {
                item.get("key"): item.get("value", None)
                for item in sample_json["attributes"].get("sample-metadata", [])
            }

            # 2. Extract all other attributes (including None)
            attributes_fields = {
                k: v for k, v in sample_json["attributes"].items()
                if k != "sample-metadata"  # already unpacked separately
            }

            # 3. Merge everything including top-level id
            flat_data = {
                "id": sample_json.get("id"),
                **attributes_fields,
                **metadata_fields
            }

            # 4. Create DataFrame
            df = pd.DataFrame([flat_data])
            sample_list.append(df)

        # Concatenate all DataFrames into one
        df = pd.concat(sample_list, ignore_index=True)
        df['study'] = study_id
    return df