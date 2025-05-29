import json
import time
from typing import List, Dict
from src.llm import OpenAIClient
import os
import pandas as pd

client = OpenAIClient()


feature_extraction_prompt_template = """
You are given a causal inference research paper in the domain of urban economics. Extract the following information in a structured and concise manner. Use standardized academic language and ensure consistency. If a category is not mentioned in the paper, leave it empty.

- Main_Topic: Identify the single most prominent research topic addressed in the paper. Use a standardized academic descriptor such as housing, transportation, inequality, public health, labor markets, crime, urban environment, etc.

- Main_Topic_Description: Provide a short, specific description that clarifies the focus within the main topic. For example, "housing affordability in gentrifying neighborhoods", "public transit access and labor market outcomes", or "air pollution and childhood health outcomes".

- Causal_Inference_Methods:
  - Method_Category: Identify the **primary causal inference category** used in the study. Choose only one from the following five strictly defined categories:
    - "Experimental"
    - "Quasi-Experimental"
    - "Observational"
    - "Structural"
    - "Advanced Statistical & ML"
    
    If the paper uses a quasi-experimental method (e.g., Difference-in-Differences) as the main identification strategy but includes supporting techniques like Propensity Score Matching (PSM), then the **Method_Category** should still be "Quasi-Experimental", while PSM should be listed under Method_Subtype.

  - Method_Subtype: List up to 3 specific techniques used in the paper, selected from the predefined subtypes below:

    - Experimental:
      - Randomized Controlled Trials (RCT)

    - Quasi-Experimental:
      - Difference-in-Differences (DiD)
      - Regression Discontinuity Design (RDD)
      - Synthetic Control Method (SCM)
      - Interrupted Time Series (ITS)
      - Instrumental Variables (IV)

    - Observational:
      - Propensity Score Matching
      - Fixed Effects Panel Regression
      - Dynamic Panel Models (GMM/Arellano–Bond)
      - Spatial Econometric Models (with valid instruments or exogenous shocks)

    - Structural:
      - Structural Equation Modeling (SEM) with Causal Identification
      - Agent-Based Modeling (ABM) calibrated to interventions or natural experiments

    - Advanced Statistical & ML:
      - Causal Bayesian Networks
      - Causal Forests

- Regions:
  - Country: Up to 3 countries mentioned in the study.
  - Continent: One of the following: Africa, North America, South America, Europe, Asia, Oceania, Global. Use "Global" only if the study spans multiple continents.

- Variables:
  - Independent_Variables: Core independent variable(s) the paper analyzes to estimate causal effects. Usually only one unless multiple treatments are central.
  - Dependent_Variable: The primary outcome variable of interest in the analysis.

- Data_Used: Briefly describe the main dataset(s) used in the study, such as census data, mobile phone data, administrative data, transportation logs, etc.

- Open_Source:
  - Data: "Yes" if data is available for public use; otherwise "No".
  - Code: "Yes" if code is available for public use; otherwise "No".

Output Format: Return only a JSON object with the exact structure below. Do not include any explanations, markdown, or additional text.

{
  "Main_Topic": "",
  "Main_Topic_Description": "",
  "Causal_Inference_Methods": {
    "Method_Category": "",
    "Method_Subtype": []
  },
  "Regions": {
    "Country": [],
    "Continent": []
  },
  "Variables": {
    "Independent_Variables": [],
    "Dependent_Variable": ""
  },
  "Data_Used": "",
  "Open_Source": {
    "Data": "Yes" or "No",
    "Code": "Yes" or "No"
  }
}
"""


def extract_features(file_path_list: List[str], prompt_template: str = feature_extraction_prompt_template) -> List[Dict]:
    import concurrent.futures
    from tqdm import tqdm
    
    features = []
    
    def process_file(file_path):
        max_retries = 3
        while max_retries > 0:
            try:
                response = client.generate_response_with_file(
                    prompt=prompt_template,
                    file_path=file_path,
                    model="gpt-4o",
                    force_json=True
                )
                
                response_json = json.loads(response)
                response_json["file_path"] = file_path
                time.sleep(15)  # Avoid rate limiting
                return response_json
            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)} -> Retrying")
                max_retries -= 1
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in file_path_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(file_path_list), desc="Extracting features"):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    features.append(result)
            except Exception as e:
                print(f"Exception for {file_path}: {str(e)}")
    
    # Save intermediate results
    if features:
        with open("data/features_intermediate.json", "w") as f:
            json.dump(features, f)
    
    return features



def flatten_features(features: List[Dict], metadata_df=None):
    """
    Flatten nested dictionaries by combining keys with '+' delimiter.
    Also convert Method_Category from string to list and add title from metadata.
    
    Args:
        features: List of feature dictionaries
        metadata_df: DataFrame containing metadata with 'filename' and 'title' columns
    
    Returns:
        List of flattened dictionaries
    """
    flattened_features = []
    
    for feature in features:
        flat_dict = {}
        
        # Extract file_path for title lookup
        file_path = feature.get("file_path", "")
        
        # Add file_path directly
        flat_dict["file_path"] = file_path
        
        # Add title by looking up file_path in metadata if available
        if metadata_df is not None:
            filename = os.path.basename(file_path)
            matching_rows = metadata_df[metadata_df['filename'].str.contains(filename, regex=False)]
            if not matching_rows.empty:
                flat_dict["title"] = matching_rows.iloc[0]['title']
            else:
                flat_dict["title"] = "Unknown Title"
        else:
            flat_dict["title"] = "Unknown Title"
        
        # Process all keys, flattening nested dictionaries
        for key, value in feature.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    # Special handling for Method_Category - convert to list
                    if sub_key in ["Method_Category", "Dependent_Variable"] and isinstance(sub_value, str):
                        if sub_value:  # Only if not empty
                            flat_dict[f"{key}+{sub_key}"] = [sub_value]
                        else:
                            flat_dict[f"{key}+{sub_key}"] = []
                    else:
                        flat_dict[f"{key}+{sub_key}"] = sub_value
            elif key != "file_path":  # Skip file_path as we've already added it
                if key == "Main_Topic":
                    flat_dict[key] = [value]
                else:
                    flat_dict[key] = value
                
        flattened_features.append(flat_dict)
    
    return flattened_features



import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

item_name_refinement_prompt_template = """You are provided with a list of items. Refine the list by grouping item names that refer to the same concept together and assigning the same new name to each group. For geographic locations, group by the city or region name.
One group can contain one or more items. Try to be concise for the new name.

Item names: {item_names}

Provide strictly JSON output as follows:
{{
  "New name for the group": ["item1", "item2", "item3"]
}}
"""

item_name_refinement_method_prompt_template = """You are provided with a list of causal inference methods. Refine the list by grouping methods that refer to the same methodology together and assigning the same new name to each group. 
The granularity should be at the level of specific methods such as "Difference-in-Differences" or "Instrumental Variables" rather than general methods such as "Quasi-experimental design" or "Econometric modeling".
One group can contain one or more items.

Item names: {item_names}

Provide strictly JSON output as follows:
{{
  "New name for the group": ["item1", "item2", "item3"]
}}
"""

# Embedding and clustering
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_items(items: List[str], threshold: float = 0.8):
    embeddings = model_embedding.encode(items)
    clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=1-threshold)
    clustering_model.fit(embeddings)

    clusters = {}
    for item, label in zip(items, clustering_model.labels_):
        clusters.setdefault(label, []).append(item)

    return list(clusters.values())

# Refinement function with embedding-based clustering and iterative redundancy removal
def refine_features(features: List[Dict]) -> List[Dict]:
    # for field in field_name_dict.keys():
    for field in ['Main_Topic', 'Causal_Inference_Methods+Method_Category', 'Causal_Inference_Methods+Method_Subtype']:
        
        all_names = set(item for feature in features for item in feature[field])
        print(f"Number of items in {field}: {len(all_names)}")
        # print(all_names)

        clusters = cluster_items(list(all_names))
        # update the field mapping with the first item in each cluster
        field_mapping = {item: cluster[0] for cluster in clusters for item in cluster}
        clusters = [[cluster[0]] for cluster in clusters]
        
        previously_refined_names = []
        items_to_refine = []

        cluster_size_list = [len(cluster) for cluster in clusters]
        idx_groups = []
        # make sure each group has sum of more than 30
        s = 0
        idx_group = []
        batch_size = 50
        for idx, cluster_size in enumerate(cluster_size_list):
            s += cluster_size
            idx_group.append(idx)
            if s > batch_size:
                idx_group.append(idx)
                s = 0
                idx_groups.append(idx_group)
                idx_group = []
        if idx_group:
            idx_groups.append(idx_group)

        for idx_group in idx_groups:
            items_group = [item for idx in idx_group for item in clusters[idx]]
            print("Items group:", items_group)
            # Include previously refined names to ensure redundancy removal
            items_to_refine = list(set(previously_refined_names) | set(items_group))
            
            if field == 'Causal_Inference_Methods+Method_Subtype':
                prompt = item_name_refinement_method_prompt_template.format(item_names=items_to_refine)
            else:
                prompt = item_name_refinement_prompt_template.format(item_names=items_to_refine)
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    # print(prompt)
                    response = client.generate_response(
                        prompt, 
                        model="gpt-4.5-preview-2025-02-27", 
                        temperature=0.0, 
                        force_json=True
                    )
                    refined_mapping = json.loads(response)
                    # print(refined_mapping)

                    # Update mapping
                    for new_name, items in refined_mapping.items():
                        for item in items:
                            field_mapping[item] = new_name

                    previously_refined_names.extend(refined_mapping.keys())

                    break  # Success
                except Exception as e:
                    print(f"Error refining cluster {idx+1} for {field}, attempt {attempt+1}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to refine cluster after {max_retries} attempts")
        

        # Apply the mapping
        if field_mapping:
            print(f"Applying mapping for {field} with {len(field_mapping)} items")
            for feature in features:
                updated_items = []
                for item in feature[field]:
                    original_item = item
                    visited = set([item])  # Track visited items to detect cycles
                    
                    while field_mapping.get(item, None) and field_mapping[item] != item:
                        item = field_mapping[item]
                        if item in visited:  # Detect cycle
                            print(f"Warning: Circular reference detected starting with {original_item}")
                            break
                        visited.add(item)
                        
                    updated_items.append(item)
                feature[field] = updated_items
        else:
            print(f"Warning: No mapping created for {field}")
        print("Number of unique items: ", len(set([item for feature in features for item in feature[field]])))
    return features


def add_other_field(features: List[Dict]):
    # remove feature with empty field in causal inference methods
    features = [feature for feature in features if feature["Causal_Inference_Methods+Method_Subtype"] != [] or feature["Causal_Inference_Methods+Method_Category"] != []]

    for feature in features:
        for k, v in feature.items():
            if isinstance(v, list):
                feature[k] = list(set(v))
    return features