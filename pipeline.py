import os
import json
import pandas as pd
from src.llm import OpenAIClient
from src.filter import filter_papers
from src.data_processing import extract_features, refine_features, add_other_field, flatten_features
from src.sankey import clean_sankey_data

PAPER_DIR = "data/Cities"
META_DATA_DIR = "data/metadata/cities.xlsx"

def main():
    meta_data = pd.read_excel(META_DATA_DIR)
    meta_data['filename'] = meta_data['doi'].apply(lambda x: os.path.join(PAPER_DIR, x.split('/')[-1] + '.pdf'))
    
    
    # step 1: get all causal research papers
    paper_file_list = [os.path.join(PAPER_DIR, file) for file in os.listdir(PAPER_DIR) if file.endswith(".pdf")]
    # match the filename with the paper_file_list and get paper file list with their titles
    meta_data = meta_data[meta_data['filename'].isin(paper_file_list)]
    paper_data = meta_data[['filename', 'title']].to_dict(orient='records')
    causal_research_list = filter_papers(paper_data)
    with open("data/causal_research_list.json", "w") as f:
        json.dump(causal_research_list, f)

    # step 2: extract features from the causal research papers
    file_path_list = [os.path.join("data/causal_research_list", file) for file in os.listdir("data/causal_research_list") if file.endswith(".pdf")]
    features = extract_features(file_path_list)
    with open("data/features.json", "w") as f:
        json.dump(features, f)
    
    features = json.load(open("data/features.json"))
    flattened_features = flatten_features(features, meta_data)
    features = add_other_field(flattened_features)
    with open("data/flattened_features.json", "w") as f:
        json.dump(flattened_features, f)

    # step 3: refine features by grouping similar features together
    features = json.load(open("data/flattened_features.json"))
    features = refine_features(features)
    with open("data/refined_features.json", "w") as f:
        json.dump(features, f)
    
    
    # step 4: generate Sankey diagram data
    # additional data refinement can be applied manually here before we load the data
    features = json.load(open("data/refined_features.json"))
    sankey_data = clean_sankey_data(features)
    sankey_data.to_csv("data/sankey_data.csv", index=False)

if __name__ == "__main__":
    main()