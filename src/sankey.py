import pandas as pd
from collections import Counter
from typing import List, Dict


def generate_sankey_data(metadata_list, flow_pairs):

    all_flows = []
    depth_dict = {}

    for metadata in metadata_list:
        for depth_index, (source_field, target_field) in enumerate(flow_pairs):
            sources = metadata.get(source_field, [])
            targets = metadata.get(target_field, [])
            for src in sources:
                for tgt in targets:
                    if src != tgt:
                        all_flows.append((src, tgt, source_field, target_field))
                        depth_dict[(src, tgt)] = depth_index

    # Count unique flows
    flow_counts = Counter(all_flows)

    # Convert to Sankey format
    sankey_data = [
        {"source": source, "target": target, "value": count, "source_field": source_field, "target_field": target_field}
        for (source, target, source_field, target_field), count in flow_counts.items()
    ]

    df = pd.DataFrame(sankey_data)

    return df


def clean_sankey_data(features: List[Dict]):
    flow_pairs = [
        ("Main_Topic", "Regions+Continent"),
        ("Regions+Continent", "Data_Types"),
        ("Data_Types", "Causal_Inference_Methods+Method_Category"),
        ("Causal_Inference_Methods+Method_Category", "Causal_Inference_Methods+Method_Subtype"),
    ]

    for feature in features:
        feature["Data_Types"] = list(feature["Data_Details"].keys())

    sankey_data = generate_sankey_data(sankey_data, flow_pairs)
    sankey_data = sankey_data[sankey_data.isna().sum(axis=1) == 0]
    sankey_data = sankey_data[sankey_data.isnull().sum(axis=1) == 0]
    sankey_data = sankey_data[sankey_data.apply(lambda x: x.astype(str).str.strip() != '').all(axis=1)]
    sankey_data = sankey_data.head(50)
    return sankey_data