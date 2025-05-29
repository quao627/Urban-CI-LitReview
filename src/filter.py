from src.llm import OpenAIClient
from typing import List
import json
import time
import shutil
import os
client = OpenAIClient()


title_prompt_template = """Based on the title of the research paper, determine if it could potentially be a paper that employs causal inference using econometric methods in urban studies.

Provide your evaluation strictly in the following JSON format:

```json
{
  "causal_inference": true or false,
  "justification": "Short and concise justification about the reason for your evaluation (one sentence)"
}
"""

full_prompt_template = """
Analyze the provided research paper. Determine if it employs causal inference using econometric methods. Specifically check for:

- Explicit mentions of causal relationships (e.g., estimating causal effects, treatment effects, causality).
- Econometric methods (instrumental variables (IV), difference-in-differences (DiD), regression discontinuity design (RDD), synthetic control, fixed effects, propensity score matching, structural modeling).
- Clear empirical identification strategies aiming to distinguish causation from correlation.

Provide your evaluation strictly in the following JSON format:

```json
{
  "causal_inference": true or false,
  "justification": "Short justification citing relevant evidence from the paper"
}
"""

def filter_papers(paper_data: List[dict]) -> List[str]:
    import concurrent.futures
    from tqdm import tqdm

    print(f"Start filtering {len(paper_data)} papers")
    
    causal_research_list = []
    causal_research_dir = "data/causal_research_list_v2"
    if not os.path.exists(causal_research_dir):
        os.makedirs(causal_research_dir)

    def process_paper(title, file_path):
        try:
            print(f"Start filtering: {file_path}")
            # first check the title
            prompt = title_prompt_template
            title_response = client.generate_response(
                prompt=prompt,
                model="gpt-4o-mini",
                force_json=True
            )
            title_is_causal_research = json.loads(title_response)["causal_inference"]
            if not title_is_causal_research:
                return None
            # then check the full paper
            prompt = full_prompt_template
            file_response = client.generate_response_with_file(
                prompt=prompt,
                file_path=file_path,
                model="gpt-4o",
                force_json=True
            )
            is_causal_research = json.loads(file_response)["causal_inference"]
            if is_causal_research:
                print(f"Causal research found: {file_path}")
                return file_path
            time.sleep(30)
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            time.sleep(10)
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all tasks and create a progress bar
        futures = {executor.submit(process_paper, paper_data['title'], paper_data['filename']): paper_data['filename'] for paper_data in paper_data}
        
        # Process results as they complete with progress tracking
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(paper_data), desc="Filtering papers"):
            result = future.result()
            if result:
                # save pdf to data/causal_research_list
                shutil.copy(result, os.path.join(causal_research_dir, os.path.basename(result)))
                causal_research_list.append(result)
    
    return causal_research_list
