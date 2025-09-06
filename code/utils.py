import os
from dotenv import load_dotenv
from typing import Optional


def get_base_dir() -> Optional[str]:
    """
    Returns the base directory from the BASE_DIR environment variable.
    If BASE_DIR is not set in environment variables, returns None.
    """
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    if not base_dir:
        print("WARNING: BASE_DIR not set in environment variables, returning None for base_dir")
    return base_dir

def get_data_dir(step: int = None) -> Optional[str]:
    """
    Returns the base data directory, or a subdirectory for a given step (1-7).
    Example: get_data_dir(3) -> ../data/step3_app_classification
    If BASE_DIR is not set in environment variables, returns None.
    """
    base_dir = get_base_dir()
    if base_dir:
        base_data_dir = os.path.join(base_dir, 'data')
    else:
        print("WARNING: BASE_DIR not set in environment variables, returning None for base_data_dir")
        base_data_dir = None
    if step is not None:
        if str(step) not in {'1','2','3','4','5','6','7'}:
            raise ValueError("step must be an integer 1 through 7")
        step_map = {
            '1': 'step1_prompts_and_keywords',
            '2': 'step2_webscraped_data',
            '3': 'step3_app_classification',
            '4': 'step4_structured_labels',
            '5': 'step5_clustering_results',
            '6': 'step6_final_dataset',
            '7': 'step7_analysis_results'
        }
        data_dir = os.path.join(base_data_dir, step_map[str(step)])
    else:
        data_dir = base_data_dir
    return data_dir

def get_out_dir() -> Optional[str]:
    """
    Returns the output directory inside the data directory, creating it if necessary.
    """
    data_dir = get_data_dir()
    if data_dir:
        out_dir = os.path.join(data_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    else:
        print("WARNING: BASE_DIR not set in environment variables, cannot create out_dir")
        return None