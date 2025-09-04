import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI


# ------------------- CONFIG -------------------
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 20

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = script_dir + "/../../data/step3_llm_annotations/"
#INPUT_FILE = DATA_DIR + "validation_data.tsv"
INPUT_FILE = DATA_DIR + "cleaned_labeled_data.tsv" 
OUTPUT_FILE = DATA_DIR + f"structured_data_{timestamp}.tsv"
OUTPUT_FILE_FOR_SHEETS = DATA_DIR + f"structured_data_for_sheets_{timestamp}.tsv"

if __name__ == "__main__":

    # ------------------- SETUP -------------------
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Classification instructions (included once per batch)
    CLASSIFICATION_RULES = """You are an expert in mental health technology research.
    You will be given multiple mental health app descriptions in one request.
    Your task is to classify each app independently into three fields:

    - Features — choose one or more from:
    Talk Therapy / Coaching - Clinician
    Talk Therapy / Coaching - Peer
    Talk Therapy / Coaching - AI/Digital
    Educational Content
    Games
    Meditation
    Journaling
    Soothing Audio
    Affirmations/Inspiration
    Assessment
    Mood Tracking
    Vocalization/Breathwork
    Other
    Unclear

    - Target Demographic — choose one or more from:
    General
    Children
    Youth
    Seniors
    LGBTQ+
    Ethnic Minority
    Neurodivergent
    Couples
    Religious
    Women
    Men
    Pregnant Women / Mothers
    Parents
    Veterans

    - Indication — choose one or more from:
    General
    Anxiety
    Depression & Mood Disorders
    ADHD / ADD
    Sleep
    Substance Use
    Trauma / PTSD
    Eating Disorders
    Bipolar Disorder
    OCD
    Personality Disorders
    Phobias
    Self-Harm
    Schizophrenia
    Relationship Difficulties
    Grief


    RULES:
    1. Select only from the provided lists — do not invent labels.
    2. Multiple labels are allowed for each field.
    3. If no label fits:
    - Use "Other" for Features.
    - Use "General" for Target Demographic and Indication.
    4. Classify each app independently, without referencing other examples.
    5. Output must be valid JSON.
    6. Return an array where each object contains:
    - id (matching the input ID)
    - features (list of strings)
    - target_demographic (list of strings)
    - indication (list of strings)
    """

    # ------------------- LOAD and preprocess DATA -------------------
    if INPUT_FILE == DATA_DIR + "validation_data.tsv":

        df = pd.read_csv(INPUT_FILE, sep='\t')

        # subset to those labeled as mental health
        data = df[df.label == 'Mental Health']

        # add ID column (or make it lowercase)
        data['id'] = range(1, len(data) + 1)

        # reset index
        data.reset_index(drop=True, inplace=True)
    elif INPUT_FILE == DATA_DIR + "cleaned_labeled_data.tsv" :

        df = pd.read_csv(INPUT_FILE, sep='\t')
        df.rename(columns={'Name': 'title', 'Description': 'description', 'ID': 'id', 'Cleaned Prediction': 'label', 'Features': 'manual_features', 'Target Demographic(s)': 'manual_demographics', 'Indications': 'manual_indications', 'Mental Health': 'manual_label'}, inplace=True)

        data = df[df.label == 'Mental Health']

        print(f'Subsetted data to {len(data)} rows labeled as Mental Health.')

        data.reset_index(drop=True, inplace=True)

    # ------------------- PROCESS IN BATCHES -------------------
    results = []
    batch_count = 0
    for start in tqdm(range(0, len(data), BATCH_SIZE)):
        batch_count += 1
        print(f'Batch {batch_count}: categorizing apps from {start} to {start+BATCH_SIZE}...')

        batch = data.iloc[start:start+BATCH_SIZE]

        # Create the batch input text
        batch_text = "Classify the following apps:\n\n"
        for _, row in batch.iterrows():
            batch_text += f"ID: {row['id']}\nName: {row['title']}\nDescription: {row['description']}\n\n"

        # API call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": CLASSIFICATION_RULES},
                {"role": "user", "content": batch_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        # Parse JSON safely
        try:
            batch_results = json.loads(response.choices[0].message.content)['apps']
            results.extend(batch_results)
        except Exception as e:
            print(f"Error parsing batch starting at {start}: {e}")

    print('Structured Annotation Complete!')

    # Merge results back into DataFrame
    results_df = pd.DataFrame(results)
    df_final = data.merge(results_df, on="id", how="left")
    df_final.drop(columns=['Unnamed: 20'], inplace=True)

    # Save version for Python analysis
    df_final.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"Saved to {OUTPUT_FILE}.")

    # Merge strings for Sheets version
    df_final['features'] = df_final['features'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df_final['target_demographic'] = df_final['target_demographic'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df_final['indication'] = df_final['indication'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Save version for Google sheets
    df_final.to_csv(OUTPUT_FILE_FOR_SHEETS, sep='\t', index=False)
    print(f"Saved to {OUTPUT_FILE_FOR_SHEETS}.")

