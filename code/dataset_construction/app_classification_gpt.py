import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from openai import OpenAI
import code.utils as utils

gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-4.1-mini"
DEBUG = True

if DEBUG:
    print("WARNING: DEBUG mode is ON. Only a subset of data will be processed.")


def get_first_k_lines(text, k):
    lines = text.split("\n")
    return " ".join(lines[:k])

def construct_prompt(data_point):
    title = data_point["title"]
    description = get_first_k_lines(data_point["description"], k=4)
    preamble = "Classify the following app as 'Mental Health', 'Not Mental Health', or 'Unclear/Borderline' based on its title and the first few lines of its description. Apps with journals or mood trackers can be considered Mental Health. A sleep support app that just focuses on white noise would not be considered Mental Health, but if it includes guided meditations or mantras, it would. Just output the classification without any further discussion."
    prompt = f"""{preamble}

    Title: {title}
    Description: {description}
    """
    return prompt


def prompt_model(prompt, model, verbose=True):
    if verbose:
        print("PROMPT:\n", prompt)
    response = gpt_client.responses.create(model=model, input=prompt)
    if verbose:
        print("MODEL:\n", response.output_text)
    return response.output_text


def get_data_point(data, index):
    return dict(data.iloc[index])


def generate_metrics_and_predictions(data, verbose=True, debug=False):
    predictions = pd.DataFrame(columns=["prediction", "label"])
    
    if debug:
        print('Debug mode: sampling 10 data points')
        data = data.sample(10, random_state=42).reset_index(drop=True)
    
    for i in range(len(data)):
        if verbose:
            print("DATA POINT ", i)
        data_point = get_data_point(data, i)
        label = data_point["label"]
        prompt = construct_prompt(data_point)
        output = prompt_model(prompt, model, verbose=False).strip()
        if verbose:
            print("OUTPUT:", output)
        if verbose:
            print("TRUE LABEL:", label)
        predictions.loc[i] = [output, label]
        if verbose:
            print("\n\n")
    predictions["correct"] = predictions["prediction"] == predictions["label"]
    accuracy = predictions[predictions.label != "Unclear/Borderline"].correct.mean()
    num_uncertain = predictions[predictions.label == "Unclear/Borderline"].shape[0]
    percent_uncertain = 100.0 * num_uncertain / predictions.shape[0]
    if verbose:
        print("ACCURACY:", accuracy)
    if verbose:
        print("PERCENT UNCERTAIN:", percent_uncertain)
    metrics = {"accuracy": accuracy, "percent_uncertain": percent_uncertain}
    return metrics, predictions


def label_data(data):
    if "prediction" not in data.columns:
        data["prediction"] = None

    data = data.reset_index(drop=True)

    for i in tqdm(range(len(data))):
        data_point = get_data_point(data, i)
        prompt = construct_prompt(data_point)
        output = prompt_model(prompt, model, verbose=False).strip()
        data.loc[i, "prediction"] = output

    return data


if __name__ == "__main__":

    data_dir = utils.get_data_dir(step=3)
    valid_file = os.path.join(data_dir, "validation_data.tsv")

    val_data = pd.read_csv(valid_file, sep="\t")
    val_metrics, val_predictions = generate_metrics_and_predictions(
        val_data, verbose=True, debug=DEBUG)
    print(val_metrics)

    input_file = os.path.join(data_dir, "app_data_plus_manual_labels.tsv")
    data = pd.read_csv(input_file, sep="\t")
    data.rename(
        columns={
            "Mental Health": "label",
            "Name": "title",
            "Description": "description",
            "Source": "source",
        },
        inplace=True,
    )

    # sample_data = data.sample(10)
    # labeled_sample = label_data(sample_data)
    # columns= ['title', 'description', 'label', 'prediction']
    # labeled_sample[columns]

    if DEBUG:
        data = data.sample(20, random_state=42)

    labeled_data = label_data(data)

    manual_labels = labeled_data[
        labeled_data.label.isin(["Mental Health", "Not Mental Health"])
    ]
    accuracy_on_all_labels = np.mean(manual_labels.label == manual_labels.prediction)
    print("Accuracy on all labels:", accuracy_on_all_labels)

    labeled_data["ID"] = range(1, len(labeled_data) + 1)

    outdir = utils.get_out_dir()
    outfile = os.path.join(outdir, "app_data_plus_ai_labels.tsv")
    labeled_data.to_csv(outfile, sep="\t", header=True, index=False)
