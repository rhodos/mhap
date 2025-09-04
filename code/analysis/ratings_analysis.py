import pandas as pd
from scipy import stats
import numpy as np
from datetime import datetime
import os

from statsmodels.sandbox.stats.multicomp import multipletests

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir + "/../../data/step5_final_dataset/single_table/"
output_dir = script_dir + "/../../data/step6_analysis_results/"

INPUT_FILE = input_dir + 'mental_health_apps_flattened_cols.tsv'
RESULTS_FILE = output_dir + f'ratings_analysis_t_test_results_{timestamp}.tsv'

if __name__ == "__main__":

    apps = pd.read_csv(INPUT_FILE, sep='\t')
    apps = apps[apps.category != 'Period Trackers']

    feature_cols = ['Affirmations/Inspiration', 'Assessment',
        'Educational Content', 'Games', 'Journaling', 'Meditation',
        'Mood Tracking', 'Other', 'Soothing Audio',
        'Talk Therapy / Coaching - AI/Digital',
        'Talk Therapy / Coaching - Clinician', 'Talk Therapy / Coaching - Peer',
        'Vocalization/Breathwork']

    indication_cols = ['ADHD / ADD', 'Anxiety', 'Bipolar Disorder',
        'Depression & Mood Disorders', 'Eating Disorders', 'General', 'Grief',
        'OCD', 'Personality Disorders', 'Phobias', 'Relationship Difficulties',
        'Schizophrenia', 'Self-Harm', 'Sleep', 'Substance Use / Addiction',
        'Trauma / PTSD']

    demographic_cols = ['Children', 'Couples', 'Ethnic Minority',
        'General_demographic', 'LGBTQ+', 'Men', 'Neurodivergent', 'Parents',
        'Pregnant Women / Mothers', 'Religious', 'Seniors', 'Veterans', 'Women',
        'Youth']

    attribute_cols = {col: 'Feature' for col in feature_cols}
    attribute_cols.update({col: 'Indication' for col in indication_cols})
    attribute_cols.update({col: 'Demographic' for col in demographic_cols})


    print(attribute_cols)

    ttest_results = []

    for col, coltype in attribute_cols.items():
        ratings_with_attribute = apps[apps[col] == True]['rating'].dropna()
        ratings_without_attribute = apps[apps[col] == False]['rating'].dropna()

        if len(ratings_with_attribute) > 1 and len(ratings_without_attribute) > 1: # Ensure there are enough samples for t-test
            ttest_result = stats.ttest_ind(ratings_with_attribute, ratings_without_attribute, equal_var=False) # Welch's t-test
            ttest_results.append({
                'attribute_type': coltype,
                'attribute': col,
                'statistic': ttest_result.statistic,
                'pvalue': ttest_result.pvalue
            })

    categories = apps['category'].unique()

    for category in categories:
        ratings_with_attribute = apps[apps['category'] == category]['rating'].dropna()
        ratings_without_attribute = apps[apps['category'] != category]['rating'].dropna()

        if len(ratings_with_attribute) > 1 and len(ratings_without_attribute) > 1: # Ensure there are enough samples for t-test
            ttest_result = stats.ttest_ind(ratings_with_attribute, ratings_without_attribute, equal_var=False) # Welch's t-test
            ttest_results.append({
                'attribute_type': 'Category',
                'attribute': category,
                'statistic': ttest_result.statistic,
                'pvalue': ttest_result.pvalue
            })

    results = pd.DataFrame(ttest_results)

    # Apply Benjamini-Hochberg correction
    reject, bh_pvalues, _, _ = multipletests(results.pvalue, method='fdr_bh')
    results['corrected_pvalue'] = bh_pvalues

    # Display the results with BH corrected p-values, sorted by corrected p-value
    results = results.sort_values(by='corrected_pvalue')
    results.to_csv(RESULTS_FILE, sep='\t', index=None, float_format='%.15f')

    # significant_results = results[results.corrected_pvalue < 0.1]
    # significant_results.to_csv('results_nominal_significance.tsv', sep='\t', index=None, float_format='%.15f')

    # significant_results = results[results.corrected_pvalue < 0.05]
    # significant_results.to_csv('results_significance.tsv', sep='\t', index=None, float_format='%.15f')


    """## Do apps that target specific indications or demographics do better (or worse) than apps geared toward a general audience?"""

    # apps that do not include the General indication label
    specific_indication = apps[~apps['General']]['rating'].dropna()

    # apps that only include the General indication label
    general_indication = apps[apps.indications == 'General']['rating'].dropna()

    ttest_result = stats.ttest_ind(specific_indication, general_indication, equal_var=False) # Welch's t-test
    print("Specific vs General Indication Ratings T-test Result:")
    print(ttest_result)

    # apps that do not include the General demographic label
    specific_demographic = apps[~apps['General_demographic']]['rating'].dropna()

    # apps that only have the General demographic label
    general_demographic = apps[apps.demographics == 'General']['rating'].dropna()

    ttest_result = stats.ttest_ind(specific_demographic, general_demographic, equal_var=False) # Welch's t-test
    print('Specific vs General Demographic Ratings T-test Result:')
    print(ttest_result)
