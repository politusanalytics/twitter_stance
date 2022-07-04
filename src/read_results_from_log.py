import sys
import re
import numpy as np
import pandas as pd

input_log_filename = sys.argv[1]

if __name__ == "__main__":
    with open(input_log_filename, "r") as log_file:
        all_conf_results = {}
        all_metrics = []

        curr_conf_results = []
        curr_seed_results = {}
        results_lines = False
        conf_name = ""
        for line in log_file:
            if "======" in line: # start of current configuration's results
                # write previous configuration's results
                if len(curr_conf_results) > 0:
                    all_conf_results[conf_name] = curr_conf_results

                conf_name = re.search("====== ([^=]+) ======", line).group(1)
                curr_conf_results = []

            elif "***** TEST RESULTS *****" in line:
                results_lines = True

            elif "TEST SCORE: " in line:
                # check if metrics match
                if len(all_conf_results) > 0 or len(curr_conf_results) > 0:
                    assert(sorted(list(curr_seed_results.keys())) == all_metrics)
                else:
                    all_metrics = sorted(list(curr_seed_results.keys()))

                curr_conf_results.append(curr_seed_results)
                curr_seed_results = {}
                results_lines = False

            elif results_lines:
                match = re.search(r"(\w+) = (0\.\d+)$", line)
                result_name = match.group(1)
                curr_result = float(match.group(2)) * 100
                curr_seed_results[result_name] = curr_result

    if len(curr_conf_results) > 0:
        all_conf_results[conf_name] = curr_conf_results

    results_df = pd.DataFrame(columns=all_metrics)
    for curr_conf_results in all_conf_results.values():
        metrics_results = {}
        for metric in all_metrics:
            curr_metric_results = np.array([seed_results[metric] for seed_results in curr_conf_results])
            curr_mean = round(np.mean(curr_metric_results), 4)
            curr_std = round(np.std(curr_metric_results), 4)
            metrics_results[metric] = str(curr_mean) + " +- " + str(curr_std)

        results_df = results_df.append(metrics_results, ignore_index=True)

    results_df.index = list(all_conf_results.keys())
    results_df.to_html(input_log_filename + ".html")
