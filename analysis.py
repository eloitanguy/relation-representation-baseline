import json
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import argparse


def bool_string_to_int(s):
    return int(s == 'True')


def analyse(file_name):
    with open(file_name, 'r') as f:
        cc_res = json.load(f)

    y_pred = [bool_string_to_int(e['is_cause']) for e in cc_res]  # 1 in y_pred means that is_cause=True was predicted
    y_gt = [bool_string_to_int(e['is_cause_gt']) for e in cc_res]

    cm = confusion_matrix(y_gt, y_pred)
    F1 = f1_score(y_gt, y_pred)

    print(cm, "\ncolumn 1: Cause-Effect GT, column 2: Component-Whole GT")
    print("line 1: Cause-Effect prediction, line 2: Component-Whole prediction")
    print("F1 score: {:.2f}% (random: 50%)".format(100*F1))

    scores_cause_correct = []
    scores_cause_incorrect = []
    scores_component_correct = []
    scores_component_incorrect = []
    s_min, s_max = 1e33, -1e33

    for e in cc_res:
        dt, gt = bool_string_to_int(e['is_cause']), bool_string_to_int(e['is_cause_gt'])
        if dt == 0:  # detected Component-Whole
            if gt == 0:  # GT = Component-Whole
                scores_component_correct.append(float(e['component_score']))
            else:
                scores_component_incorrect.append(float(e['component_score']))
        else:  # detected Cause-Effect
            if gt == 0:  # ... incorrectly
                scores_cause_incorrect.append(float(e['cause_score']))
            else:
                scores_cause_correct.append(float(e['cause_score']))
        s_min = min(float(e['component_score']), float(e['cause_score']), s_min)
        s_max = max(float(e['component_score']), float(e['cause_score']), s_max)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Score distributions")
    n_bins = 30

    ax1.hist(scores_cause_correct, bins=n_bins, density=True, alpha=0.5, range=(s_min, s_max),
             label='Correct prediction', color='green')
    ax1.hist(scores_cause_incorrect, bins=n_bins, density=True, alpha=0.5, range=(s_min, s_max),
             label='Incorrect prediction', color='red')

    ax1.title.set_text('Cause detections')
    ax2.hist(scores_component_correct, bins=n_bins, density=True, alpha=0.5, range=(s_min, s_max),
             label='Correct prediction', color='green')
    ax2.hist(scores_component_incorrect, bins=n_bins, density=True, alpha=0.5, range=(s_min, s_max),
             label='Incorrect prediction', color='red')
    ax2.title.set_text('Component detections')

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '--f', required=True, help='File of scores to analyse')
    args = parser.parse_args()

    file_name = args.file
    analyse(file_name)
