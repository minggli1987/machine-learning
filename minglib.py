import statsmodels.api as sm

__author__ = 'Ming Li @ London, UK'


def forward_select(data, target, alpha=0.05, display=True):

    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response w

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    remaining = set([i for i in data.columns if i not in target.columns])
    selected_var = []
    current_score, best_new_score = 0.0, 0.0

    print('beginning forward stepwise variable selection...\n')
    while remaining and current_score == best_new_score:

        scores_with_candidates = []  # containing variables

        for candidate in remaining:

            X = sm.add_constant(data[selected_var + [candidate]]) # inherit variables from last step and try new ones
            reg = sm.OLS(target, X).fit()
            score, p = reg.rsquared, reg.pvalues[-1]  # r2 (changeable) and two-tailed p value of the candidate
            scores_with_candidates.append((score, p, candidate))

        scores_with_candidates.sort(reverse=False)  # order variables by score in ascending
        disqualified_candidates = [i for i in scores_with_candidates if ~(i[1] < alpha)]
        scores_with_candidates = [i for i in scores_with_candidates if i[1] < alpha]
        try:
            best_new_score, best_candidate_p, best_candidate = scores_with_candidates.pop()
        except IndexError:  # when no candidate passes significance test at critical value
            disqualified_score, disqualified_p, disqualified_candidate = disqualified_candidates.pop(0)
            remaining.remove(disqualified_candidate)  # remove the worst disqualified candidate
#            print(disqualified_score, disqualified_p, disqualified_candidate)
            continue  # continuing the while loop
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            current_score = best_new_score
            selected_var.append(best_candidate)
            if display:
                print(selected_var)

    model = sm.OLS(target, sm.add_constant(data[selected_var])).fit()

    print('forward stepwise selection completed...\n')

    return model, selected_var


def backward_select(data, target, alpha=0.05, display=True):
    """Linear model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response w

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    selected_var = list(set([i for i in data.columns if i not in target.columns]))

    if display:
        print('beginning backward stepwise variable selection...\n')

    while selected_var:
        if display:
            print(selected_var)

        scores_with_candidates = []  # containing variables

        X = sm.add_constant(data[selected_var])  # inherit variables from last step and try new ones
        reg = sm.OLS(target, X).fit()
        score = reg.rsquared  # r2 (changeable) and two-tailed p value of the candidate
        p = reg.pvalues[1:]  # first p value belongs to constant
        for i, var in enumerate(selected_var):
            scores_with_candidates.append((p[i], score,  var))
        scores_with_candidates.sort(reverse=False)  # order variables by p value in ascending order
        disqualified_candidates = [i for i in scores_with_candidates if ~(i[0] < alpha)]
        try:
            scores_with_candidates.remove(disqualified_candidates.pop())
            selected_var = [i[2] for i in scores_with_candidates]

        except IndexError:  # when no candidate fails the significance test at critical value
            selected_var = [i[2] for i in scores_with_candidates]
            break

    model = sm.OLS(target, sm.add_constant(data[selected_var])).fit()

    print('\nbackward stepwise selection completed...\n')

    return model, selected_var
