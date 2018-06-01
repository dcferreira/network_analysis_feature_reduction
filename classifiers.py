from IPython.core.display import display, HTML
import sklearn.metrics
from tabulate import tabulate


class ClassifierMetrics(object):
    def __init__(self, x_train, y_train, cats_train,
                 x_test, y_test, cats_test):
        self.x_train = x_train
        self.y_train = y_train
        self.cats_train = cats_train
        self.x_test = x_test
        self.y_test = y_test
        self.cats_test = cats_test

    def test_classifier(classifier):
        out = {'metric': [], 'original': [], 'reduced': []}
        
        # original
        clf = classifier().fit(self.x_train, self.y_train)
        dt_preds_u = clf.predict(self.x_test)
        orig_scores = get_metrics(self.y_test, dt_preds_u)
    
        # reduced
        clf_reduced = classifier().fit(x_enc_train, self.y_train)
        dtr_preds_u = clf_reduced.predict(x_enc_test)
        reduced_scores = get_metrics(self.y_test, dtr_preds_u)
        
        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['reduced'].append(reduced_scores[k])
        display(HTML(tabulate(out, headers='keys', tablefmt='html')))
        
        out = {'metric': [], 'original': [], 'reduced': []}
        # original with cats
        clf_cats = classifier().fit(self.x_train, self.cats_train)
        dt_preds_u_cats = clf_cats.predict(self.x_test)
        orig_scores_cats = get_metrics_cats(self.cats_test, dt_preds_u_cats)
        
        # reduced with cats
        clf_reduced_cats = classifier().fit(x_enc_train, self.cats_train)
        dtr_preds_u_cats = clf_reduced_cats.predict(x_enc_test)
        reduced_scores_cats = get_metrics_cats(self.cats_test, dtr_preds_u_cats)
        
        for k in sorted(orig_scores_cats):
            out['metric'].append(k)
            out['original'].append(orig_scores_cats[k])
            out['reduced'].append(reduced_scores_cats[k])
        print('with 9 attack types')
        display(HTML(tabulate(out, headers='keys', tablefmt='html')))


def get_metrics(y_true, y_pred):
    return {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'precision': sklearn.metrics.precision_score(y_true, y_pred),
        'recall': sklearn.metrics.recall_score(y_true, y_pred),
        'f1': sklearn.metrics.f1_score(y_true, y_pred)
    }

def get_metrics_cats(y_true, y_pred):
    out = {'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred)}
#     out.update({'f1_%d' % i : v for i, v in enumerate(
#         sklearn.metrics.f1_score(y_true, y_pred, average=None)
#     )})
#     out.update({'precision_%d' % i : v for i, v in enumerate(
#         sklearn.metrics.precision_score(y_true, y_pred, average=None)
#     )})
#     out.update({'recall_%d' % i : v for i, v in enumerate(
#         sklearn.metrics.recall_score(y_true, y_pred, average=None)
#     )})
    out['f1_macro'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    out['f1_micro'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    out['precision_macro'] = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    out['precision_micro'] = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    out['recall_macro'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    out['recall_micro'] = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    return out

