from IPython.core.display import display, HTML
import numpy as np
import sklearn.metrics
from sklearn import svm, linear_model, tree
from tabulate import tabulate


classifiers = {'Decision Tree': tree.DecisionTreeClassifier,
               'SVM': svm.LinearSVC,
               'Logistic Regression': linear_model.LogisticRegression}


class ClassifierMetrics(object):
    def __init__(self, data, model):
        """

        Args:
            data (data.Data):
            model:
        """
        self.data = data
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.cats_train = self.data.cats_nr_train
        self.x_test = self.data.x_test
        self.y_test = self.data.y_test
        self.cats_test = self.data.cats_nr_test

        self.model = model
        self.x_enc_train = self.model.get_embeddings(self.x_train)
        self.x_enc_test = self.model.get_embeddings(self.x_test)

    def test_classifier(self, classifier, display_scores=True):
        out = {'metric': [], 'original': [], 'reduced': []}
        
        # original
        clf = classifier().fit(self.x_train, self.y_train)
        dt_preds_u = clf.predict(self.x_test)
        orig_scores = get_metrics(self.y_test, dt_preds_u)
    
        # reduced
        clf_reduced = classifier().fit(self.x_enc_train, self.y_train)
        dtr_preds_u = clf_reduced.predict(self.x_enc_test)
        reduced_scores = get_metrics(self.y_test, dtr_preds_u)
        
        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['reduced'].append(reduced_scores[k])

        out_cat = {'metric': [], 'original': [], 'reduced': []}
        # original with cats
        clf_cats = classifier().fit(self.x_train, self.cats_train)
        dt_preds_u_cats = clf_cats.predict(self.x_test)
        orig_scores_cats = get_metrics_cats(self.cats_test, dt_preds_u_cats)
        
        # reduced with cats
        clf_reduced_cats = classifier().fit(self.x_enc_train, self.cats_train)
        dtr_preds_u_cats = clf_reduced_cats.predict(self.x_enc_test)
        reduced_scores_cats = get_metrics_cats(self.cats_test, dtr_preds_u_cats)
        
        for k in sorted(orig_scores_cats):
            out_cat['metric'].append(k)
            out_cat['original'].append(orig_scores_cats[k])
            out_cat['reduced'].append(reduced_scores_cats[k])

        if display_scores:
            print('binary class:')
            display(HTML(tabulate(out, headers='keys', tablefmt='html')))
            print('with attack categories:')
            display(HTML(tabulate(out_cat, headers='keys', tablefmt='html')))
        else:
            return out, out_cat


class Aggregator(object):
    def __init__(self, model_class, number, *args, **kwargs):
        self.models = [model_class(*args, **kwargs) for _ in range(number)]
        self.metrics = None
        self.scores = None
        self.histories = None

    def train(self, **kwargs):
        self.histories = [model.train(**kwargs) for model in self.models]

    def get_metrics(self, data):
        self.metrics = [ClassifierMetrics(data, model) for model in self.models]
        self.scores = {}
        for cname, c in classifiers.items():
            self.scores[cname] = [met.test_classifier(c, display_scores=False) for met in self.metrics]
        return self.scores

    def apply_op(self, op, display_scores=True):
        out = {}
        for cname, metrics in self.scores.items():
            out[cname] = {}
            out[cname]['binary'] = {}
            out[cname]['binary']['reduced'] = {
                metrics[0][0]['metric'][met_idx]: op([met[0]['reduced'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][0]['reduced']))
            }
            out[cname]['binary']['original'] = {
                metrics[0][0]['metric'][met_idx]: op([met[0]['original'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][0]['original']))
            }

            out[cname]['multi'] = {}
            out[cname]['multi']['reduced'] = {
                metrics[0][1]['metric'][met_idx]: op([met[1]['reduced'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][1]['reduced']))
            }
            out[cname]['multi']['original'] = {
                metrics[0][1]['metric'][met_idx]: op([met[1]['original'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][1]['original']))
            }
        if not display_scores:
            return out
        else:
            for name in out:
                print('metrics for %s' % name)
                tmp = out[name]

                b = tmp['binary']
                b_out = {'metric': [], 'original': [], 'reduced': []}
                for k in sorted(b['reduced']):
                    b_out['metric'].append(k)
                    b_out['original'].append(b['original'][k])
                    b_out['reduced'].append(b['reduced'][k])

                m = tmp['multi']
                m_out = {'metric': [], 'original': [], 'reduced': []}
                for k in sorted(m['reduced']):
                    m_out['metric'].append(k)
                    m_out['original'].append(m['original'][k])
                    m_out['reduced'].append(m['reduced'][k])

                print('binary class:')
                display(HTML(tabulate(b_out, headers='keys', tablefmt='html')))
                print('with attack categories:')
                display(HTML(tabulate(m_out, headers='keys', tablefmt='html')))


    def mean(self, display_scores=True):
        return self.apply_op(np.mean, display_scores=display_scores)

    def std(self, display_scores=True):
        return self.apply_op(np.std, display_scores=display_scores)


def test_model(data, model):
    for name, c in classifiers.items():
        metrics = ClassifierMetrics(data, model)
        print('metrics for %s' % name)
        metrics.test_classifier(c)


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
