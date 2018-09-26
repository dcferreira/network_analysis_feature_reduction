import numpy as np
import pandas as pd
from tabulate import tabulate
from bokeh.io import curdoc
from bokeh.layouts import Column, Row
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, Button, Slider, CategoricalColorMapper,
                          CDSView, GroupFilter, Legend, Div, TapTool, Circle, CircleCross)
from bokeh.palettes import Category10
try:
    from classifiers import VisualClassifier
except ImportError:
    raise ImportError('Error importing! Add the root of the repository to your PYTHONPATH and try again.')


df = pd.read_pickle('../dataframe.pkl')
df_train = np.load('../cats_ae_x_train_scaled.npy')
cats_nr_train = np.load('../cats_nr_train.npy')
categories = ['attack_cat_Analysis', 'attack_cat_Backdoor', 'attack_cat_DoS',
              'attack_cat_Exploits', 'attack_cat_Fuzzers', 'attack_cat_Generic',
              'attack_cat_Reconnaissance', 'attack_cat_Shellcode', 'attack_cat_Worms',
              'label_0']
categories_short = ['Analysis', 'Backdoor', 'DoS',
                    'Exploits', 'Fuzzers', 'Generic',
                    'Reconnaissance', 'Shellcode', 'Worms',
                    'label_0']
headers = ['', 'Normal', 'Attack']


source = ColumnDataSource(df.iloc[:100])
colors = Category10[10]
plot = figure(x_range=(0,1), y_range=(0,1), width=800, height=400,
              tools='hover')
color_mapper = CategoricalColorMapper(palette=colors, factors=categories)
legend_list = []
for i in range(len(categories)):
    cat = categories[i]
    view = CDSView(source=source, filters=[GroupFilter(column_name='cat_str', group=str(i))])
    rend = plot.scatter('x_cats_ae', 'y_cats_ae', color=colors[i], size=15,
                        line_color='black', source=source, view=view)
    rend.selection_glyph = Circle(fill_alpha=1, fill_color=colors[i], line_color='black')
    rend.nonselection_glyph = CircleCross(fill_alpha=0.1, fill_color=colors[i], line_color=colors[i])
    legend_list.append((cat, [rend]))

legend = Legend(items=legend_list, location=(20, 0))
legend.click_policy = 'hide'
plot.add_layout(legend, 'left')


point_info = Div()
point_probabilities = Div()
visual_classifier = VisualClassifier(leafsize=1000)
visual_classifier.fit(df_train, cats_nr_train)


radius_slider = Slider(start=0, end=0.2, value=0.01, step=0.0001, title="Radius for visual", format="0[.]0000")
radius_source = ColumnDataSource({'x': [], 'y': [], 'rad': []})
plot.circle('x', 'y', radius='rad', source=radius_source, line_color='black',
            line_width=2, color=None, line_dash='dashed', line_alpha=0.7)


def get_attributes(attr, old, new):
    # source.data = {x: [] for x in df.columns}
    try:
        idx = source.selected._property_values['indices'][0]
    except KeyError:
        radius_source.data = {'x': [], 'y': [], 'rad': []}
        return
    dpoint = df.iloc[source.data['index'][idx]]

    # draw radius circle
    radius_source.data = {'x': [dpoint.x_cats_ae],
                          'y': [dpoint.y_cats_ae],
                          'rad': [radius_slider.value]}

    # update table with point info
    bin_class = {0: 'Normal', 1: 'Attack'}
    visual_pred = visual_classifier.predict(np.array([
        dpoint.x_cats_ae,
        dpoint.y_cats_ae
    ]), eps=radius_slider.value)
    visual_pred_str = ('✅ ' if visual_pred == dpoint.category else '❌ ') + \
                      categories_short[visual_pred] if visual_pred != -1 else 'Unknown'

    cats_pred = ('✅ ' if dpoint.cats_ae_pred == (dpoint.category != 9) else '❌ ') + bin_class[dpoint.cats_ae_pred]
    orig_pred = ('✅ ' if dpoint.original_pred == (dpoint.category != 9) else '❌ ') + bin_class[dpoint.original_pred]
    point_info.text = '<div class="attrs">' + tabulate([
        ['category', categories_short[dpoint.category]],
        ['cats_ae_pred', cats_pred],
        ['original_pred', orig_pred],
        ['x', '{:.4f}'.format(dpoint.x_cats_ae)],
        ['y', '{:.4f}'.format(dpoint.y_cats_ae)],
        ['visual_pred', visual_pred_str],
         ], tablefmt='html') + '</div>'
    probs = visual_classifier.predict_proba(np.array([
        dpoint.x_cats_ae,
        dpoint.y_cats_ae
    ]), eps=radius_slider.value)
    point_probabilities.text = '<div class="probs">' + tabulate(
        [['{:.4f}'.format(x) for x in probs]],
        headers=categories_short,
        numalign='left',
        tablefmt='html') + '</div>'


radius_slider.on_change('value', get_attributes)

plot.add_tools(TapTool(behavior='select'))
source.on_change('selected', get_attributes)
# plot.add_tools(HoverTool(tooltips="@index", show_arrow=False, point_policy='follow_mouse'))


def animate_update():
    flow_nr = time_slider.value + 1
    if flow_nr >= len(df):
        flow_nr = 0
    time_slider.value = flow_nr

def slider_update(attrname, old, new):
    flow_nr = time_slider.value
    # get data
    source.stream(df.iloc[flow_nr], flows_max_slider.value)

def update_speed(attrname, old, new):
    global callback_id
    curdoc().remove_periodic_callback(callback_id)  # this is not working
    callback_id = curdoc().add_periodic_callback(animate_update, speed_slider.value)


cmatrix_cats_ae = Div()
cmatrix_original = Div()


def update_conf_matrix(mat, pred_name):
    conf_matrix = np.zeros((len(categories), 2))
    data = source.data
    for gold, pred in zip(data['category'], data[pred_name]):
        conf_matrix[gold, pred] += 1
    heads = headers.copy()
    heads[0] = '{} predictions'.format(pred_name)
    mat.text = tabulate(add_column_headers(conf_matrix), headers=heads, tablefmt='html', numalign='center')


def update_cats_ae_mat():
    update_conf_matrix(cmatrix_cats_ae, 'cats_ae_pred')

def update_original_mat():
    update_conf_matrix(cmatrix_original, 'original_pred')

curdoc().add_periodic_callback(update_cats_ae_mat, 200)
curdoc().add_periodic_callback(update_original_mat, 200)


def add_column_headers(mat):
    out = mat.T.tolist()
    out.insert(0, ['<b>%s</b>' % x for x in categories_short])
    return np.array(out).T


time_slider = Slider(start=0, end=len(df), value=0, step=1, title="Flow nr")
time_slider.on_change('value', slider_update)
speed_slider = Slider(start=10, end=500, value=10, title="New flow every (ms)")
speed_slider.on_change('value', update_speed)
flows_max_slider = Slider(start=10, end=10000, value=200, step=10, title="Number of flows to keep")

callback_id = None


def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, speed_slider.value)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)


button = Button(label='► Play')#, width=100)
button.on_click(animate)


layout = Column(children=[
    Row(children=[
        # WidgetBox(button, width=100),
        Column(children=[button, speed_slider]),
        Column(children=[time_slider, flows_max_slider]),
        radius_slider,
    ]),
    Row(children=[
        plot,
        Column(children=[point_info, point_probabilities]),
    ]),
    Row(children=[
        cmatrix_cats_ae,
        cmatrix_original,
    ])
])

curdoc().add_root(layout)
