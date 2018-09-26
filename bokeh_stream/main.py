import numpy as np
import pandas as pd
from tabulate import tabulate
from bokeh.io import curdoc
from bokeh.layouts import Column, Row
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, Slider, CategoricalColorMapper,
                          CDSView, GroupFilter, Legend, Div, TapTool, Circle, CircleCross)
from bokeh.palettes import Category10
from attributes import get_attributes, train_visual_classifier, categories, categories_short
from widgets import get_widgets


# get these files by running `generate_pickles.py` (need to add root to PYTHONPATH)
df = pd.read_pickle('dataframe.pkl')
df_train = np.load('cats_ae_x_train_scaled.npy')
cats_nr_train = np.load('cats_nr_train.npy')
train_visual_classifier(df_train, cats_nr_train)

headers = ['', 'Normal', 'Attack']


source = ColumnDataSource(df.iloc[:100])
colors = Category10[10]
plot = figure(x_range=(0, 1), y_range=(0, 1), width=800, height=400,
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


radius_slider = Slider(start=0, end=0.2, value=0.01, step=0.0001, title="Radius for visual", format="0[.]0000")
radius_source = ColumnDataSource({'x': [], 'y': [], 'rad': []})
plot.circle('x', 'y', radius='rad', source=radius_source, line_color='black',
            line_width=2, color=None, line_dash='dashed', line_alpha=0.7)


def get_attributes_cb(attr, old, new):
    get_attributes(source, df, radius_source, radius_slider, point_info, point_probabilities)


radius_slider.on_change('value', get_attributes_cb)

plot.add_tools(TapTool(behavior='select'))
source.on_change('selected', get_attributes_cb)
# plot.add_tools(HoverTool(tooltips="@index", show_arrow=False, point_policy='follow_mouse'))


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


time_slider, speed_slider, flows_max_slider, button = get_widgets(curdoc, source, df)


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
