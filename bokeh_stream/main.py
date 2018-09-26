import numpy as np
import pandas as pd
from tabulate import tabulate
from bokeh.io import curdoc
from bokeh.layouts import Column, Row
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, Slider, CategoricalColorMapper, Select, Button,
                          CDSView, GroupFilter, Legend, Div, TapTool, Circle, CircleCross)
from bokeh.palettes import Category10
from attributes import get_attributes, train_visual_classifier, categories, categories_short, binary


# get these files by running `generate_pickles.py` (need to add root to PYTHONPATH)
df = pd.read_pickle('dataframe.pkl')
df_train = np.load('cats_ae_x_train_scaled.npy')
cats_nr_train = np.load('cats_nr_train.npy')
try:
    train_visual_classifier(df_train, cats_nr_train)
except AssertionError:
    pass  # avoids "Train has already been called once" error

headers = ['', 'Normal', 'Attack']


source = ColumnDataSource(df.iloc[:100])
colors = Category10[10]
global_plot = figure(x_range=(0, 1), y_range=(0, 1), width=800, height=400,
                     tools='hover')
color_mapper = CategoricalColorMapper(palette=colors, factors=categories)

print(global_plot.renderers)


radius_slider = Slider(start=0, end=0.2, value=0.01, step=0.0001, title="Radius for visual classifier",
                       format="0[.]0000")
radius_source = ColumnDataSource({'x': [], 'y': [], 'rad': []})

color_from_dropdown = Select(title="Color by:", value='Categories',
                             options=['Categories', 'cats_ae_pred', 'original_pred'])


global_plot = Row(children=[])


def add_renderers():
    color_attribute, label_mapping = {
        'Categories': ('cat_str', categories),
        'cats_ae_pred': ('cats_ae_pred_str', binary),
        'original_pred': ('original_pred_str', binary),
    }[color_from_dropdown.value]
    plot = figure(x_range=(0, 1), y_range=(0, 1), width=800, height=400,
                  tools='hover')
    print(plot.renderers)
    legend_list = []
    for i in range(len(pd.unique(df.loc[:, color_attribute]))):
        cat = label_mapping[i]
        view = CDSView(source=source, filters=[GroupFilter(column_name=color_attribute, group=str(i))])
        rend = plot.scatter('x_cats_ae', 'y_cats_ae', color=colors[i], size=15,
                            line_color='black', source=source, view=view)
        rend.selection_glyph = Circle(fill_alpha=1, fill_color=colors[i], line_color='black')
        rend.nonselection_glyph = CircleCross(fill_alpha=0.1, fill_color=colors[i], line_color=colors[i])
        legend_list.append((cat, [rend]))

    legend = Legend(items=legend_list, location=(20, 0))
    legend.click_policy = 'hide'
    plot.add_layout(legend, 'left')

    plot.circle('x', 'y', radius='rad', source=radius_source, line_color='black',
                line_width=2, color=None, line_dash='dashed', line_alpha=0.7)
    plot.add_tools(TapTool(behavior='select'))
    return plot


def reload_plot(attr, old, new):
    global_plot.children = [add_renderers()]


color_from_dropdown.on_change('value', reload_plot)

global_plot.children = [add_renderers()]
point_info = Div()
point_probabilities = Div()



def get_attributes_cb(attr, old, new):
    # global_plot.children = [add_renderers()]
    get_attributes(source, df, radius_source, radius_slider, point_info, point_probabilities)


radius_slider.on_change('value', get_attributes_cb)


source.on_change('selected', get_attributes_cb)


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


# play button & slider
def slider_update(attrname, old, new):
    flow_nr = time_slider.value
    # get data
    source.stream(df.iloc[flow_nr], flows_max_slider.value)


time_slider = Slider(start=0, end=len(df), value=0, step=1, title="Flow nr")
time_slider.on_change('value', slider_update)
speed_slider = Slider(start=10, end=500, value=10, title="New flow every (ms)")
flows_max_slider = Slider(start=10, end=10000, value=200, step=10, title="Number of flows to keep")

callback_id = None


def animate_update():
    flow_nr = time_slider.value + 1
    if flow_nr >= len(df):
        flow_nr = 0
    time_slider.value = flow_nr


def update_speed(attrname, old, new):
    global callback_id
    curdoc.remove_periodic_callback(callback_id)  # this is not working
    callback_id = curdoc.add_periodic_callback(animate_update, speed_slider.value)


speed_slider.on_change('value', update_speed)


def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, speed_slider.value)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)


button = Button(label='► Play')
button.on_click(animate)


layout = Column(children=[
    Row(children=[
        Column(children=[button, speed_slider]),
        Column(children=[time_slider, flows_max_slider]),
        Column(children=[radius_slider, color_from_dropdown]),
    ]),
    Row(children=[
        global_plot,
        Column(children=[point_info, point_probabilities]),
    ]),
    Row(children=[
        cmatrix_cats_ae,
        cmatrix_original,
    ])
])

curdoc().add_root(layout)
