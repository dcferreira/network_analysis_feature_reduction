from collections import deque
import numpy as np
import pandas as pd
from tabulate import tabulate
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.plotting import output_notebook, figure
from bokeh.models import (ColumnDataSource, Button, Slider, CategoricalColorMapper,
                          HoverTool, CDSView, GroupFilter, Legend, Div)
from bokeh.palettes import Category10


MAX_POINTS = 1000


df = pd.read_pickle('../dataframe.pkl')
categories = ['attack_cat_Analysis', 'attack_cat_Backdoor', 'attack_cat_DoS',
              'attack_cat_Exploits', 'attack_cat_Fuzzers', 'attack_cat_Generic',
              'attack_cat_Reconnaissance', 'attack_cat_Shellcode', 'attack_cat_Worms',
              'label_0']
categories_short = ['Analysis', 'Backdoor', 'DoS',
                    'Exploits', 'Fuzzers', 'Generic',
                    'Reconnaissance', 'Shellcode', 'Worms',
                    'label_0']
headers = ['', 'Normal', 'Attack']


data_list = deque([])
new_data_x = [deque([]) for i in range(len(categories))]
new_data_y = [deque([]) for i in range(len(categories))]
sources = [ColumnDataSource({'x': [], 'y': []}) for i in range(len(categories))]
colors = Category10[10]
plot = figure(x_range=(0,1), y_range=(0,1), width=800, height=400,
              tools='hover')
color_mapper = CategoricalColorMapper(palette=colors, factors=categories)
legend_list = []
for i in range(len(categories)):
    cat = categories[i]
    rend = plot.scatter('x', 'y', color=colors[i], source=sources[i])
    legend_list.append((cat, [rend]))

legend = Legend(items=legend_list, location=(20, 0))
legend.click_policy = 'hide'
plot.add_layout(legend, 'left')
#plot.add_tools(HoverTool(tooltips="@index", show_arrow=False, point_policy='follow_mouse'))


def animate_update():
    flow_nr = slider.value + 1
    if flow_nr >= len(df):
        flow_nr = 0
    slider.value = flow_nr

def slider_update(attrname, old, new):
    flow_nr = slider.value
    # get data
    dpoint = df.iloc[flow_nr]
    cat = int(dpoint.category)
    source = sources[cat]
    nd_x = new_data_x[cat]
    nd_y = new_data_y[cat]

    # add new point to data
    nd_x.append(dpoint.x_cats_ae)
    nd_y.append(dpoint.y_cats_ae)
    data_list.append(dpoint)
    source.data = {'x': nd_x, 'y': nd_y}

    add_to_stats(dpoint)

    # remove old points
    if len(data_list) >= MAX_POINTS:
        old_dpoint = data_list[0]
        old_cat = int(old_dpoint.category)
        new_data_x[old_cat].popleft()
        new_data_y[old_cat].popleft()
        data_list.popleft()

        sources[old_cat].data = {'x': new_data_x[old_cat], 'y': new_data_y[old_cat]}
        remove_from_stats(old_dpoint)

    update_table()


conf_matrix = np.zeros((len(categories), 2))
cmatrix = Div()
def add_to_stats(dpoint):
    x, y = get_table_coords(dpoint)
    conf_matrix[x, y] += 1

def remove_from_stats(dpoint):
    x, y = get_table_coords(dpoint)
    conf_matrix[x, y] -= 1

def get_table_coords(dpoint):
    return int(dpoint.category), int(dpoint.cats_ae_pred)

def update_table():
    cmatrix.text = tabulate(add_column_headers(conf_matrix), headers=headers, tablefmt='html', numalign='center')

def add_column_headers(mat):
    out = mat.T.tolist()
    out.insert(0, ['<b>%s</b>' % x for x in categories_short])
    return np.array(out).T
    

slider = Slider(start=0, end=len(df), value=0, step=1, title="Flow nr")
slider.on_change('value', slider_update)

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 10)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [slider, button],
    [plot],
    [cmatrix],
])#, sizing_mode='scale_height')


curdoc().add_root(layout)
