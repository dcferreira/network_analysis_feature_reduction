from bokeh.models import Slider, Button


def animate_update(time_slider, df):
    flow_nr = time_slider.value + 1
    if flow_nr >= len(df):
        flow_nr = 0
    time_slider.value = flow_nr


def slider_update(source, df, time_slider):
    flow_nr = time_slider.value
    # get data
    source.stream(df.iloc[flow_nr], flows_max_slider.value)


def update_speed(curdoc, animate_update_cb):
    global callback_id
    curdoc.remove_periodic_callback(callback_id)  # this is not working
    callback_id = curdoc.add_periodic_callback(animate_update_cb, speed_slider.value)


def animate(curdoc, animate_update_cb):
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc.add_periodic_callback(animate_update_cb, speed_slider.value)
    else:
        button.label = '► Play'
        curdoc.remove_periodic_callback(callback_id)


button = Button(label='► Play')
speed_slider = Slider(start=10, end=500, value=10, title="New flow every (ms)")
flows_max_slider = Slider(start=10, end=10000, value=200, step=10, title="Number of flows to keep")

callback_id = None


def get_widgets(curdoc, source, df):
    time_slider = Slider(start=0, end=len(df), value=0, step=1, title="Flow nr")

    def animate_update_cb():
        animate_update(time_slider, df)

    def animate_cb():
        animate(curdoc, animate_update_cb)

    def update_speed_cb(attrname, old, new):
        update_speed(curdoc, animate_update_cb)

    def slider_update_cb(attrname, old, new):
        slider_update(source, df, time_slider)

    button.on_click(animate_cb)
    speed_slider.on_change('value', update_speed_cb)
    time_slider.on_change('value', slider_update_cb)

    return time_slider, speed_slider, flows_max_slider, button
