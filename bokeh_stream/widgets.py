from bokeh.models import Slider, Button












callback_id = None


def get_widgets(curdoc, source, df):

    def update_speed_cb(attrname, old, new):
        update_speed(curdoc, animate_update_cb)

    def slider_update_cb(attrname, old, new):
        slider_update(source, df, time_slider)



    return speed_slider, flows_max_slider
