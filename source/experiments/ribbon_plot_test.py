import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go

import urllib.request as urllib
import numpy as np

url = "https://raw.githubusercontent.com/plotly/datasets/master/spectral.csv"
f = urllib.urlopen(url)
spectra=np.loadtxt(f, delimiter=',')

def main():
    traces = []
    y_raw = spectra[:, 0] # wavelength
    sample_size = spectra.shape[1]-1
    for i in range(1, sample_size):
        z_raw = spectra[:, i]
        print(z_raw[i])
        print(y_raw[i])

        x = []
        y = []
        z = []
        ci = int(255/sample_size*i) # ci = "color index"
        for j in range(0, len(z_raw)):
            z.append([z_raw[j], z_raw[j]])
            y.append([y_raw[j], y_raw[j]])
            x.append([i*2, i*2+1])
        print(x)
        print(y)
        print(z)
        exit(1)
        trace = dict(
            z=z,
            x=x,
            y=y,
            colorscale=[ [i, 'rgb(%d,%d,255)'%(ci, ci)] for i in np.arange(0,1.1,0.1) ],
            showscale=False,
            type='surface',
        )
        print(trace)
        exit(1)
        traces.append(dict(
            z=z,
            x=x,
            y=y,
            colorscale=[ [i, 'rgb(%d,%d,255)'%(ci, ci)] for i in np.arange(0,1.1,0.1) ],
            showscale=False,
            type='surface',
        ))

    fig = go.Figure()
    fig.add_traces(traces)
    fig.write_html("ribbon_test.html")

if __name__ == "__main__":
    main()