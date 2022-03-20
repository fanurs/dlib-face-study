import base64
import binascii
import glob
import hashlib
import io
import sqlite3
import pathlib

import cv2
import dash
import face_recognition as fr
import numpy as np
import pandas as pd
import PIL
import plotly.io as pio
pio.templates.default = 'plotly_white'
import plotly.graph_objects as go
from sklearn.decomposition import PCA

class ImagePreProcessor:
    @staticmethod
    def crop_landscape(frame):
        shape_x, shape_y = frame.shape[1], frame.shape[0]
        if shape_x > shape_y:
            x0 = int((shape_x - shape_y) / 2)
            frame = frame[:, x0:x0+shape_y]
        return frame

    @staticmethod
    def resize_smaller(frame, min_side_size=360):
        shape_x, shape_y = frame.shape[1], frame.shape[0]
        ratio = min_side_size / min(shape_x, shape_y)
        frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
        return frame

    @staticmethod
    def standardize_size(frame):
        frame = ImagePreProcessor.crop_landscape(frame)
        frame = ImagePreProcessor.resize_smaller(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
class ImageHash:
    @staticmethod
    def hash_sha256(std_frame):
        return hashlib.sha256(std_frame.tobytes()).digest()

    @staticmethod
    def b64_encode(sha256_hashed_bytes):
        return base64.b64encode(sha256_hashed_bytes).decode('utf-8')

    @staticmethod
    def b64_decode(b64_str):
        return binascii.hexlify(base64.b64decode(b64_str)).decode('utf-8')

    @staticmethod
    def hash_frame(std_frame):
        return ImageHash.b64_encode(ImageHash.hash_sha256(std_frame))

class ImageDatabase:
    database_path = pathlib.Path('./data.db')
    database_name = 'data'
    column_names = [
        'hash',
        'path',
        'name',
        *[f'loc[{i}]' for i in range(4)],
        *[f'enc[{i}]' for i in range(128)],
    ]
    column_dtypes = [
        'str',
        'str',
        'str',
        *['float' for _ in range(4)],
        *['float' for _ in range(128)],
    ]

    def __init__(self):
        self.database = sqlite3.connect(str(self.database_path))
    
    def _reset(self):
        df = pd.DataFrame(columns=self.column_names)
        for i, dtype in enumerate(self.column_dtypes):
            df.iloc[:, i] = df.iloc[:, i].astype(dtype)
        self._to_sql(df, 'replace')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def close(self):
        self.database.close()

    def _to_sql(self, df, if_exists, **kwargs):
        return df.to_sql(
            self.database_name,
            con=self.database,
            if_exists=if_exists,
            index=False,
            **kwargs
        )
    
    def _row_to_dict(self, row):
        return dict(
            hash=row['hash'],
            path=str(row['path']),
            name=row['name'],
            face_loc=np.array([row[f'loc[{i}]'] for i in range(4)]),
            face_enc=np.array([row[f'enc[{i}]'] for i in range(128)]),
        )

    def cache_frame(self, std_frame, path, name, face_loc, face_enc):
        df = pd.DataFrame([[
            ImageHash.hash_frame(std_frame),
            str(path),
            name,
            *face_loc,
            *face_enc,
        ]], columns=self.column_names)
        self._to_sql(df, 'append')
        return self._row_to_dict(df.loc[0])

    def load_cache(self, hash):
        query = pd.read_sql(
            f'SELECT * FROM {self.database_name} WHERE hash = "{hash}"',
            con=self.database,
        )
        if len(query) == 0:
            return False
        if len(query) > 1:
            print(f'Hash collision for "{hash}":')
            print(query[['name', 'path']])
            return False
        series = query.iloc[0]
        return self._row_to_dict(series)

    def encode_frame(self, frame, path, force_recache=False):
        data = None
        if not force_recache:
            data = self.load_cache(ImageHash.hash_frame(frame))
        if data:
            return data
        face_locations = fr.face_locations(frame)
        face_encodings = fr.face_encodings(frame, face_locations)
        if len(face_encodings) == 0:
            print(f'No face found in "{path}"')
            return False
        return self.cache_frame(frame, path, path.parent.name, face_locations[0], face_encodings[0])

def b64_frame(frame):
    buffer = io.BytesIO()
    PIL.Image.fromarray(frame).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')



img_paths = glob.glob('images/*/*.*')
encodings = dict()
b64_images = dict()
with ImageDatabase() as db:
    for i, path in enumerate(img_paths):
        path = pathlib.Path(path)
        path_truncated_str = str(path)[:40] if len(str(path)) > 40 else str(path)
        print('\r{:40s}'.format(path_truncated_str), flush=True, end='')
        frame = cv2.imread(str(path))
        frame = ImagePreProcessor.standardize_size(frame)
        data = db.encode_frame(frame, path, force_recache=True if i == 0 else False)
        name = data['name']
        if name in encodings:
            encodings[name].append(data['face_enc'])
        else:
            encodings[name] = [data['face_enc']]
        b64_images[data['name']] = b64_frame(frame)
print()

for name, enc in encodings.items():
    encodings[name] = np.array(enc).mean(axis=0)
encodings = pd.DataFrame(encodings).T



"""Performance PCA and visualize using Plotly Dash"""
n_components = 3
df_pca = PCA(n_components=n_components).fit_transform(encodings)
df_pca = pd.DataFrame(df_pca, columns=list(range(n_components)))
df_pca.insert(0, 'name', encodings.index)

min_1 = df_pca[1].min()
max_1 = df_pca[1].max()
def sizes(x, x_min, x_max):
    size = 5 * (x - x_min) / (x_max - x_min) + 5
    return int(size)

scat3d = go.Scatter3d(
    x=10*df_pca[0], y=10*df_pca[1], z=10*df_pca[2],
    mode='markers',
    hoverinfo=None,
    hovertemplate='%{text}<extra></extra>',
    text=df_pca['name'],
    marker=dict(
        color=df_pca[0],
        colorscale='viridis_r',
        size=[2 * sizes(x, min_1, max_1) for x in df_pca[1]],
    ),
)
fig = go.Figure(data=scat3d)
fig.update_layout(
    width=1000,
    height=800,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        camera=dict(
            eye=dict(x=2.0, y=1.0, z=0.3),
        ),
    )
)



if __name__ == '__main__':
    app = dash.Dash(__name__)
    app.layout = dash.html.Div(
        className='container',
        children=[
            dash.dcc.Graph(id='face-plot', figure=fig, clear_on_unhover=True),
            dash.dcc.Tooltip(id='tooltip', direction='bottom')
        ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'height': '100vh',
        }
    )

    @app.callback(
        dash.Output('tooltip', 'show'),
        dash.Output('tooltip', 'bbox'),
        dash.Output('tooltip', 'children'),
        dash.Input('face-plot', 'hoverData'),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, dash.no_update, dash.no_update
        
        hover_data = hoverData['points'][0]
        bbox = hover_data['bbox']
        num = hover_data['pointNumber']

        children = [
            dash.html.Div([
                dash.html.Img(
                    src=b64_images[df_pca.iloc[num]['name']],
                    style={
                        'display': 'block',
                        'width': '120px',
                        'margin': '0',
                        'padding': '0',
                    }
                )
            ])
        ]
        return True, bbox, children

    app.run_server(debug=False, port=5500)
