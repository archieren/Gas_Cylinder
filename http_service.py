import os
import sys
import traceback
import json
import numpy as np
from flask import Flask, request
from PIL import Image
from tensorgroup.services.model_gas import model_gas as MG
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.

app = Flask(__name__)

config = {
    'saved_model_dir' : os.path.join(os.getcwd(), 'work'),
    'pgn_i_size': 1024,
    'kpn_i_size': 512,
    'centernet_datasetname': 'catenary',
    'classes_num': 8 ,
    'keypointnet_datasetname': 'three_point'}

model = MG(config)

@app.route("/")
def hello():
    return "Hello!"

@app.route("/test")
def test():
    html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>object detection</title>
        </head>
        <body>
            <form action="/inference" method="post" enctype="multipart/form-data">
                <input type="file" name="image"/><br>
                <input type="submit" value="Detect">
            </form>
        </body>
        </html>
    '''
    return html


@app.route('/inference', methods=['POST'])
def inference():
    result = {}

    try:
        file = request.files['image']
        # file is an werkzeug datastructure FileStorage! ‘如同’一般文件
        print(file.filename)
        print(file.mimetype)
        image = Image.open(file)
        # file.save('tmp_image.dat')
        # x_test = np.load('tmp_image.dat')
        image = image.convert("RGB")
        image_array = np.array(image)
        p_result, k_result = model.inference(image_array)
        # print(output)

        result['ret'] = 0
        result['msg'] = 'success'
        result['result'] = (p_result.shape, k_result.shape)
    except Exception as e:
        print('{} error {}'.format(sys._getframe().f_code.co_name, traceback.format_exc()))
        result['ret'] = 0
        result['msg'] = e.args[0]
    finally:
        # print(result)
        return json.dumps(result, ensure_ascii=False, default=lambda o: o.__dict__)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5003)
