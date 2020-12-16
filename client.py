import os
import requests

http_url = 'http://127.0.0.1:5003'

def inference(file_path):
    files = {}

    if not os.path.exists(file_path):
        return None
    # In (FileName, DataStream, MimeType) format?
    files['image'] = (os.path.basename(file_path), open(file_path, 'rb'), 'image/jpeg')
    response = requests.post(http_url+'/inference', files=files)
    result = response.json()
    result['httpcode'] = response.status_code

    # if 'result' in result:
    #     return result['result']
    # else:
    #     return None
    return result


if __name__ == '__main__':
    file_path = './data_coco_kp/three_point/TestImages/({}).jpg'.format(1)
    for i in range(1000):
        print(inference(file_path))
    # import drgk_pose.data as D
    # import numpy as np
    # points = np.array([[100, 3], [200, 156]])
    # target = D.generate_target(points)
    # print(target.shape)
    # target = target.flatten()
    # print(target.shape)
