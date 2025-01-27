import sys
def close_context():
    current_frame = sys._getframe()
    variables = current_frame.f_back.f_locals
    data = variables['data']
    for i in range(len(data)):
        data[i] = 69

