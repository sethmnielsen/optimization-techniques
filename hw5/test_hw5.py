import numpy as np
from hw5 import TrajectoryGenerator
import pytest
import pickle
from typing import Dict

@pytest.fixture(scope='module')
def tg():
    tg = TrajectoryGenerator()
    tg.init_problem()
    yield tg

@pytest.fixture(scope='module')
def data():
    with open('/home/seth/school/optimization/hw5/data.pkl', 'rb') as f:
        data = pickle.load(f)
    yield data
    del data
    
def test_position(tg:TrajectoryGenerator, data:Dict):
    px_arr = np.array(data['px'])
    py_arr = np.array(data['py'])
    x_arr = np.array(data['x'])
    y_arr = np.array(data['y'])
    for i in range(len(px_arr)):
        pxt = px_arr[i]
        pyt = py_arr[i]
        xt = x_arr[i]
        yt = y_arr[i]
        x, y = tg.position(pxt, pyt)
    
        assert np.allclose(x,xt)
        assert np.allclose(y,yt)

def test_velocity(tg:TrajectoryGenerator, data:Dict):
    px_arr = np.array(data['px'])
    py_arr = np.array(data['py'])
    x_arr = np.array(data['vx'])
    y_arr = np.array(data['vy'])
    for i in range(len(px_arr)):
        pxt = px_arr[i]
        pyt = py_arr[i]
        xt = x_arr[i]
        yt = y_arr[i]
        x, y = tg.velocity(pxt, pyt)
    
        assert np.allclose(x,xt)
        assert np.allclose(y,yt)

def test_acceleration(tg:TrajectoryGenerator, data:Dict):
    px_arr = np.array(data['px'])
    py_arr = np.array(data['py'])
    x_arr = np.array(data['ax'])
    y_arr = np.array(data['ay'])
    for i in range(len(px_arr)):
        pxt = px_arr[i]
        pyt = py_arr[i]
        xt = x_arr[i]
        yt = y_arr[i]
        x, y = tg.acceleration(pxt, pyt)
    
        assert np.allclose(x,xt)
        assert np.allclose(y,yt)

def test_jerk(tg:TrajectoryGenerator, data:Dict):
    px_arr = np.array(data['px'])
    py_arr = np.array(data['py'])
    jerk_arr = np.array(data['obj'])
    for i in range(len(px_arr)):
        pxt = px_arr[i]
        pyt = py_arr[i]
        jerkt = jerk_arr[i]
        jerk = tg.jerk(pxt, pyt)
    
        assert np.allclose(jerk, jerkt, rtol=1e-2)

def test_gamma(tg:TrajectoryGenerator, data:Dict):
    px_arr = np.array(data['px'])
    py_arr = np.array(data['py'])
    gam_arr = np.array(data['gam'])
    gam2_arr = np.array(data['gam2'])
    for i in range(len(px_arr)):
        pxt = px_arr[i]
        pyt = py_arr[i]
        gamt = gam_arr[i]
        gam2t = gam2_arr[i]
        
        vx, vy = tg.velocity(pxt,pyt)
        ax, ay = tg.acceleration(pxt,pyt)
        
        v = np.sqrt(vx**2 + vy**2)
        a = np.sqrt(ax**2 + ay**2)
        
        current_gam, gam_minus, gam_plus = tg.gamma(v, vx, vy, ax, ay)
        assert np.allclose(gam_minus, gamt, rtol=1e-2)
        assert np.allclose(gam_plus, gam2t, rtol=1e-2)
        