import argparse
import numpy as np
from mayavi import mlab
from scipy.integrate import odeint
from moviepy import editor
import tempfile
import shutil
import cv2
import os


def take_input():
    parser = argparse.ArgumentParser(description='Lorenz Equations')
    parser.add_argument('o')
    args = parser.parse_args()
    return args


def lorenzfunc(x, y, z, s=10, r=28, b=8/3):
    u = s*(y - x)
    v = r*x - y - x*z
    w = x*y - b*z
    return u, v, w


def lorenz_ode(state, t):
    x, y, z = state
    return np.array(lorenzfunc(x, y, z))


def plot_save(r, r1, t):
    x, y, z = r.T
    x1, y1, z1 = r1.T
    dirpath = tempfile.mkdtemp()
    mlab.figure(size=(1024, 768))
    for i in range(20, 2000, 20):
        mlab.clf()
        mlab.plot3d(x[:i], y[:i], z[:i], t[0:i], representation='wireframe')
        mlab.points3d(x[i-1], y[i-1], z[i-1])
        mlab.plot3d(x1[:i], y1[:i], z1[:i], t[0:i], representation='wireframe')
        mlab.points3d(x1[i-1], y1[i-1], z1[i-1], mode='2dsquare')
        mlab.view(azimuth=151.45, elevation=78.08, distance=199.75,
                  focalpoint=np.array([2.083, 13.881, 39.950]))
        mlab.savefig(os.path.join(dirpath, "%02d.png" % (i/20)))
    return dirpath


def make_video(folder, fname):
    images = []
    for file in os.listdir(folder):
        images.append(cv2.imread(os.path.join(folder, file)))
    clip = editor.ImageSequenceClip(images, fps=6)
    clip.write_videofile(fname)
    shutil.rmtree(folder)


def main():
    input = take_input()
    t = np.linspace(0, 20, 2000)
    r = odeint(lorenz_ode, (10.0, 50.0, 50.0), t)
    r1 = odeint(lorenz_ode, (10.1, 50,  50), t)
    dest_path = plot_save(r, r1, t)
    make_video(dest_path, input.o)


if __name__ == '__main__':
    main()
