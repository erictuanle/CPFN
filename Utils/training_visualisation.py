import torch
import numpy as np
from torch import nn
from visdom import Visdom

ORANGE = np.array([[255, 105, 0]])
BLUE = np.array([[40, 40, 255]])
RED = np.array([[255, 40, 40]])

class Visualiser(object):
    def __init__(self, plotting_interval, port=8097):
        self.vis = Visdom(port=port)
        self.line_plotter = VisdomLinePlotter(self.vis)
        self.plotting_interval = plotting_interval
        self.plotting_step = 0
        self.loss_history_dict = {}
        self.image_dict = {}
        self.window_elements = []

    def log_image(self, image, name):
        image = torch.clamp(image, 0, 1)
        image = image.cpu().detach().numpy()
        self.image_dict[name] = image
        if not name in self.window_elements:
            self.window_elements.append(name)
    
    def log_loss(self, loss, name):
        current_history = self.loss_history_dict.get(name, [np.nan] * self.plotting_interval)
        updated_history = current_history[1:] + [loss]
        self.loss_history_dict[name] = updated_history
        if not name in self.window_elements:
            self.window_elements.append(name)

    def update(self):
        if self.plotting_step % self.plotting_interval == 0:
            loss_avg_dict = {k: torch.tensor(self.loss_history_dict[k]).mean().item() for k in self.loss_history_dict}
            for name in loss_avg_dict:
                loss_avg = loss_avg_dict[name]
                self.line_plotter.plot(name, name, name, self.plotting_step, loss_avg, color=ORANGE)
            for name in self.image_dict:
                self.vis.image(self.image_dict[name], opts=dict(title=name), win=self.window_elements.index(name))
        self.plotting_step += 1

class VisdomLinePlotter(object):
    def __init__(self, vis, env_name='main'):
        self.vis = vis 
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, color):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                linecolor=color,
                xlabel='Training steps',
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env,
                opts=dict(
                legend=[split_name],
                title=title_name,
                linecolor=color,
                xlabel='Training steps',
                ylabel=var_name
            ),
                  win=self.plots[var_name], name=split_name, update = 'append')