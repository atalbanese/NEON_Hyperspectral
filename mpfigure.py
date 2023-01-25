
import rasterio.features as rf
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.style as mplstyle
from matplotlib.widgets import Button, RectangleSelector
from matplotlib.patches import Rectangle
import math
import torch
from einops import rearrange

mplstyle.use('fast')

class MultiPanelFigure:
    def __init__(self, fig, axes, hs_image, tru_color_hs, rgb_image, hs_bands, slice_params, polygon, hs_affine, rgb_affine, tree_crowns_hs, title, taxa, taxonID, f_name, save_dir):
        self.f_name = f_name
        self.save_dir = save_dir
        self.taxonID = taxonID
        self.title = title
        self.spectrograph = axes[0]
        self.derivative_graph = axes[1]
        self.hs_viz = axes[2]
        self.rgb_viz = axes[3]
        self.all_axes = axes
        self.fig = fig
        self.taxa = taxa
        self.hs_image = hs_image
        self.tru_color_hs = tru_color_hs
        self.rgb_image = rgb_image

        self.hs_overlay, self.selected_hs_pixels = make_tree_mask(polygon, hs_affine, slice_params, 1, 2.5, 1.0, all_touched=False)
        self.rgb_overlay, self.selected_rgb_pixels = make_tree_mask(polygon, rgb_affine, slice_params, 10, 1.0, 0.5, all_touched=False)

        self.hs_crowns_x, self.hs_crowns_y = tree_crowns_hs
        self.rgb_crowns_x, self.rgb_crowns_y = [crown*10 for crown in self.hs_crowns_x], [crown*10 for crown in self.hs_crowns_y]

        self.bands = hs_bands
        self.slice_params = slice_params

        self.rgb_ticks = np.arange(5, 400, 10)

        self.hs_xlim = (0, 0)
        self.hs_ylim= (0, 0)

        self.rgb_xlim = (0, 0)
        self.rgb_ylim = (0, 0)

        self.line_holder = dict()

        self.draw_spectral_plots()
        self.hs_im = self.draw_hs_plot()
        self.rgb_im = self.draw_rgb_plot()
        self.save_button = self.draw_save_button()
        self.patch_select_button = self.draw_select_button()
        self.selector_rect = RectangleSelector(self.hs_viz, self.select_callback, useblit=True, minspanx=1.0, minspany=1.0)
        self.selector_rect.set_active(False)
        self.last_patch = None

        plt.suptitle(title)

        self.current_axis = None

    
    def draw_save_button(self):
        ax_button = self.fig.add_axes([0.45, 0.05, 0.1, 0.06])
        save_button = Button(ax_button, 'Save Tensor')
        save_button.on_clicked(self.save_selected_as_tensor)
        return save_button

    def save_selected_as_tensor(self, _):
        if self.last_patch is not None:
            extents = self.last_patch.get_bbox()
            y_min = math.ceil(extents.y0)
            y_max = math.ceil(extents.y1)
            x_min = math.ceil(extents.x0)
            x_max = math.ceil(extents.x1)

            cropped = self.hs_image[y_min:y_max,x_min:x_max,...]
            target_mask = self.selected_hs_pixels[y_min:y_max,x_min:x_max]

            orig_crop = torch.tensor(cropped, dtype=torch.float32)
            orig_crop = rearrange(orig_crop, 'h w c -> c h w')

            label = torch.zeros((len(self.taxa.keys()),*target_mask.shape), dtype=torch.float32)
            label[self.taxa[self.taxonID]] = 1.0

            to_save = {
                'orig': orig_crop,
                'mask': target_mask,
                'target': label,
                #'height': height,
            }

            
            with open(os.path.join(self.save_dir, self.f_name), 'wb') as f:
                torch.save(to_save, f)

        pass

    def make_mask(self):
        pass

    def handle_patch_click(self, _):
        self.selector_rect.set_active(not self.selector_rect.active)


    def select_callback(self, eclick, erelease):
        x1, y1 = math.floor(eclick.xdata-0.5) +0.5, math.floor(eclick.ydata-0.5) +0.5
        x2, y2 = round(erelease.xdata)+ 0.5, round(erelease.ydata)+ 0.5
        new_patch = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red')
        self.hs_viz.add_patch(new_patch)
        self.last_patch = new_patch
        self.hs_viz.figure.canvas.draw()
        pass
        

    def draw_select_button(self):
        ax_button = self.fig.add_axes([0.3, 0.05, 0.1, 0.06])
        draw_select = Button(ax_button, 'Draw')
        draw_select.on_clicked(self.handle_patch_click)
        return draw_select

    def draw_spectral_plots(self):
        selected_indexes = np.argwhere(self.selected_hs_pixels)
        current_lines_length = len(self.spectrograph.get_lines())
        tracker=0
        for ix, location in enumerate(selected_indexes):
            location = tuple(location)
            
            if location not in self.line_holder:
                pix = self.hs_image[location[0], location[1]]
                per_change = np.diff(pix) / pix[1:]
                self.spectrograph.plot(self.bands, pix, alpha=0.8)
                self.derivative_graph.plot(self.bands[1:], per_change, alpha=0.8)
                self.derivative_graph.set_ylim((-1,1))
                
                self.line_holder[location] = current_lines_length + tracker
                tracker+=1

        remove_keys = []
        remove_values = []
        for k, v in self.line_holder.items():
            if not self.selected_hs_pixels[k[0],k[1]]:
                remove_keys.append(k)
                remove_values.append(v)
                self.spectrograph.lines.pop(v)
                self.derivative_graph.lines.pop(v)
        
        remove_values.sort()
        for ix in remove_values:
            temp_dict = self.line_holder.copy()
            for k, v in temp_dict.items():
                if v > ix:
                    self.line_holder[k] = v - 1
        
        for remove_key in remove_keys:
            del self.line_holder[remove_key]


    def draw_hs_plot(self):
        viz_im = self.hs_viz.imshow(self.tru_color_hs*self.hs_overlay, picker=True)
        self.hs_viz.scatter(self.hs_crowns_x, self.hs_crowns_y)
        self.hs_xlim = self.hs_viz.get_xlim()
        self.hs_ylim = self.hs_viz.get_ylim()
        return viz_im

    def draw_rgb_plot(self):
        rgb_im = self.rgb_viz.imshow(self.rgb_image*self.rgb_overlay, picker=True)
        self.rgb_viz.scatter(self.rgb_crowns_x, self.rgb_crowns_y)
        self.rgb_viz.set_xticks(self.rgb_ticks)
        self.rgb_viz.set_yticks(self.rgb_ticks)
        self.rgb_viz.grid()
        self.rgb_xlim = self.rgb_viz.get_xlim()
        self.rgb_ylim = self.rgb_viz.get_ylim()
        return rgb_im
        
    def on_click(self, event):
        if not self.selector_rect.active:
            artist = event.artist
            if artist.axes == self.hs_viz:
                self.handle_hs_click(event)
            if artist.axes == self.rgb_viz:
                self.handle_rgb_click(event)
            self.update_after_click()

    
    def on_enter_axis(self, event):
        self.current_axis = event.inaxes
    
    
    def on_release(self, event):
        if not self.selector_rect.active:
            if self.current_axis == self.hs_viz:
                self.handle_hs_release()
            if self.current_axis == self.rgb_viz:
                self.handle_rgb_release()
    
    def handle_hs_release(self):
        xlim_check = self.hs_viz.get_xlim()
        ylim_check = self.hs_viz.get_ylim()

        if xlim_check != self.hs_xlim:
            self.hs_xlim = xlim_check
            self.rgb_viz.set_xlim(xlim_check[0]*10, xlim_check[1]*10)

        if ylim_check != self.hs_ylim:
            self.hs_ylim = ylim_check
            self.rgb_viz.set_ylim(ylim_check[0]*10, ylim_check[1]*10)

        self.rgb_viz.figure.canvas.draw()


    def handle_rgb_release(self):
        xlim_check = self.rgb_viz.get_xlim()
        ylim_check = self.rgb_viz.get_ylim()

        if xlim_check != self.rgb_xlim:
            self.rgb_xlim = xlim_check
            self.hs_viz.set_xlim(xlim_check[0]/10, xlim_check[1]/10)

        if ylim_check != self.rgb_ylim:
            self.rgb_ylim = ylim_check
            self.hs_viz.set_ylim(ylim_check[0]/10, ylim_check[1]/10)

        self.hs_viz.figure.canvas.draw()

    
    def handle_hs_click(self, event):
        x_loc = round(event.mouseevent.xdata)
        y_loc = round(event.mouseevent.ydata)
        self.selected_hs_pixels[y_loc, x_loc] = ~self.selected_hs_pixels[y_loc, x_loc]
        self.hs_overlay = make_tree_overlay(self.selected_hs_pixels, 2.5, 1.0)

    
    def handle_rgb_click(self, event):
        x_loc = round(event.mouseevent.xdata/10)
        y_loc = round(event.mouseevent.ydata/10)
        self.selected_hs_pixels[y_loc, x_loc] = ~self.selected_hs_pixels[y_loc, x_loc]
        self.hs_overlay = make_tree_overlay(self.selected_hs_pixels, 2.5, 1.0)



    def update_after_click(self):
        self.hs_im.set_data(self.tru_color_hs*self.hs_overlay)
        self.hs_im.axes.figure.canvas.draw()

        self.draw_spectral_plots()
        self.spectrograph.figure.canvas.draw()
        self.derivative_graph.figure.canvas.draw()



def make_tree_overlay(mask, upper, lower):
    mask = mask * upper
    mask[mask == 0] = lower
    return mask[...,np.newaxis]


def make_tree_mask(polygon, transform, slice_params, scale, upper, lower, all_touched=False):
    mask = rf.geometry_mask([polygon], [1000*scale, 1000*scale], transform=transform, all_touched=all_touched, invert=True)
    ym, yma, xm, xma = slice_params
    mask = mask[ym*scale:yma*scale, xm*scale:xma*scale, ...]
    return make_tree_overlay(mask, upper, lower), mask

def make_tree_plot(hs_image, tru_color_hs, rgb_image, hs_bands, slice_params, polygon, hs_affine, rgb_affine, tree_crowns_hs, plotID, tree_id, save_dir, ix, crown_diam, taxa, taxonID, f_name):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax=np.ravel(ax)
    title = tree_id + ' - ' + plotID + ' - ' + str(crown_diam) + 'm'
    this_fig = MultiPanelFigure(fig, ax, hs_image, tru_color_hs, rgb_image, hs_bands, slice_params, polygon, hs_affine, rgb_affine, tree_crowns_hs, title, taxa, taxonID, f_name, save_dir)
   
    # if save_dir is not None:
    #     plt.savefig(os.path.join(save_dir, f'{plotID}_{tree_id}{ix}.png'))
    #plt.tight_layout()
    fig.canvas.mpl_connect('axes_enter_event', this_fig.on_enter_axis)
    fig.canvas.mpl_connect('pick_event', this_fig.on_click)
    fig.canvas.mpl_connect('button_release_event', this_fig.on_release)
    plt.show()

def myround(x, prec=2, base=.5):
  return round(base * round(float(x)/base),prec)