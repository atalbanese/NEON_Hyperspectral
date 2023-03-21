import os
import numpy as np
import itertools
from typing import Literal
import math
from ortools.linear_solver import pywraplp

class TreeData:
    def __init__(
        self,
         hyperspectral,
         rgb, 
         rgb_mask,
         hyperspectral_mask,
         hyperspectral_bands,
         chm,
         utm_origin,
         taxa,
         plot_id, 
         site_id,
         mpsi,
         ndvi,
         pca,
         algo_type,
         file_loc
        ):
        self.hyperspectral = hyperspectral
        self.rgb = rgb
        self.rgb_mask = rgb_mask
        self.hyperspectral_mask = hyperspectral_mask
        self.hyperspectral_bands = hyperspectral_bands
        self.chm = chm
        self.ndvi = ndvi
        self.mpsi = mpsi
        self.pca = pca
        self.utm_origin = utm_origin
        self.algo_type = algo_type
        self.taxa = str(taxa[()])
        self.plot_id = str(plot_id[()])
        self.site_id = str(site_id[()])
        self.file_loc = file_loc

    @classmethod
    def from_npz(cls, file_loc):
        with np.load(file_loc) as data:
            new_instance = cls(**data, file_loc=file_loc)
        return new_instance


    def pad_arr(self, arr, out_dim):
        three_d = len(arr.shape) == 3
        y_pad = out_dim - arr.shape[0]
        x_pad = out_dim - arr.shape[1]

        if y_pad <0 or x_pad<0:
            print(f"{self.file_loc}larger than given dimension, cropping..")
            return arr[:out_dim, :out_dim,...]
        else:
            if three_d:
                return np.pad(arr, [(0,y_pad), (0,x_pad), (0,0)], mode='mean')
            else:
                return np.pad(arr, [(0,y_pad), (0,x_pad)])


    def get_dict(self, choices):
        return_dict = dict()
       
        if "hs" in choices:
            return_dict['hs'] = self.hyperspectral
            return_dict['hs_mask'] = self.hyperspectral_mask
        
        if "chm" in choices:
            return_dict['chm'] = self.chm
        
        if "rgb" in choices:
            return_dict["rgb"] = self.rgb
        
        if "origin" in choices:
            return_dict["utm_origin"] = self.utm_origin

        if "mpsi" in choices:
            return_dict['mpsi'] = self.mpsi

        if "pca" in choices:
            return_dict['pca'] = self.pca
        
        if "ndvi" in choices:
            return_dict['ndvi'] = self.ndvi

        
        return_dict['taxa'] = self.taxa


        return return_dict

class PlotData:
    def __init__(
        self,
        name,
        taxa,
        init_tree = None
    ):

        self.name = name
        self.trees = {t:[] for t in taxa}
        if init_tree is not None:
            self.trees[init_tree.taxa] = [init_tree]

    def add_tree(self, tree: TreeData):
        self.trees[tree.taxa].append(tree)


    @property
    def taxa_counts(self):
        out = dict()
        for k, v in self.trees.items():
            out[k] = len(v)
        return out

    @property
    def taxa_array(self):
        return [v for v in self.taxa_counts.values()]
    
    @property
    def num_trees(self):
        return sum([len(v) for v in self.trees.values()])

    @property
    def all_trees(self):
        return list(itertools.chain(*[v for v in self.trees.values()]))

class SiteData:
    def __init__(
        self,
        site_dir: str,
        random_seed: int,
        train: float,
        test: float,
        valid: float,
        ndvi_filter = 0.2,
        mpsi_filter = 0.03,
        apply_filters = False
        ):

        self.site_dir = site_dir
        self.all_trees = self.find_all_trees()
        self.filter_trees(ndvi_filter, mpsi_filter, apply_filters)
        self.all_taxa = self.find_all_taxa()
        self.all_plots = self.find_all_plots()
        self.key = {k: ix for ix, k in enumerate(sorted(self.all_taxa.keys()))}

        self.rng = np.random.default_rng(random_seed)
        self.train_proportion = train
        self.test_proportion = test
        self.valid_proportion = valid

        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        self.split_solution = None

    @property
    def taxa_counts(self):
        out = dict()
        for k, v in self.all_taxa.items():
            out[k] = len(v)
        return out
    
    @property
    def num_plots(self):
        return len(self.all_plots.keys())

    @property
    def num_taxa(self):
        return len(self.all_taxa.keys())

    @property
    def class_weights(self):
        if self.training_data is not None:
            class_weights = {}
            #n_samples / (n_classes * class_count)
            n_classes = len(self.all_taxa.keys())
            n_samples = len(self.training_data)
            for k in self.all_taxa.keys():
                class_count = len(["x" for tree in self.training_data if tree.taxa == k])
                class_weights[k] = n_samples/(n_classes*class_count)
            return class_weights
            
        else:
            return None

    def find_all_trees(self):
        all_dirs = [os.scandir(d) for d in os.scandir(self.site_dir) if d.is_dir()]
        return [TreeData.from_npz(f.path) for f in itertools.chain(*all_dirs) if f.name.endswith('.npz')]
    
    def filter_trees(self, ndvi_filter, mpsi_filter, apply_filters):
        #This acts as a check if apply_filters is false, this operation should have been done when trees were made
        to_drop = []
        for ix, tree in enumerate(self.all_trees):
            tree = tree.get_dict(['chm', 'pca', 'ndvi'])
            chm_mask = tree['chm'] > 1.99

            if apply_filters:
                ndvi_mask = tree['ndvi'] > ndvi_filter
                #Todo: check this is the right direction for mpsi
                mpsi_mask = tree['mpsi'] > mpsi_filter
                chm_mask = chm_mask * ndvi_mask * mpsi_mask
            
            if chm_mask.sum() <= 0:
                to_drop.append[ix]
        to_drop = set(to_drop)
        filtered_trees = [i for j, i in enumerate(self.all_trees) if j not in to_drop]
        self.all_trees = filtered_trees


    def find_all_plots(self):
        plots_dict = dict()
        for tree in self.all_trees:
            if tree.plot_id in plots_dict:
                plots_dict[tree.plot_id].add_tree(tree)
            else:
                plots_dict[tree.plot_id] = PlotData(tree.plot_id, self.all_taxa.keys(), tree)
        return plots_dict

    def find_all_taxa(self):
        taxa_dict = dict()
        for tree in self.all_trees:
            if tree.taxa in taxa_dict:
                taxa_dict[tree.taxa].append(tree)
            else:
                taxa_dict[tree.taxa] = [tree]
        return {k: taxa_dict[k] for k in sorted(taxa_dict)}
    
    def make_splits(self, split_style: Literal["tree", "plot"]):
        if split_style == "tree":
            self.make_tree_level_splits()
        if split_style == "plot":
            self.make_plot_level_splits()

    def make_tree_level_splits(self):
        training, testing, validation = [], [], []

        #Splits trees proportionally to how frequently taxa appear ie if there are 100 pine trees and 10 spruce and a 6/2/2 split, there will be 60/20/20 pines and 6/2/2 spruce
        for tree_list in self.all_taxa.values():
            list_length = len(tree_list)
            train_len = math.floor(self.train_proportion * list_length)
            validation_len = math.floor(self.valid_proportion * list_length)

            self.rng.shuffle(tree_list)

            training = training + tree_list[0:train_len]
            validation = validation + tree_list[train_len:train_len+validation_len]
            testing = testing + tree_list[train_len+validation_len:]


        self.training_data = training
        self.testing_data = testing
        self.validation_data = validation

    def make_plot_level_splits(self):
        training_goals = [math.floor(v*self.train_proportion) for v in self.taxa_counts.values()]
        testing_goals = [math.floor(v*self.test_proportion) for v in self.taxa_counts.values()]
        valid_goals = [v for v in self.taxa_counts.values()]
        valid_goals = [v - training_goals[ix] -testing_goals[ix] for ix, v in enumerate(valid_goals)]

        solutions = self.solve_splits([training_goals, testing_goals, valid_goals], ['training', 'testing', 'valid'])

        #Get every tree from every selected plot and unpack them into a flat list
        self.training_data = list(itertools.chain(*[self.all_plots[plot].all_trees for plot in solutions['training']]))
        self.testing_data = list(itertools.chain(*[self.all_plots[plot].all_trees for plot in solutions['testing']]))
        self.validation_data = list(itertools.chain(*[self.all_plots[plot].all_trees for plot in solutions['valid']]))
        #Save the solution in case we need it later
        self.split_solution = solutions

    def solve_splits(self, goals_array, goals_labels, max_buffer=5, min_buffer=5) -> dict:
  
        taxa_counts = [plot.taxa_array for plot in self.all_plots.values()]
        num_plots = len(self.all_plots.keys())
        num_goals = len(goals_array)

        solver = pywraplp.Solver.CreateSolver('SCIP')

        vars_dict = dict()

        #For each plot and each split category (testing, training, validation), we want to create a binary variable to choose which category that plot ends up in
        for i, plot in enumerate(self.all_plots.keys()):
            for j in range(num_goals):
                vars_dict[i, j] = solver.IntVar(0, 1, plot)

        #Our objective terms to minimize
        objective_terms = []

        #For each split category and each taxa, we want to find the difference between (taxa total across selected plots) and (category goal)
        for i in range(num_goals):
            for j in range(self.num_taxa):
                #Taxa Goal for Category - Sum(Taxa count in assigned to category)
                objective_terms.append(goals_array[i][j]- solver.Sum([vars_dict[k, i] * taxa_counts[k][j] for k in range(num_plots)]))

                #Make sure things dont go too high or too low above the goal, theres no precise solution usually so we need to create a zone of constraints
                #Theres definitely a way to do this by minimizing squared or absolute value difference instead but ortools is confusing
                #Sum of taxa in a category must be less than the goal + buffer
                solver.Add(solver.Sum([vars_dict[k, i] * taxa_counts[k][j] for k in range(num_plots)]) <= goals_array[i][j] + max_buffer)
                #Sum of taxa in a category must be greater than the goal + buffer
                solver.Add(solver.Sum([vars_dict[k, i] * taxa_counts[k][j] for k in range(num_plots)]) >= goals_array[i][j] - min_buffer)
        
        for i in range(num_plots):
            solver.Add(solver.Sum([vars_dict[i, j] for j in range(num_goals)]) == 1)
        
        solver.Minimize(solver.Sum(objective_terms))
        status = solver.Solve()

        solutions_dict = {k:[] for k in goals_labels}
        
        #Split each plot into a category based on the solution.
        for k, v in vars_dict.items():
            solution_cat = goals_labels[k[1]]
            if v.solution_value() == 1.0:
                solutions_dict[solution_cat].append(v.name())

        #Print the solution
        sol_text = ""
        for i, (k, v) in enumerate(solutions_dict.items()):
            sol_text = sol_text + f"{k} Solution:\n"
            for j, taxa in enumerate(self.all_taxa.keys()):
                sol_text = sol_text + f'{taxa}: Goal: {goals_array[i][j]}, Solution: {sum([self.all_plots[plot].taxa_counts[taxa] for plot in v])}\n'
        print(sol_text)
        solutions_dict['sol_text'] = sol_text

        return solutions_dict

    def get_data(self, 
        data_selection: Literal["training", "testing", "validation", "training and validation", "all"], 
        data_choices, 
        make_key=False):
        working_data = self.select_working_data(data_selection)
        data_list = []
        for tree in working_data:
            to_append = tree.get_dict(data_choices)
            if make_key:
                to_append['target_arr'] = self.make_key(tree)
                to_append['single_target'] = np.array(self.key[tree.taxa], dtype=np.float32)
                to_append['pixel_target'] = np.ones((self.num_taxa, self.num_taxa), dtype=np.float32)*self.key[tree.taxa]
                channel_target = np.zeros((self.num_taxa, self.num_taxa, len(self.key.values())), dtype=np.float32)
                channel_target[...,self.key[tree.taxa]] = 1.0
                to_append['channel_target'] = channel_target
        
            data_list.append(to_append)
        return data_list
    

    def make_key(self, tree):
        new_key = np.zeros((self.num_taxa), np.float32)

        this_tree = tree.taxa
        new_key[self.key[this_tree]] = 1.0
        return new_key

    def select_working_data(self, data_selection):
        if data_selection == "training":
            return self.training_data
        if data_selection == "testing":
            return self.testing_data
        if data_selection == "validation":
            return self.validation_data
        if data_selection == "all":
            return self.all_trees
        if data_selection == "training and validation":
            return self.training_data + self.validation_data


if __name__ == "__main__":
    test = SiteData(
        site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.1,
        valid = 0.3)


    test.make_splits('plot_level')
    for x in test.get_data('training', ['hs', 'chm', 'rgb', 'origin'], 16, make_key=True):
        print(x)

   
