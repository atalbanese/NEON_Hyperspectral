library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)


# pair_results <- function(df, pair_by){
#   
# }


results_1 <- read.csv('C:/Users/tonyt/Documents/Research/dl_model/lidar_hs_unsup_dl_model/experiment_set_1_results.csv')
results_7 <- read.csv('C:/Users/tonyt/Documents/Research/dl_model/lidar_hs_unsup_dl_model/experiment_set_7_results.csv')
results_4 <- read.csv('C:/Users/tonyt/Documents/Research/dl_model/lidar_hs_unsup_dl_model/experiment_set_4_results.csv')
results_8 <- read.csv('C:/Users/tonyt/Documents/Research/dl_model/lidar_hs_unsup_dl_model/experiment_set_8_results.csv')

medians_1 <- results_1 %>%
              group_by(exp_number) %>%
              mutate(ova_median = median(test_ova)) %>%
              select(-test_ova) %>%
              slice_head(n=1) %>%
              ungroup %>%
              mutate(pre_training_site = "NONE") %>%
              mutate(test_site = sitename) %>%
              mutate(remove_taxa = if_else(remove_taxa=="", 'NONE', remove_taxa)) %>%
              mutate(exp_set = 1) %>% 
              relocate(ova_median, .after = last_col())

medians_7 <- results_7 %>%
              group_by(exp_number) %>%
              mutate(ova_median = median(test_ova)) %>%
              select(-test_ova) %>%
              slice_head(n=1) %>%
              ungroup %>%
              mutate(pre_training_site = str_extract(pt_ckpt, "[A-Z]+"), .keep='unused') %>%
              replace_na(list(pre_training_site="NONE")) %>%
              mutate(test_site = if_else(test_site=="", sitename, test_site)) %>%
              mutate(remove_taxa = if_else(remove_taxa=="", 'NONE', remove_taxa)) %>%
              mutate(exp_set = 7)

medians_8 <- results_8 %>%
            group_by(exp_number) %>%
            mutate(ova_median = median(test_ova)) %>%
            select(-test_ova) %>%
            slice_head(n=1) %>%
            ungroup %>%
            mutate(pre_training_site = str_extract(pt_ckpt, "[A-Z]+"), .keep='unused') %>%
            replace_na(list(pre_training_site="NONE")) %>%
            mutate(test_site=sitename) %>%
            mutate(remove_taxa = if_else(remove_taxa=="", 'NONE', remove_taxa)) %>%
            mutate(exp_set = 8)

medians_4 <- results_4 %>%
              group_by(exp_number) %>%
              mutate(ova_median = median(test_ova)) %>%
              select(-test_ova) %>%
              slice_head(n=1) %>%
              ungroup %>%
              mutate(pre_training_site = "NONE") %>%
              mutate(test_site = sitename) %>%
              mutate(remove_taxa = if_else(remove_taxa=="", 'NONE', remove_taxa)) %>%
              mutate(exp_set = 4) %>%
              relocate(ova_median, .after = last_col())

all_results_with_hs <- bind_rows(medians_1, medians_7, medians_4, medians_8)

all_results <- all_results_with_hs %>% filter(inp_key != 'hs', apply_filters=='FALSE')
all_results$model[all_results$model == 'DL'] <- 'Deep Learning'
all_results$model[all_results$model == 'RF'] <- 'Random Forest'

all_results$anno_method[all_results$anno_method == 'filtering'] <- 'Filtering'
all_results$anno_method[all_results$anno_method == 'snapping'] <- 'Snapping'
all_results$anno_method[all_results$anno_method == 'scholl'] <- 'Scholl'

all_results$split_method[all_results$split_method == 'pixel'] <- 'Pixel'
all_results$split_method[all_results$split_method == 'plot'] <- 'Plot'
all_results$split_method[all_results$split_method == 'tree'] <- 'Tree'

transfer_results <- all_results %>% filter(sitename != test_site, pre_training_site == 'NONE')

same_site_results <- all_results %>% filter(sitename == test_site)

same_site_dl <- same_site_results %>% filter(model == 'DL')
same_site_rf <- same_site_results %>% filter(model == 'RF')

transfer_dl <- transfer_results %>% filter(model == 'DL')
transfer_rf <- transfer_results %>% filter(model == 'RF')

pre_training_results <- all_results %>% filter(pre_training_site != 'NONE')

snapping_results <- all_results %>% filter(anno_method == 'snapping')
filtering_results <- all_results %>% filter(anno_method == 'filtering')
scholl_results <- all_results %>% filter(anno_method == 'scholl')


niwo_results <- same_site_results %>% filter(sitename == 'NIWO', apply_filters == 'FALSE', pre_training_site == 'NONE')
harv_results <- same_site_results %>% filter(sitename == 'HARV', apply_filters == 'FALSE', pre_training_site == 'NONE')


label_dict <- as_labeller(c(
  `PIFL2` = 'Omitted Taxa: Limber Pine',
  `Pixel`="Split Method: Pixel",
  `Plot`="Split Method: Plot",
  `Tree`="Split Method: Tree",
  `NONE` = 'Omitted Taxa: None',
   `Filtering` = 'Label Method: Filtering',
  `Snapping` = 'Label Method: Snapping',
  `Scholl` = 'Label Method: Scholl',
  `RMNP`='Test Site: RMNP',
  `RMNP_from_NIWO`='Test Site: RMNP ← NIWO',
  `NIWO`='Test Site: NIWO'
  ))


#Can we improve on label selection methods by using lidar to calibrate tree locations from ground surveys?
# + scale_color_manual(labels=c('Deep Learning', 'Random Forest'), values=c('#ef8a62', '#67a9cf'))

base_plot <- ggplot(data=niwo_results, aes(anno_method, ova_median, fill=model)) + geom_bar(stat='identity', position='dodge') + scale_fill_brewer(type='qual',palette='Dark2')
panels <- base_plot + 
  facet_grid( remove_taxa ~ split_method , labeller = label_dict) + 
  ggtitle('Label Selection Method') + 
  labs(fill="Model Type", x = "Label Selection Method", y = "Median Overall Accuracy (5 Trials)") +
  theme_bw(base_size=18)
panels
ggsave('C:/Users/tonyt/Documents/Research/thesis_final/Figures/Final_Figures/Results_1_Label_Selection.png', width=14, height=9, units='in')



#Looking at these, label selection is improved at NIWO with filtering but not at HARV

#What role does spatial autocorrelation play in classification results and how can we account for it to make a more general model at NEON sites?


base_plot <- ggplot(data=transfer_results, aes(split_method, ova_median, fill=model))  + geom_bar(stat='identity', position='dodge')+ scale_fill_brewer(type='qual',palette='Dark2')
panels <- base_plot + 
  facet_grid(factor(test_site, levels=c('RMNP_from_NIWO', 'RMNP')) ~ anno_method, labeller = label_dict) + 
  ggtitle('Transfer Learning')+
  labs(fill="Model Type", x = 'Data Splitting Method', y = "Median Overall Accuracy (5 Trials)") +
  theme_bw(base_size=18)
panels
ggsave('C:/Users/tonyt/Documents/Research/thesis_final/Figures/Final_Figures/Results_2_Transfer_Learning.png', width=14, height=9, units='in')

#Does deep learning, including self-supervised deep learning, offer an improvement over machine 
#learning classification techniques for this problem?

dl_proof <- all_results %>% filter(test_site != 'RMNP', test_site!='HARV',model=='Deep Learning', split_method != 'Pixel')
dl_proof$pre_training_site[dl_proof$pre_training_site == 'NONE'] <- 'None'
base_plot <- ggplot(data=dl_proof, aes(split_method, ova_median, fill=factor(pre_training_site, levels=c('None', 'NIWO', 'STEI')))) + geom_bar(aes(), stat='identity', position='dodge') + scale_fill_brewer(type='qual',palette='Dark2', direction = -1)
panels <- base_plot + 
  facet_grid(test_site ~ anno_method, labeller = label_dict) + 
  ggtitle('Pre-Training') + 
  labs(fill="Pre-Training Site", x = 'Data Splitting Method', y = "Median Overall Accuracy (5 Trials)") +
  theme_bw(base_size=18)
panels  
ggsave('C:/Users/tonyt/Documents/Research/thesis_final/Figures/Final_Figures/Results_3_Pre_Training.png', width=14, height=9, units='in')




#Addenda: 
#   How much does PCA fit matter?
#   How much does removing a taxa matter in classification results?
#   Can we use simple NDVI and MPSI filters to improve results?

