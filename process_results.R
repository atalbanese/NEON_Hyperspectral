library(dplyr)
library(ggplot2)


results <- read.csv('C:/Users/tonyt/Documents/Research/dl_model/lidar_hs_unsup_dl_model/experiment_set_2_results.csv')


niwo_medians <- results %>%
            group_by(exp_number) %>%
            mutate(ova_median = median(test_ova)) %>%
            select(-test_ova) %>%
            slice_head(n=1) %>%
            ungroup %>%
            filter(inp_key != 'hs') %>%
            filter(sitename == 'NIWO') %>%
            arrange(model, anno_method, remove_taxa)

harv_medians <- results %>%
              group_by(exp_number) %>%
              mutate(ova_median = median(test_ova)) %>%
              select(-test_ova) %>%
              slice_head(n=1) %>%
              ungroup %>%
              filter(inp_key != 'hs') %>%
              filter(sitename == 'HARV')%>%
              arrange(model, anno_method, remove_taxa)



base_plot <- ggplot(data=niwo_medians, aes(split_method, ova_median)) + geom_col()
panels <- base_plot + facet_wrap( ~ anno_method + model + remove_taxa, labeller = label_both) + ggtitle('NIWO - PCA')
panels


base_plot <- ggplot(data=harv_medians, aes(split_method, ova_median)) + geom_col()
panels <- base_plot + facet_wrap( ~ anno_method + model, labeller = label_both, nrow=3) + ggtitle('HARV - PCA')
panels
