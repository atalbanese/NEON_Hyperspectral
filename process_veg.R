require(sp)
library(raster)
library(neonUtilities)
library(neonOS)
library(geoNEON)
library(dplyr)
library(lubridate)
library(ggplot2)
library(patchwork)

veglist <- loadByProduct(dpID="DP1.10098.001", 
                         site="NIWO", 
                         package="basic", 
                         check.size = FALSE)

vegmap <- getLocTOS(veglist$vst_mappingandtagging, 
                    "vst_mappingandtagging")

veg <- joinTableNEON(veglist$vst_apparentindividual, 
                     vegmap, 
                     name1="vst_apparentindividual",
                     name2="vst_mappingandtagging")

plot_locs <- veglist$vst_perplotperyear %>%
            select(plotID, easting, northing) %>%
            rename("easting_plot"=easting, "northing_plot"=northing)
            group_by(plotID) %>%
            slice_head(n=1) %>%
            ungroup() 

veg$date.x <- ymd(veg$date.x)

veg_filtered <- veg %>% filter(grepl("Live", plantStatus)) %>%
  filter(!grepl("\\d\\d\\d\\d\\d[A-Z]", individualID)) %>%
  filter(!is.na(height)) %>%
  filter(!is.na(taxonID)) %>%
  filter(!is.na(adjNorthing)) %>%
  filter(!is.na(date.x)) %>%
  filter(height > 2) %>%
  filter(date.x < ymd("2021-01-01")) %>%
  select(plotID, individualID, adjEasting, adjNorthing, taxonID, height, ninetyCrownDiameter, canopyPosition) %>%
  rename("easting_tree" = adjEasting, "northing_tree" = adjNorthing) %>%
  arrange(-height) %>%
  group_by(individualID) %>%
  slice_head(n=1) %>%
  ungroup()


#write.csv(veg_filtered, "W:/Classes/Research/Niwo_data_by_me.csv", row.names = FALSE)

all_chm <- list.files("C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test/", full.names=TRUE)
frames <- vector(mode="list", length=length(all_chm))

for (i in 1:length(all_chm)) {
  chm <- raster(all_chm[i])
  vegsub <- veg_filtered[which(veg_filtered$adjEasting >= extent(chm)[1] &
                                 veg_filtered$adjEasting <= extent(chm)[2] &
                                 veg_filtered$adjNorthing >= extent(chm)[3] & 
                                 veg_filtered$adjNorthing <= extent(chm)[4]),]
  if (nrow(vegsub)>0){
    vegsub$chm_height <- extract(chm, 
                                 cbind(vegsub$adjEasting,
                                       vegsub$adjNorthing),
                                 buffer=vegsub$adjCoordinateUncertainty, 
                                 fun=max)
    vegsub$chm_dif <- abs(vegsub$chm_height-vegsub$height)
    frames[[i]] <- vegsub
  }
}

unfiltered <- bind_rows(frames)

height_dif_sd <- sd(unfiltered$chm_dif)

filtered <- unfiltered[unfiltered$chm_dif<height_dif_sd,]



before <- ggplot(unfiltered) + geom_point(aes(x=height, y= chm_height)) +
          labs(x="Height (m) from Ground Survey", y="Height (m) from Canopy Height Model", title="Before Filtering") + xlim(0, 40)
after <- ggplot(filtered) + geom_point(aes(x=height, y= chm_height)) +
  labs(x="Height (m) from Ground Survey", y="", title="After Filtering") + xlim(0, 40)

before+after+plot_annotation(title="Ground Survey Tree Heights vs Canopy Height Model at Same Coordinates at NIWO",
                              subtitle = "Points filtered to be within one standard deviation of absolute value of height difference")


write.csv(unfiltered, "W:/Classes/Research/Niwo_Canopy_Height_Dif.csv", row.names = FALSE)






