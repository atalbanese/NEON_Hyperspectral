library(sp)

library(raster)

library(neonUtilities)

library(neonOS)

library(geoNEON)

library(dplyr)
library(lubridate)


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

veg$date.x <- ymd(veg$date.x)

veg_filtered <- veg %>% filter(grepl("Live", plantStatus)) %>%
                        filter(!grepl("\\d\\d\\d\\d\\d[A-Z]", individualID)) %>%
                        filter(!is.na(height)) %>%
                        filter(!is.na(taxonID)) %>%
                        filter(!is.na(adjNorthing)) %>%
                        filter(!is.na(date.x)) %>%
                        filter(height > 2) %>%
                        #filter(date.x < ymd("2021-01-01")) %>%
                        arrange(-height) %>%
                        group_by(individualID) %>%
                        slice_head(n=1)

write.csv(veg_filtered, "W:/Classes/Research/Niwo_data_by_me.csv", row.names = FALSE)



chm <- raster("C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test/NIWO_452000_4432000_chm.tif")                        
              
plot(chm)

vegsub <- veg_filtered[which(veg_filtered$adjEasting >= extent(chm)[1] &
                      veg_filtered$adjEasting <= extent(chm)[2] &
                      veg_filtered$adjNorthing >= extent(chm)[3] & 
                      veg_filtered$adjNorthing <= extent(chm)[4]),]

bufferCHM <- extract(chm, 
                     cbind(vegsub$adjEasting,
                           vegsub$adjNorthing),
                     buffer=vegsub$adjCoordinateUncertainty, 
                     fun=max)



plot(bufferCHM~vegsub$height, pch=20, xlab="Height", 
     ylab="Canopy height model")

lines(c(0,50), c(0,50), col="grey")

cor(bufferCHM, vegsub$height, use="complete")
