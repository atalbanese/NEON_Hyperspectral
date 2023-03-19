suppressPackageStartupMessages(require(optparse))

option_list = list(
  make_option(c("-s", "--sitename"), type="character", default=NULL, 
              help="NEON Sitename"),
  make_option(c("-d", "--savedir"), type="character", default=NULL, 
              help="Base directory to create site folder in"),
  make_option(c("-y", "--year"), type="character", default=NULL, 
              help="Year for which data will be downloaded"),
  make_option(c("-w", "--windowsize"), type="numeric", default=4, 
              help="Search window size for treetop id")
); 

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if(is.null(opt$sitename)){
  print_help(opt_parser)
  stop("Missing sitename", call.=FALSE)
}

if(is.null(opt$savedir)){
  print_help(opt_parser)
  stop("Missing savedir", call.=FALSE)
}

if(is.null(opt$year)){
  print_help(opt_parser)
  stop("Missing year", call.=FALSE)
}

suppressPackageStartupMessages(require(sp))
suppressPackageStartupMessages(require(sf))
suppressPackageStartupMessages(require(raster))
suppressPackageStartupMessages(require(neonUtilities))
suppressPackageStartupMessages(require(neonOS))
suppressPackageStartupMessages(require(geoNEON))
suppressPackageStartupMessages(require(dplyr))
suppressPackageStartupMessages(require(lubridate))
suppressPackageStartupMessages(require(fs))
suppressPackageStartupMessages(require(lidR))

download_and_filter_veg <- function(sitename) {
  veglist <- loadByProduct(dpID="DP1.10098.001", 
                           site=sitename, 
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
    rename("easting_plot"=easting, "northing_plot"=northing) %>%
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
    select(plotID, individualID, adjEasting, adjNorthing, taxonID, height, ninetyCrownDiameter, canopyPosition, adjCoordinateUncertainty, utmZone) %>%
    rename("easting_tree" = adjEasting, "northing_tree" = adjNorthing) %>%
    arrange(-height) %>%
    group_by(individualID) %>%
    slice_head(n=1) %>%
    ungroup() %>%
    left_join(plot_locs, by="plotID", keep=FALSE)
  
  
  return(veg_filtered)
}

download_rs_products <- function(sitename, veg_filtered, year){
  #RGB
  byTileAOP(dpID="DP3.30010.001", site=sitename, 
            year=year, easting=veg_filtered$easting_tree,
            northing=veg_filtered$northing_tree,
            buffer=20,
            check.size=FALSE)
  #CHM
  byTileAOP(dpID="DP3.30015.001", site=sitename, 
            year=year, easting=veg_filtered$easting_tree,
            northing=veg_filtered$northing_tree,
            buffer=20,
            check.size=FALSE)
  
  #HS
  byTileAOP(dpID="DP3.30006.001", site=sitename, 
            year=year, easting=veg_filtered$easting_tree,
            northing=veg_filtered$northing_tree,
            buffer=20,
            check.size=FALSE)
  
  #Move downloaded files into more human readable folders
  dt <- dir_tree()
  
  rgb_files <- dt[grepl("*_image.tif", dt)]
  chm_files <- dt[grepl("*_CHM.tif", dt)]
  hs_files <- dt[grepl("*.h5", dt)]
  
  file_move(rgb_files, "RGB")
  file_move(chm_files, "CHM")
  file_move(hs_files, "HS")
  
}

get_chm_dif <- function(veg_filtered){
  all_chm <- list.files("CHM", full.names=TRUE)
  frames <- vector(mode="list", length=length(all_chm))
  
  for (i in 1:length(all_chm)) {
    chm <- raster(all_chm[i])
    vegsub <- veg_filtered[which(veg_filtered$easting_tree >= extent(chm)[1] &
                                   veg_filtered$easting_tree <= extent(chm)[2] &
                                   veg_filtered$northing_tree >= extent(chm)[3] & 
                                   veg_filtered$northing_tree <= extent(chm)[4]),]
    if (nrow(vegsub)>0){
      vegsub$chm_height <- extract(chm, 
                                   cbind(vegsub$easting_tree,
                                         vegsub$northing_tree),
                                   buffer=vegsub$adjCoordinateUncertainty, 
                                   fun=max)
      vegsub$chm_dif <- abs(vegsub$chm_height-vegsub$height)
      frames[[i]] <- vegsub
    }
  }
  
  return(bind_rows(frames))
}

get_ttops <- function() {
  all_chm <- list.files("CHM", full.names=TRUE)
  frames <- vector(mode="list", length=length(all_chm))
  
  for (i in 1:length(all_chm)) {
    chm <- raster(all_chm[i])
    ttops <- locate_trees(chm, lmf(ws=opt$windowsize, shape='circular'),  uniqueness = "bitmerge")
    frames[[i]] <- ttops
  }
  
  return(bind_rows(frames))
}


##MAIN SCRIPT
#Change to opt options for command line
sitename <- opt$sitename
basedir <- opt$savedir
year <- opt$year

#Filesystem management
dir.create(file.path(basedir, sitename), showWarnings = FALSE)
dir.create(file.path(basedir, sitename, "CHM"), showWarnings = FALSE)
dir.create(file.path(basedir, sitename, "RGB"), showWarnings = FALSE)
dir.create(file.path(basedir, sitename, "HS"), showWarnings = FALSE)
dir.create(file.path(basedir, sitename, "PCA"), showWarnings = FALSE)
setwd(file.path(basedir, sitename))

#Get data on woody vegetation and sampling plots
veg_filtered <- download_and_filter_veg(sitename)

#Get spatially relevant RS products
download_rs_products(sitename, veg_filtered, year)

#Find the difference between the canopy height model and the observed height from ground survey
veg_filtered <- get_chm_dif(veg_filtered)

write.csv(veg_filtered, paste(sitename,year,"woody_vegetation.csv", sep="_"), row.names = FALSE)

#Get treetops from canopy height model
all_ttops <- get_ttops()

st_write(all_ttops, paste(opt$sitename, "tree_tops.gpkg", sep="_"))



