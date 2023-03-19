suppressPackageStartupMessages(require(optparse))

option_list = list(
  make_option(c("-s", "--sitename"), type="character", default=NULL, 
              help="NEON Sitename"),
  make_option(c("-d", "--savedir"), type="character", default=NULL, 
              help="Base directory to create site folder in"),
)


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


library(lidR)
library(raster)
library(terra)
library(future)
library(sf)




#Main function
get_crown_stats <- function(chunk) {
  if (is.null(chunk)) {
    return(NULL)
  }
  

  # Read a chunk out of the LASCatalog
  las <- readLAS(chunk)

  # Check if empty
  if (is.empty(las)) {
    return(NULL)
  }

  norm <- lidR::normalize_height(las, knnidw(), na.rm=TRUE)
  norm <- filter_poi(norm, Z>=0, Z<=200)
  ttops <- locate_trees(norm, lmf(ws =3),  uniqueness = "bitmerge")

  return(ttops)
}


##I THINK THESE NEED TO BE RERUN
process_all_ttops <- function(cat_folder, out_folder){
  ctg <- readLAScatalog(cat_folder, filter = "-drop_withheld")
  # Set output
  opt_output_files(ctg) <- paste0(out_folder, "/NIWO_{XLEFT}_{YBOTTOM}_ttops")
  ctg@output_options$drivers$sf$extension <- ".geojson"
  

 # plan(multisession)
  opt_chunk_size(ctg) <- 1000
  opt_chunk_buffer(ctg) <- 0
  options <- list(automerge = TRUE)
  
  # Make kernel and get crown stats for all files in LASCatalog

  output <- catalog_apply(ctg, get_crown_stats, .options = options)
  return(out_folder)
}

process_all_chm <- function(cat_folder, out_folder){
  ctg <- readLAScatalog(cat_folder, filter = "-drop_withheld")
  # Set output
  opt_output_files(ctg) <- paste0(out_folder, "/NIWO_{XLEFT}_{YBOTTOM}_chm")
  
  
 # plan(multisession)
  opt_chunk_size(ctg) <- 1000
  opt_chunk_buffer(ctg) <- 0
  options <- list(automerge = TRUE)
  
  # Make kernel and get crown stats for all files in LASCatalog
  
  output <- catalog_apply(ctg, get_chms, .options = options)
  return(out_folder)
}

get_chms <- function(chunk) {
  
  if (is.null(chunk)) {
    return(NULL)
  }
  
  
  # Read a chunk out of the LASCatalog
  las <- readLAS(chunk)
  
  # Check if empty
  if (is.empty(las)) {
    return(NULL)
  }
  fill.na <- function(x, i=5) { if (is.na(x)[i]) { return(mean(x, na.rm = TRUE)) } else { return(x[i]) }}
  w <- matrix(1, 3, 3)
  norm <- lidR::normalize_height(las, knnidw(), na.rm=TRUE)
  norm <- filter_poi(norm, Z>=0, Z<=200)
  chm <- rasterize_canopy(norm, res = 0.1, pitfree(thresholds = c(0, 10, 20), max_edge = c(0, 2)))
  chm <- terra::focal(chm, w, fun = fill.na)
  chm <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
  return(chm)
}




niwo_folder <- "C:/Users/tonyt/Documents/Research/datasets/lidar/niwo_point_cloud/temp"
out_folder <- "C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test"

# test_file <- "C:/Users/tonyt/Documents/Research/datasets/lidar/niwo_point_cloud/valid_sites_orig/NEON_D13_NIWO_DP1_452000_4432000_classified_point_cloud_colorized.laz"
# col <- height.colors(25)
# 
# chm<- get_chms(test_file)
# # w <- matrix(1, 3, 3)
# 
# filled <- terra::focal(chm, w, fun = fill.na)
# smoothed <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
# 
# plot(chm, col = col)

process_all_chm(niwo_folder, out_folder)

