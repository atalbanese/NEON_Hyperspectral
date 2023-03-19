suppressPackageStartupMessages(require(optparse))

option_list = list(
  make_option(c("-s", "--sitename"), type="character", default=NULL, 
              help="NEON Sitename"),
  make_option(c("-d", "--basedir"), type="character", default=NULL, 
              help="Base directory to create site folder in"),
  make_option(c("-w", "--windowsize"), type="numeric", default=4, 
              help="Search window size for treetop id")
)


opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if(is.null(opt$sitename)){
  print_help(opt_parser)
  stop("Missing sitename", call.=FALSE)
}

if(is.null(opt$basedir)){
  print_help(opt_parser)
  stop("Missing basedir", call.=FALSE)
}


library(lidR)
library(raster)
library(sf)
library(dplyr)

#Main function
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


setwd(file.path(opt$basedir, opt$sitename))

all_ttops <- get_ttops()

st_write(all_ttops, paste(opt$sitename, "tree_tops.gpkg", sep="_"))

# test_chm <- "C:/Users/tonyt/Documents/Research/final_data/NIWO/CHM/NEON_D13_NIWO_DP3_450000_4432000_CHM.tif"
# open_chm <- raster(test_chm)
# ttops <- locate_trees(open_chm, lmf(ws=4),  uniqueness = "bitmerge")
# 
# 
# tv <- vector(mode="list", length=2)
# tv[[1]] <- ttops
# tv[[2]] <-ttops
# bound <- bind_rows(tv)
# st_write(bound, "C:/Users/tonyt/Documents/Research/final_data/NIWO/test.geojson")
