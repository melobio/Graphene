# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)



##data
data <- read.csv("./mental_8_GatLogit_normLong_R1_norm.csv")

dis_id <- data.frame(ids = unique(c(as.character(data$dis1), as.character(data$dis2))))
edges <- data.frame(from = "origin", to = dis_id)

vertices  <-  data.frame(
  name = unique(c(as.character(edges$from), as.character(edges$ids)))
) 
##################label
vertices$id <- NA
myleaves <- which(is.na(vertices$id ))
nleaves <- length(myleaves)
vertices$id[ myleaves ] <- seq(1:nleaves)
vertices$angle <- 90 - 360 * vertices$id / nleaves

vertices$angle <- ifelse(vertices$angle < -90, vertices$angle+180, vertices$angle)

data$value <- ifelse(data$value == 0,0,100*data$value) #log(1000*data$value)

ymin = min(data$value)
ymax = max(data$value)

lower = 0  # lower = 0.06  
upper = 1
#
k = (upper-lower)/(ymax-ymin)
data$value <- ifelse(data$value == 0, 0, lower + k*(data$value - ymin))  
##normalize
mygraph <- igraph::graph_from_data_frame( edges, vertices=vertices )

from1  <-  match( data$dis1, vertices$name)
to1  <-  match( data$dis2, vertices$name)


va1  <- rep(data$value, each = 3)
colorr <- ifelse(va1 == 0, "red", "orange")

from2  <-  match( data$dis1[21:58], vertices$name)
to2  <-  match( data$dis2[21:58], vertices$name)

va2  <-  rep(data$value[21:58], each = 100)*5
####

ggraph(mygraph, layout = 'dendrogram', circular = TRUE) +
  
  geom_node_point(aes(filter = leaf, x = x*1, y=y*1), size = 3, colour = "skyblue", shape = 1, stroke = 1.7) +
  
  geom_conn_bundle(data = get_con(from = from1, to = to1), alpha=0.2, width=5, tension=0.2,aes(colour=va1)) + #, 
  
  scale_edge_color_continuous(low="white", high="red")+
  #scale_edge_colour_distiller(palette = "BuPu") +
  geom_node_text(aes(x = x*1.1, y=y*1.1, filter = leaf, label=name), size=5, alpha=1) +  #, angle = angle
  theme_void() +
  theme(
    #legend.position="none",
    plot.margin=unit(c(0,0,0,0),"cm"),
  ) +
  expand_limits(x = c(-1.2, 1.2), y = c(-1.2, 1.2))

