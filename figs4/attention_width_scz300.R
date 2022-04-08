# library
library(igraph)

data <- read.csv("./scz_300_unique_pair.csv")
class_go <- read.csv("../fig4-e/scz_logits_300_annot_bp_class.csv")

colorr <- c()
# create data:
links <- data.frame(
  source=data$node1,
  target=data$node2,
  #importance=log(data$importance,10) 
  importance=data$importance,10
  #importance=ifelse(data$importance < 0.05,0,data$importance)
)
nodes <- data.frame(
  name=unique( append(c(data$node1[!duplicated(data$node1)]),c(data$node2[!duplicated(data$node2)]))) 
)

legend_s = data.frame(
  col = c("#FFFF33","#4DAF4A","#377EB8","#F781BF","#984EA3","#E41A1C","#A65628","#FF7F00"), #brewer.pal(8, "Set1"),
  name = c("oligosaccharide metabolic process","gene expression regulation",
           "cytoskeleton organization","nervous system development",
           "synaptic signaling","cell-cell adhesion",
           "other","ion transport")
)

# adding color
for (nn in nodes$name)
  colorr <- append(colorr ,class_go[class_go[1] == nn][3])
# adding color
# mapping color
for (i in c(1:length(colorr))){
  if (is.na(colorr[i]))
    colorr[i] <- "#A65628"
  if (colorr[i] == ""){
    colorr[i] <- "#A65628"
    next;
  } 
  if (colorr[i] == "ger"){
    colorr[i] <- "#377EB8"
    next;
  } 
  if (colorr[i] == "nc"){
    colorr[i] <- "#FF7F00"
    next;
  } 
  if (colorr[i] == "cyt"){
    colorr[i] <- "#4DAF4A"
    next;
  } 
  if (colorr[i] == "nsd"){
    colorr[i] <- "#984EA3"
    next;
  } 
  if (colorr[i] == "cca"){
    colorr[i] <- "#FFFF33"
    next;
  } 
  if (colorr[i] == "ion"){
    colorr[i] <- "#F781BF"
    next;
  } 
  if (colorr[i] == "gly"){
    colorr[i] <- "#E41A1C"
    next;
  } 
  
  
}

# mapping color

# Turn it into igraph object
network <- graph_from_data_frame(d=links, vertices=nodes, directed=F) 

# Make a palette of 3 colors
library(RColorBrewer)
coul  <- brewer.pal(3, "Set1") 

# Create a vector of color
my_color <- coul[as.numeric(as.factor(V(network)$carac))]

#Make the plot
plot(network,
     layout=layout.fruchterman.reingold,
     vertex.color=colorr,
     edge.arrow.mode=0,
     vertex.label.cex =1,
     edge.width=E(network)$importance*150,
     edge.color = "#D9D9D9",
     vertex.size=2*log(5))
legend(x=1.0289738, y=1.801983, legend=levels(as.factor(legend_s$name))  , col = legend_s$col ,#"bottomleft",
       bty = "n", pch=20 , pt.cex = 3, cex = 0.8
       , text.col=legend_s$col , horiz = FALSE, inset = c(0.1, 0.1))  #, text.col=coul