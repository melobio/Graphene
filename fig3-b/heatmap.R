library(RColorBrewer)
library(gplots)
library(circlize)
# The mtcars dataset:
#mtgat 300tpgs
csv <- read.csv("./js_value_mat.csv")  # for graphene
#csv <- read.csv("./js_value_mat_gwas.csv")  # for gwas
mat<- data.frame(csv[-1])

mat <- as.matrix(mat)
rownames(mat)<-t(csv[1])
mat <- -log(mat, 10)
mat <- scale(mat)

col <- colorRampPalette(c("orange", "white", "blue"))(256)

mat2 <- t(mat)

heatmap.2(mat, 
          trace="none",
          
          #hclustfun = hclust,
        # cexCol = 1.5,
        dendrogram = "none",
        #Rowv = "False",
        margins=c(12,16),
        cexRow=0.35,
        cexCol = 0.5,
        #symm = FALSE,
        scale = "none",
        #rowsep = c(15,18, 28,31,34,40,45),
        #sepwidth=c(0.25,0.25,0.25,0.25,0.25,0.25,0.25),
        #rowsep,
        density.info = "none",
        keysize = 0.5,
        #srtCol=45,
        col=  col #scale="column", 
        ) #Colv = NA, Rowv = NA,
dev.off()

