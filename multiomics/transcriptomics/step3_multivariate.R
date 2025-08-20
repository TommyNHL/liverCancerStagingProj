# the R script below is adapted from Daniel Beiting and modified by me

# Load packages ------
library(tidyverse) # you're familiar with this fromt the past two lectures
library(DT) # for making interactive tables
library(plotly) # for making interactive plots
library(gt) # A layered 'grammar of tables' - think ggplot, but for tables

# ==============================================================================

# Identify variables of interest in study design file ----
targets <- read_tsv("studydesign.txt")
targets
group <- targets$group
group <- factor(group)
sampleLabels <- targets$sample

# ==============================================================================

# Prepare your data -------
# for this part of the class you'll use your normalized and filtered data in log2 cpm
log2.cpm.filtered.norm.df <- read_csv("../myDGEListDiesellog2_filtered_normalized.csv")
log2.cpm.filtered.norm.df

load(file = "../myDGEListDieselFilterNorm")
log2.cpm.filtered.norm <- edgeR::cpm(myDGEList.filtered.norm, log=TRUE)

# ==============================================================================

# Hierarchical clustering ---------------
distance <- dist(t(log2.cpm.filtered.norm), method = "euclidean") # "euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"
clusters <- hclust(distance, method = "complete") # "ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid"
plot(clusters, labels=sampleLabels)

# ==============================================================================

# Principal component analysis (PCA) -------------
pca.res <- prcomp(t(log2.cpm.filtered.norm), scale.=F, retx=T)

#look at the PCA result (pca.res) that you just created
ls(pca.res)
summary(pca.res) # Prints variance summary for all principal components.
pca.res$rotation #$rotation shows you how much each gene influenced each PC (called 'scores')
pca.res$x # 'x' shows you how much each sample influenced each PC (called 'loadings')

#note that these have a magnitude and a direction (this is the basis for making a PCA plot)
screeplot(pca.res) # A screeplot is a standard way to view eigenvalues for each PCA
pc.var<-pca.res$sdev^2 # sdev^2 captures these eigenvalues from the PCA result
pc.per<-round(pc.var/sum(pc.var)*100, 1) # we can then use these eigenvalues to calculate the percentage variance explained by each PC
pc.per

# ==============================================================================

# Visualize your PCA result ------------------
pca.res.df <- as_tibble(pca.res$x)

ggplot(pca.res.df) + 
    aes(x=PC1, y=PC2, label=sampleLabels, color=group) + 
    geom_point(size=4) + 
    #geom_label() + 
    stat_ellipse() + 
    xlab(paste0("PC1 (",pc.per[1],"%",")")) + 
    ylab(paste0("PC2 (",pc.per[2],"%",")")) + 
    labs(title="PCA plot", 
         #caption=paste0("produced on ", Sys.time())) + 
         caption="RNA-Seq: Diesel-Treated HepG2 Cells & Control Groups") +
    coord_fixed() + 
    theme_bw()

# ==============================================================================

# Create a PCA 'small multiples' chart ----
pca.res.df <- pca.res$x[,1:4] %>%  # 75%, Magrittr package
    as_tibble() %>% 
    add_column(sample = sampleLabels, 
               group = group)
  
pca.pivot <- pivot_longer(
    pca.res.df, # dataframe to be pivoted
    cols = PC1:PC4, # column names to be stored as a SINGLE variable
    names_to = "PC", # name of that new variable (column)
    values_to = "loadings") # name of new variable (column) storing all the values (data)

ggplot(pca.pivot) + 
    aes(x=sample, y=loadings, fill=group) + 
    geom_bar(stat="identity") + 
    facet_wrap(~PC) + 
    labs(title="PCA 'small multiples' plot", 
         #caption=paste0("produced on ", Sys.time())) + 
         caption="RNA-Seq: Diesel-Treated HepG2 Cells & Control Groups") +
    theme_bw() + 
    coord_flip()

# ==============================================================================

# Use dplyr 'verbs' to modify our dataframe ----
# use dplyr 'mutate' function to add new columns based on existing data
mydata.df <- log2.cpm.filtered.norm.df %>% 
    mutate(healthy.AVG = (`HepG2_control1` + `HepG2_control2` + `HepG2_control3`)/3, 
           disease.AVG = (`HepG2_treat1_Diesel` + `HepG2_treat2_Diesel` + `HepG2_treat3_Diesel`)/3, 
           LogFC = (disease.AVG - healthy.AVG)) %>% 
    mutate_if(is.numeric, round, 2)

#now look at this modified data table
mydata.df

# Use dplyr 'arrange' and 'select' to sort your dataframe based on any variable
mydata.sort <- mydata.df %>% 
    dplyr::arrange(desc(LogFC)) %>% 
    dplyr::select(geneID, LogFC)

# Use dplyr "filter" and "select" functions to pick out genes of interest 
mydata.filter <- mydata.df %>% 
    dplyr::filter(geneID=="Cyp2a4" | geneID=="Lcn2" | geneID=="Gsta1" | geneID=="Mt2" | geneID=="Tlr1"
                  | geneID=="Rad51b" | geneID=="Scd2" | geneID=="Ocstamp" | geneID=="Ly6d" ) %>% 
    dplyr::select(geneID, healthy.AVG, disease.AVG, LogFC) %>% 
    dplyr::arrange(desc(LogFC))

# you can also filter based on any regular expression
mydata.grep <- mydata.df %>% 
    dplyr::filter(grepl('Cyp|Rft', geneID)) %>% 
    dplyr::select(geneID, healthy.AVG, disease.AVG, LogFC) %>% 
    dplyr::arrange(desc(geneID))
