path <- getwd()
setwd(path)
print(path)
dataSource = "D:/PM/pos/MS1/"
setwd(dataSource)
#load("./HKBUSKLP/.data/posDemoData/Result/object")  # on VS Code
#load("./.data/posDemoData/Result/object")  # on RStudio
load("./Result/object")  # on RStudio

library(metid)

data("snyder_database_rplc0.0.3")
snyder_database_rplc0.0.3

# MS1 identification
posObject <- 
    annotate_metabolites_mass_dataset(object = object, 
                                      ms1.match.ppm = 10, 
                                      rt.match.tol = 10000000,  # dysfunction by assigning a value of 10,000,000
                                      polarity = "positive",
                                      database = snyder_database_rplc0.0.3)

anotationTable <- posObject@annotation_table

#write.csv(anotation_table, "./.data/posDemoData/Result/L001_01_preprocessingResult.csv")
write.csv(anotationTable, "./Result/posT3AnotationTable.csv")
