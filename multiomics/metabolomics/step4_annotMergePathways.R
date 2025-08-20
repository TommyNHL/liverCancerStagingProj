library(masscleaner)
library(tidyverse)

dataSource1 = "D:/PM/pos/MS1/"
setwd(dataSource1)

load("./object_final4Merge_T3Pos")
bPos = object_final4Merge

##---------------------add NEG--------------------------------------------------

dataSource2 = "D:/PM/neg/MS1/"
setwd(dataSource2)

load("./object_final4Merge_T3Neg") 
bNeg = object_final4Merge

##---------------------merge Pos with Neg---------------------------------------
dataSource = "D:/PM/pos/MS1/"
setwd(dataSource)

finalT3Object <- merge_mass_dataset(x = bPos, 
                                  y = bNeg, 
                                  sample_direction = "inner", 
                                  variable_direction = "full", 
                                  sample_by = "sample_id", 
                                  variable_by = c("variable_id", "mz", "rt"))

finalT3Object

dir.create(path = "statistical_analysis", showWarnings = FALSE)

report_parameters(object = finalT3Object, path = "statistical_analysis/")


# finalObject %>%
#   activate_mass_dataset(what = "sample_info") %>%
#   dplyr::count(group)

# qc_id = finalObject %>% 
#   activate_mass_dataset(what = "sample_info") %>%
#   metid::filter(class == "QC") %>%
#   pull(sample_id)

control_id = finalT3Object %>%
  activate_mass_dataset(what = "sample_info") %>%
  metid::filter(group == "Control") %>%
  pull(sample_id)

case_id = finalT3Object %>%
  activate_mass_dataset(what = "sample_info") %>%
  metid::filter(group == "Case") %>%
  pull(sample_id)

# case2_id = finalObject %>%
#   activate_mass_dataset(what = "sample_info") %>%
#   filter(group == "case2") %>%
#   pull(sample_id)

# finalObject = finalObject %>%
#   mutate_variable_na_freq(according_to_samples = qc_id) %>%
#   mutate_variable_na_freq(according_to_samples = control_id) %>%
#   mutate_variable_na_freq(according_to_samples = case_id)

# head(extract_variable_info(finalObject))

library(tidymass)
library(tidyverse)

finalT3Object_0010 <- mutate_fc(object = finalT3Object, 
                             control_sample_id = control_id, 
                             case_sample_id = case_id, 
                             mean_median = "mean")

finalT3Object_0010 <- mutate_p_value(object = finalT3Object_0010,
                                  control_sample_id = control_id,
                                  case_sample_id = case_id,
                                  method = "t.test", 
                                  p_adjust_methods = "none")

volcano_plot(object = finalT3Object_0010, 
             add_text = TRUE,
             text_from = "Compound.name", 
             point_size_scale = "p_value") + 
  scale_size_continuous(range = c(0.5, 5))

differential_metabolites <- extract_variable_info(object = finalT3Object_0010) %>% 
  metid::filter(fc > 2 | fc < 0.5) %>% 
  metid::filter(p_value_adjust < 0.05)

readr::write_csv(differential_metabolites, 
                 file = "../../PM_metabolites.csv")

save(finalT3Object_0010, file = "../../finalT3Object_PM")
