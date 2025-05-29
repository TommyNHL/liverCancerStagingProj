VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
#ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
using Random
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: GradientBoostingClassifier
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import tree: DecisionTreeClassifier
@sk_import metrics: recall_score
@sk_import neural_network: MLPClassifier
@sk_import svm: LinearSVC
@sk_import neighbors: KNeighborsClassifier
@sk_import inspection: permutation_importance
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

## import packages from Python ##
jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef
make_scorer = pyimport("sklearn.metrics").make_scorer
f1 = make_scorer(f1_score, average="macro")

## input training set ## 146649 x 16 df, 139465 x 16 df, 162094 x 16 df
train1011DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_train_raMSIn4nonInDI_norm.csv", DataFrame)
train2021DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df2021prog_train_raMSIn4nonInDI_norm.csv", DataFrame)
train3031DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df3031prog_train_raMSIn4nonInDI_norm.csv", DataFrame)
train3031DEFSDf[train3031DEFSDf.type .== 1, "type"] .= 2
#trainDEFSDf = trainDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
train1011DEFSDf[train1011DEFSDf.type .== 1, :]
train2021DEFSDf[train2021DEFSDf.type .== 1, :]
train3031DEFSDf[train3031DEFSDf.type .== 2, :]
    ## calculate weight ## 0: 59248, 52994, 80943, 1: 87401, 86471, 81151
    ## No risk, 59248+54994+80943 =193185; Risk, 87401+86471 =173872; HCC, 81151
trainDEFSDf = vcat(train1011DEFSDf, train2021DEFSDf, train3031DEFSDf)

    Yy_train = deepcopy(trainDEFSDf[:, end])  # 0.7734; 0.8593; 1.8410
    sampleW = []
    for w in Vector(Yy_train)
        if w == 0
            push!(sampleW, 0.7734)
        elseif w == 1
            push!(sampleW, 0.8593)
        elseif w == 2
            push!(sampleW, 1.8410)
        end
    end 

## input ext val set ## 44057 x 16 df, 52759 x 16 df, 49067 x 16 df
ext1011DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_ext_raMSIn4nonInDI_norm.csv", DataFrame)
ext2021DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df2021prog_ext_raMSIn4nonInDI_norm.csv", DataFrame)
ext3031DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df3031prog_ext_raMSIn4nonInDI_norm.csv", DataFrame)
ext3031DEFSDf[ext3031DEFSDf.type .== 1, "type"] .= 2
#extDEFSDf = extDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
ext1011DEFSDf[ext1011DEFSDf.type .== 1, :]
ext2021DEFSDf[ext2021DEFSDf.type .== 1, :]
ext3031DEFSDf[ext3031DEFSDf.type .== 2, :]
    ## calculate weight ## 0: 19448, 23194, 25471, 1: 24609, 29565, 23596
    ## No risk, 19448+23194+25471 =68113; Risk, 24609+29565 =54174; HCC, 23596
extDEFSDf = vcat(ext1011DEFSDf, ext2021DEFSDf, ext3031DEFSDf)

    Yy_ext = deepcopy(extDEFSDf[:, end])  # 0.7139; 0.8976; 2.0608
    sampleExtW = []
    for w in Vector(Yy_ext)
        if w == 0
            push!(sampleExtW, 0.7139)
        elseif w == 1
            push!(sampleExtW, 0.8976)
        elseif w == 2
            push!(sampleExtW, 2.0608)
        end
    end 

## reconstruct a whole set ## 190706 x 16 df, 192224 x 16 df, 211161 x 16 df
ingested1011DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_ingested_raMSIn4nonInDI_norm.csv", DataFrame)
ingested2021DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df2021prog_ingested_raMSIn4nonInDI_norm.csv", DataFrame)
ingested3031DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df3031prog_ingested_raMSIn4nonInDI_norm.csv", DataFrame)
ingested3031DEFSDf[ingested3031DEFSDf.type .== 1, "type"] .= 2
#ingestedDEFSDf = ingestedDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
ingested1011DEFSDf[ingested1011DEFSDf.type .== 1, :]
ingested2021DEFSDf[ingested2021DEFSDf.type .== 1, :]
ingested3031DEFSDf[ingested3031DEFSDf.type .== 2, :]
    ## calculate weight ## 0: 78696, 76188, 106414, 1: 112010, 116036, 104747
    ## No risk, 78696+76188+106414 =261298; Risk, 112010+116036 =228046; HCC, 104747
ingestedDEFSDf = vcat(ingested1011DEFSDf, ingested2021DEFSDf, ingested3031DEFSDf)

    Yy_ingested = deepcopy(ingestedDEFSDf[:, end])  # 0.7579; 0.8684, 1.8906
    sampleIngestedW = []
    for w in Vector(Yy_ingested)
        if w == 0
            push!(sampleIngestedW, 0.7579)
        elseif w == 1
            push!(sampleIngestedW, 0.8684)
        elseif w == 2
            push!(sampleIngestedW, 1.8906)
        end
    end 

## input FNA set ## 92506 x 16 df, 95231 x 16 df, 88701 x 16 df
fna1011DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_FNA_raMSIn4nonInDI_norm.csv", DataFrame)
fna2021DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df2021prog_FNA_raMSIn4nonInDI_norm.csv", DataFrame)
fna3031DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df3031prog_FNA_raMSIn4nonInDI_norm.csv", DataFrame)
fna3031DEFSDf[fna3031DEFSDf.type .== 1, "type"] .= 2
#fnaDEFSDf = fnaDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
fna1011DEFSDf[fna1011DEFSDf.type .== 1, :]
fna2021DEFSDf[fna2021DEFSDf.type .== 1, :]
fna3031DEFSDf[fna3031DEFSDf.type .== 2, :]
    ## calculate weight ##  0: 44636, 49500, 44540, 1: 47870, 45731, 44161
    ## No risk, 44636+49500+44540 =138676; Risk, 47870+45731 =93601; HCC, 44161
fnaDEFSDf = vcat(fna1011DEFSDf, fna2021DEFSDf, fna3031DEFSDf)

   Yy_FNA = deepcopy(fnaDEFSDf[:, end])  # 0.6645; 0.9845, 2.0866
    sampleFNAW = []
    for w in Vector(Yy_FNA)
        if w == 0
            push!(sampleFNAW, 0.6645)
        elseif w == 1
            push!(sampleFNAW, 0.9845)
        elseif w == 2
            push!(sampleFNAW, 2.0866)
        end
    end  

## input DirectIn set ## 2464 x 16 df, 4969 x 16 df, 5024 x 16 df
di1011DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)
di2021DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df2021prog_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)
di3031DEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df3031prog_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)
di3031DEFSDf[di3031DEFSDf.type .== 1, "type"] .= 2
#diDEFSDf = diDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
di1011DEFSDf[di1011DEFSDf.type .== 1, :]
di2021DEFSDf[di2021DEFSDf.type .== 1, :]
di3031DEFSDf[di3031DEFSDf.type .== 2, :]
    ## calculate weight ##  0: 1231, 2478, 2500, 1: 1233, 2491, 2524
    ## No risk, 1231+2478+2500 =138676; Risk, 1233+2491 =93601; HCC, 2524
diDEFSDf = vcat(di1011DEFSDf, di2021DEFSDf, di3031DEFSDf)

    Yy_DI = deepcopy(diDEFSDf[:, end])  # 0.6688; 1.1150, 1.6451
    sampleDiW = []
    for w in Vector(Yy_DI)
        if w == 0
            push!(sampleDiW, 0.6688)
        elseif w == 1
            push!(sampleDiW, 1.1150)
        elseif w == 2
            push!(sampleDiW, 1.6451)
        end
    end  

## input PMDirectIn set ## 3364 x 16 df
di1011PMDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\df1011prog_nonInPMDI_raMSIn4nonInDI_norm.csv", DataFrame)
#di1011PMDEFSDf = di1011PMDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
di1011PMDEFSDf[di1011PMDEFSDf.type .== 1, :]
    ## calculate weight ##  0: 1673, 1: 1691

    Yy_PMDI = deepcopy(di1011PMDEFSDf[:, end])  # 1.0054, 0.9947
    samplePMDiW = []
    for w in Vector(Yy_PMDI)
        if w == 0
            push!(samplePMDiW, 1.0054)
        elseif w == 1
            push!(samplePMDiW, 0.9947)
        end
    end  

## input Val DirectIn set ## 6057 x 16 df
di3031OldDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\dfOld3031prog_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)
di3031OldDEFSDf[di3031OldDEFSDf.type .== 1, "type"] .= 2
#di3031OldDEFSDf = di3031OldDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
di3031OldDEFSDf[di3031OldDEFSDf.type .== 2, :]
    ## calculate weight ##  0: 3027, 2: 3030

    Yy_3031OldDI = deepcopy(di3031OldDEFSDf[:, end])  # 1.0005, 0.9995
    sample3031OldDiW = []
    for w in Vector(Yy_3031OldDI)
        if w == 0
            push!(sample3031OldDiW, 1.0005)
        elseif w == 2
            push!(sample3031OldDiW, 0.9995)
        end
    end  

## define functions for performace evaluation ##
    # Average score
    function avgScore(arrAcc, cv)
        sumAcc = 0
        for acc in arrAcc
            sumAcc += acc
        end
        return sumAcc / cv
    end


# ==================================================================================================
## define a function for Random Forest ##
function optimRandomForestClass(inputDB_ingested, inputDB_FNA, inputDB_di, inputDB_PMdi, inputDB_OldDi)
    leaf_r = vcat(2, 8, 18)  # 5
    #leaf_r = vcat(2, 4, 8, 12, 18)  # 5
    #leaf_r = vcat(collect(2:1:8))  # 7
    #leaf_r = vcat(collect(18:1:28))  # 11
    depth_r = vcat(collect(2:4:10))  # 9
    #depth_r = vcat(collect(2:1:10))  # 9
    #depth_r = vcat(collect(6:1:14))  # 9
    #depth_r = vcat(collect(4:1:10))  # 7
    split_r = vcat(collect(2:4:10))  # 9
    #split_r = vcat(collect(2:1:10))  # 9
    #split_r = vcat(collect(10:5:20))  # 3
    tree_r = vcat(collect(50:100:250))  # 6
    #tree_r = vcat(collect(50:50:300))  # 6

    rs = 42
    z = zeros(1,47)
    itr = 1

    M_train = inputDB_ingested
    M_val = inputDB_FNA
    M_test = inputDB_di
    M_test2 = inputDB_PMdi
    M_ext = inputDB_OldDi

    for l in leaf_r
        for d in depth_r
            for s in split_r
                for t in tree_r
                    println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", s, ", tree=", t)
                    println("## loading in data ##")
                    Xx_train = deepcopy(M_train[:, 2:end-1])
                    Xx_val = deepcopy(M_val[:, 2:end-1])
                    Xx_test = deepcopy(M_test[:, 2:end-1])
                    Xx_test2 = deepcopy(M_test2[:, 2:end-1])
                    Xx_ext = deepcopy(M_ext[:, 2:end-1])
                    #
                    Yy_train = deepcopy(M_train[:, end])
                    Yy_val = deepcopy(M_val[:, end])
                    Yy_test = deepcopy(M_test[:, end])
                    Yy_test2 = deepcopy(M_test2[:, end])
                    Yy_ext = deepcopy(M_ext[:, end])
                    println("## Classification ##")
                    reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=s, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.7579, 1=>0.8684, 2=>1.8906))
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                    print(importances["importances_mean"])
                    if itr == 1
                        z[1,1] = l
                        z[1,2] = t
                        z[1,3] = d
                        z[1,4] = s
                        z[1,5] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                        z[1,6] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                        z[1,7] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                        z[1,8] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                        println("## CV ##")
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        z[1,9] = avgScore(f1_5_train, 5)
                        z[1,10] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                        z[1,11] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                        z[1,12] = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                        z[1,13] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                        z[1,14] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                        z[1,15] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                        z[1,16] = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                        z[1,17] = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                        z[1,18] = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                        z[1,19] = rs
                        z[1,20] = importances["importances_mean"][1]
                        z[1,21] = importances["importances_mean"][2]
                        z[1,22] = importances["importances_mean"][3]
                        z[1,23] = importances["importances_mean"][4]
                        z[1,24] = importances["importances_mean"][5]
                        z[1,25] = importances["importances_mean"][6]
                        z[1,26] = importances["importances_mean"][7]
                        z[1,27] = importances["importances_mean"][8]
                        z[1,28] = importances["importances_mean"][9]
                        z[1,29] = importances["importances_mean"][10]
                        z[1,30] = importances["importances_mean"][11]
                        z[1,31] = importances["importances_mean"][12]
                        z[1,32] = importances["importances_mean"][13]
                        z[1,33] = importances["importances_mean"][14]
                        z[1,34] = importances["importances_std"][1]
                        z[1,35] = importances["importances_std"][2]
                        z[1,36] = importances["importances_std"][3]
                        z[1,37] = importances["importances_std"][4]
                        z[1,38] = importances["importances_std"][5]
                        z[1,39] = importances["importances_std"][6]
                        z[1,40] = importances["importances_std"][7]
                        z[1,41] = importances["importances_std"][8]
                        z[1,42] = importances["importances_std"][9]
                        z[1,43] = importances["importances_std"][10]
                        z[1,44] = importances["importances_std"][11]
                        z[1,45] = importances["importances_std"][12]
                        z[1,46] = importances["importances_std"][13]
                        z[1,47] = importances["importances_std"][14]
                    else
                        itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                        jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                        ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                        jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                        println("## CV ##")
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        traincvtrain = avgScore(f1_5_train, 5) 
                        f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                        mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                        rec = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                        f1s2 = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                        mccs2 = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                        rec2 = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                        f1s3 = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                        mccs3 = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                        rec3 = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                        im1 = importances["importances_mean"][1]
                        im2 = importances["importances_mean"][2]
                        im3 = importances["importances_mean"][3]
                        im4 = importances["importances_mean"][4]
                        im5 = importances["importances_mean"][5]
                        im6 = importances["importances_mean"][6]
                        im7 = importances["importances_mean"][7]
                        im8 = importances["importances_mean"][8]
                        im9 = importances["importances_mean"][9]
                        im10 = importances["importances_mean"][10]
                        im11 = importances["importances_mean"][11]
                        im12 = importances["importances_mean"][12]
                        im13 = importances["importances_mean"][13]
                        im14 = importances["importances_mean"][14]
                        sd1 = importances["importances_std"][1]
                        sd2 = importances["importances_std"][2]
                        sd3 = importances["importances_std"][3]
                        sd4 = importances["importances_std"][4]
                        sd5 = importances["importances_std"][5]
                        sd6 = importances["importances_std"][6]
                        sd7 = importances["importances_std"][7]
                        sd8 = importances["importances_std"][8]
                        sd9 = importances["importances_std"][9]
                        sd10 = importances["importances_std"][10]
                        sd11 = importances["importances_std"][11]
                        sd12 = importances["importances_std"][12]
                        sd13 = importances["importances_std"][13]
                        sd14 = importances["importances_std"][14]
                        z = vcat(z, [l t d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 f1s3 mccs3 rec3 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14])
                        println(z[end, :])
                    end
                    println("End of ", itr, " iterations")
                    itr += 1
                end
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_fnaVal = z[:,7], mcc_fnaVal = z[:,8], f1_5Ftrain = z[:,9], f1_DIn = z[:,10], mcc_DIn = z[:,11], recall_DIn = z[:,12], f1_PMDIn = z[:,13], mcc_PMDIn = z[:,14], recall_PMDIn = z[:,15], f1_oldDIn = z[:,16], mcc_oldDIn = z[:,17], recall_oldDIn = z[:,18], state = z[:,19], im1 = z[:,20], im2 = z[:,21], im3 = z[:,22], im4 = z[:,23], im5 = z[:,24], im6 = z[:,25], im7 = z[:,26], im8 = z[:,27], im9 = z[:,28], im10 = z[:,29], im11 = z[:,30], im12 = z[:,31], im13 = z[:,32], im14 = z[:,33], sd1 = z[:,34], sd2 = z[:,35], sd3 = z[:,36], sd4 = z[:,37], sd5 = z[:,38], sd6 = z[:,39], sd7 = z[:,40], sd8 = z[:,41], sd9 = z[:,42], sd10 = z[:,43], sd11 = z[:,44], sd12 = z[:,45], sd13 = z[:,46], sd14 = z[:,47])
    z_df_sorted = sort(z_df, [:recall_PMDIn, :recall_oldDIn, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Random Forest ##
optiSearch_df = optimRandomForestClass(ingestedDEFSDf, fnaDEFSDf, diDEFSDf, di1011PMDEFSDf, di3031OldDEFSDf)

## save ##
savePath = "I:\\4_output_FIBproj\\4_3_Output_raMSIn\\modeling\\hyperparameterTuning_modelSelection_RF1.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Decision Tree ##
function optimDecisionTreeClass(inputDB_ingested, inputDB_FNA, inputDB_di, inputDB_PMdi, inputDB_OldDi)
    #leaf_r = vcat(2, 8, 18)  # 3
    #leaf_r = vcat(2, 4, 8, 10)  # 4
    leaf_r = vcat(collect(2:4:30), collect(35:15:80), 100, 200, 500)  # 8+4+3=15
    #leaf_r = vcat(collect(20:1:30))  # 11
    #depth_r = vcat(collect(2:4:10))  # 3
    #depth_r = vcat(collect(5:1:10))  # 6
    depth_r = vcat(collect(6:1:14))  # 9
    #depth_r = vcat(collect(4:1:8))  # 5
    #split_r = vcat(collect(2:4:10))  # 3
    #split_r = vcat(collect(2:1:10))  # 9
    split_r = vcat(collect(5:5:25))  # 5

    rs = 42
    z = zeros(1,46)
    itr = 1

    M_train = inputDB_ingested
    M_val = inputDB_FNA
    M_test = inputDB_di
    M_test2 = inputDB_PMdi
    M_ext = inputDB_OldDi

    for l in leaf_r
        for d in depth_r
            for s in split_r
                println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", s)
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, 2:end-1])
                Xx_val = deepcopy(M_val[:, 2:end-1])
                Xx_test = deepcopy(M_test[:, 2:end-1])
                Xx_test2 = deepcopy(M_test2[:, 2:end-1])
                Xx_ext = deepcopy(M_ext[:, 2:end-1])
                #
                Yy_train = deepcopy(M_train[:, end])
                Yy_val = deepcopy(M_val[:, end])
                Yy_test = deepcopy(M_test[:, end])
                Yy_test2 = deepcopy(M_test2[:, end])
                Yy_ext = deepcopy(M_ext[:, end])
                println("## Classification ##")
                reg = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, min_samples_split=s, random_state=rs, class_weight=Dict(0=>0.7579, 1=>0.8684, 2=>1.8906))
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = l
                    z[1,2] = d
                    z[1,3] = s
                    z[1,4] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                    z[1,5] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                    z[1,6] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                    println("## CV ##")
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    z[1,8] = avgScore(f1_5_train, 5)
                    z[1,9] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                    z[1,11] = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                    z[1,12] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                    z[1,13] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                    z[1,14] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                    z[1,15] = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                    z[1,16] = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                    z[1,17] = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                    z[1,18] = rs
                    z[1,19] = importances["importances_mean"][1]
                    z[1,20] = importances["importances_mean"][2]
                    z[1,21] = importances["importances_mean"][3]
                    z[1,22] = importances["importances_mean"][4]
                    z[1,23] = importances["importances_mean"][5]
                    z[1,24] = importances["importances_mean"][6]
                    z[1,25] = importances["importances_mean"][7]
                    z[1,26] = importances["importances_mean"][8]
                    z[1,27] = importances["importances_mean"][9]
                    z[1,28] = importances["importances_mean"][10]
                    z[1,29] = importances["importances_mean"][11]
                    z[1,30] = importances["importances_mean"][12]
                    z[1,31] = importances["importances_mean"][13]
                    z[1,32] = importances["importances_mean"][14]
                    z[1,33] = importances["importances_std"][1]
                    z[1,34] = importances["importances_std"][2]
                    z[1,35] = importances["importances_std"][3]
                    z[1,36] = importances["importances_std"][4]
                    z[1,37] = importances["importances_std"][5]
                    z[1,38] = importances["importances_std"][6]
                    z[1,39] = importances["importances_std"][7]
                    z[1,40] = importances["importances_std"][8]
                    z[1,41] = importances["importances_std"][9]
                    z[1,42] = importances["importances_std"][10]
                    z[1,43] = importances["importances_std"][11]
                    z[1,44] = importances["importances_std"][12]
                    z[1,45] = importances["importances_std"][13]
                    z[1,46] = importances["importances_std"][14]
                else
                    itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                    jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                    ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                    jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                    println("## CV ##")
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    traincvtrain = avgScore(f1_5_train, 5) 
                    f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                    mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                    rec = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                    f1s2 = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                    mccs2 = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                    rec2 = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                    f1s3 = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                    mccs3 = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                    rec3 = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    im9 = importances["importances_mean"][9]
                    im10 = importances["importances_mean"][10]
                    im11 = importances["importances_mean"][11]
                    im12 = importances["importances_mean"][12]
                    im13 = importances["importances_mean"][13]
                    im14 = importances["importances_mean"][14]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    sd9 = importances["importances_std"][9]
                    sd10 = importances["importances_std"][10]
                    sd11 = importances["importances_std"][11]
                    sd12 = importances["importances_std"][12]
                    sd13 = importances["importances_std"][13]
                    sd14 = importances["importances_std"][14]
                    z = vcat(z, [l d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 f1s3 mccs3 rec3 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], depth = z[:,2], minSplit = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_fnaVal = z[:,6], mcc_fnaVal = z[:,7], f1_5Ftrain = z[:,8], f1_DIn = z[:,9], mcc_DIn = z[:,10], recall_DIn = z[:,11], f1_PMDIn = z[:,12], mcc_PMDIn = z[:,13], recall_PMDIn = z[:,14], f1_oldDIn = z[:,15], mcc_oldDIn = z[:,16], recall_oldDIn = z[:,17], state = z[:,18], im1 = z[:,19], im2 = z[:,20], im3 = z[:,21], im4 = z[:,22], im5 = z[:,23], im6 = z[:,24], im7 = z[:,25], im8 = z[:,26], im9 = z[:,27], im10 = z[:,28], im11 = z[:,29], im12 = z[:,30], im13 = z[:,31], im14 = z[:,32], sd1 = z[:,33], sd2 = z[:,34], sd3 = z[:,35], sd4 = z[:,36], sd5 = z[:,37], sd6 = z[:,38], sd7 = z[:,39], sd8 = z[:,40], sd9 = z[:,41], sd10 = z[:,42], sd11 = z[:,43], sd12 = z[:,44], sd13 = z[:,45], sd14 = z[:,46])
    z_df_sorted = sort(z_df, [:recall_PMDIn, :recall_oldDIn, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Decision Tree ##
optiSearch_df = optimDecisionTreeClass(ingestedDEFSDf, fnaDEFSDf, diDEFSDf, di1011PMDEFSDf, di3031OldDEFSDf)

## save ##
savePath = "C:\\Users\\T1208\\PyLAB\\4_output_FIBproj\\4_3_Output_raMSIn\\modeling\\hyperparameterTuning_modelSelection_DT3.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Gradient Boost ##
function optimGradientBoostClass(inputDB_ingested, inputDB_FNA, inputDB_di, inputDB_PMdi, inputDB_OldDi)
    lr_r = vcat(0.1, 0.5, 1, 5)  # 4
    #lr_r = vcat(0.5, collect(1:2:9))  # 6
    #lr_r = vcat(collect(2:0.5:9))  # 15
    #lr_r = vcat(collect(3.5:0.1:4.5))  # 11
    leaf_r = vcat(2, 8, 18)  # 3
    #leaf_r = vcat(collect(2:4:10))  # 3
    #leaf_r = vcat(collect(2:1:8))  # 7
    depth_r = vcat(collect(4:2:10))  # 4
    #depth_r = vcat(collect(4:1:8))  # 5
    #depth_r = vcat(collect(5:1:10))  # 6
    split_r = vcat(collect(2:4:10))  # 3
    #split_r = vcat(collect(10:10:20))  # 2
    #split_r = vcat(collect(15:15:30))  # 2
    #split_r = vcat(30)  # 1
    #split_r = vcat(10, 30, 50)  # 3
    tree_r = vcat(25, 50)  # 2
    #tree_r = vcat(collect(50:100:250))  # 3
    #tree_r = vcat(collect(25:25:75))  # 3
    #tree_r = vcat(50)  # 1
    
    rs = 42
    z = zeros(1,48)
    itr = 1

    M_train = inputDB_ingested
    M_val = inputDB_FNA
    M_test = inputDB_di
    M_test2 = inputDB_PMdi
    M_ext = inputDB_OldDi

    for lr in lr_r
        for l in leaf_r
            for d in depth_r
                for s in split_r
                    for t in tree_r
                        println("itr=", itr, ", lr=", lr, ", leaf=", l, ", depth=", d, ", minSsplit=", s, ", tree=", t)
                        println("## loading in data ##")
                        Xx_train = deepcopy(M_train[:, 2:end-1])
                        Xx_val = deepcopy(M_val[:, 2:end-1])
                        Xx_test = deepcopy(M_test[:, 2:end-1])
                        Xx_test2 = deepcopy(M_test2[:, 2:end-1])
                        Xx_ext = deepcopy(M_ext[:, 2:end-1])
                        #
                        Yy_train = deepcopy(M_train[:, end])
                        Yy_val = deepcopy(M_val[:, end])
                        Yy_test = deepcopy(M_test[:, end])
                        Yy_test2 = deepcopy(M_test2[:, end])
                        Yy_ext = deepcopy(M_ext[:, end])
                        println("## Classification ##")
                        reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=s, random_state=rs, n_iter_no_change=5, sample_weight=Dict(0=>0.7579, 1=>0.8684, 2=>1.8906))
                        println("## fit ##")
                        fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                        importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42, n_jobs=-1)
                        print(importances["importances_mean"])
                        if itr == 1
                            z[1,1] = lr
                            z[1,2] = l
                            z[1,3] = t
                            z[1,4] = d
                            z[1,5] = s
                            z[1,6] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                            z[1,7] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampsampleIngestedWleW)
                            z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                            z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                            println("## CV ##")
                            f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                            z[1,10] = avgScore(f1_5_train, 5)
                            z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                            z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                            z[1,13] = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                            z[1,14] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                            z[1,15] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                            z[1,16] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                            z[1,17] = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                            z[1,18] = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                            z[1,19] = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                            z[1,20] = rs
                            z[1,21] = importances["importances_mean"][1]
                            z[1,22] = importances["importances_mean"][2]
                            z[1,23] = importances["importances_mean"][3]
                            z[1,24] = importances["importances_mean"][4]
                            z[1,25] = importances["importances_mean"][5]
                            z[1,26] = importances["importances_mean"][6]
                            z[1,27] = importances["importances_mean"][7]
                            z[1,28] = importances["importances_mean"][8]
                            z[1,29] = importances["importances_mean"][9]
                            z[1,30] = importances["importances_mean"][10]
                            z[1,31] = importances["importances_mean"][11]
                            z[1,32] = importances["importances_mean"][12]
                            z[1,33] = importances["importances_mean"][13]
                            z[1,34] = importances["importances_mean"][14]
                            z[1,35] = importances["importances_std"][1]
                            z[1,36] = importances["importances_std"][2]
                            z[1,37] = importances["importances_std"][3]
                            z[1,38] = importances["importances_std"][4]
                            z[1,39] = importances["importances_std"][5]
                            z[1,40] = importances["importances_std"][6]
                            z[1,41] = importances["importances_std"][7]
                            z[1,42] = importances["importances_std"][8]
                            z[1,43] = importances["importances_std"][9]
                            z[1,44] = importances["importances_std"][10]
                            z[1,45] = importances["importances_std"][11]
                            z[1,46] = importances["importances_std"][12]
                            z[1,47] = importances["importances_std"][13]
                            z[1,48] = importances["importances_std"][14]
                        else
                            itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                            jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                            ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                            jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                            println("## CV ##")
                            f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                            traincvtrain = avgScore(f1_5_train, 5) 
                            f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                            mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=Xx_test)
                            rec = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                            f1s2 = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                            mccs2 = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=sampleDiW)
                            rec2 = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                            f1s3 = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                            mccs3 = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                            rec3 = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                            im1 = importances["importances_mean"][1]
                            im2 = importances["importances_mean"][2]
                            im3 = importances["importances_mean"][3]
                            im4 = importances["importances_mean"][4]
                            im5 = importances["importances_mean"][5]
                            im6 = importances["importances_mean"][6]
                            im7 = importances["importances_mean"][7]
                            im8 = importances["importances_mean"][8]
                            im9 = importances["importances_mean"][9]
                            im10 = importances["importances_mean"][10]
                            im11 = importances["importances_mean"][11]
                            im12 = importances["importances_mean"][12]
                            im13 = importances["importances_mean"][13]
                            im14 = importances["importances_mean"][14]
                            sd1 = importances["importances_std"][1]
                            sd2 = importances["importances_std"][2]
                            sd3 = importances["importances_std"][3]
                            sd4 = importances["importances_std"][4]
                            sd5 = importances["importances_std"][5]
                            sd6 = importances["importances_std"][6]
                            sd7 = importances["importances_std"][7]
                            sd8 = importances["importances_std"][8]
                            sd9 = importances["importances_std"][9]
                            sd10 = importances["importances_std"][10]
                            sd11 = importances["importances_std"][11]
                            sd12 = importances["importances_std"][12]
                            sd13 = importances["importances_std"][13]
                            sd14 = importances["importances_std"][14]
                            z = vcat(z, [lr l t d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 f1s3 mccs3 rec3 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14])
                            println(z[end, :])
                        end
                        println("End of ", itr, " iterations")
                        itr += 1
                    end
                end
            end
        end
    end
    z_df = DataFrame(lr = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_fnaVal = z[:,8], mcc_fnaVal = z[:,9], f1_5Ftrain = z[:,10], f1_DIn = z[:,11], mcc_DIn = z[:,12], recall_DIn = z[:,13], f1_PMDIn = z[:,14], mcc_PMDIn = z[:,15], recall_PMDIn = z[:,16],  f1_oldDIn = z[:,17], mcc_oldDIn = z[:,18], recall_oldDIn = z[:,19], state = z[:,20], im1 = z[:,21], im2 = z[:,22], im3 = z[:,23], im4 = z[:,24], im5 = z[:,25], im6 = z[:,26], im7 = z[:,27], im8 = z[:,28], im9 = z[:,29], im10 = z[:,30], im11 = z[:,31], im12 = z[:,32], im13 = z[:,33], im14 = z[:,34], sd1 = z[:,35], sd2 = z[:,36], sd3 = z[:,37], sd4 = z[:,38], sd5 = z[:,39], sd6 = z[:,40], sd7 = z[:,41], sd8 = z[:,42], sd9 = z[:,43], sd10 = z[:,44], sd11 = z[:,45], sd12 = z[:,46], sd13 = z[:,47], sd14 = z[:,48])
    z_df_sorted = sort(z_df, [:recall_PMDIn, :recall_oldDIn, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Gradient Boost ##
optiSearch_df = optimGradientBoostClass(ingestedDEFSDf, fnaDEFSDf, diDEFSDf, di1011PMDEFSDf, di3031OldDEFSDf)

## save ##
savePath = "C:\\Users\\T1208\\PyLAB\\4_output_FIBproj\\4_3_Output_raMSIn\\modeling\\hyperparameterTuning_modelSelection_GBM1.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for k-Nearest Neighbors ##
function optimKNN(inputDB_ingested, inputDB_FNA, inputDB_di, inputDB_PMdi, inputDB_OldDi)
    k_n_r = vcat(collect(10:10:50), collect(75:25:175), collect(199:50:399))  # 5+5+5=15
    #k_n_r = vcat(collect(377:2:399))  # 11
    #k_n_r = vcat(collect(377:2:399))  # 11
    #k_n_r = vcat(collect(377:2:399))  # 11
    #k_n_r = vcat(collect(377:2:399))  # 11
    #k_n_r = vcat(collect(377:2:399))  # 11
    leaf_r = vcat(2, 5, 10, 25, collect(50:50:400))  # 12
    w_r = ["uniform", "distance"]
    met_r = ["minkowski", "euclidean", "manhattan"]
    p_r = vcat(1, 2)

    rs = 42
    z = zeros(1,48)
    w = 1
    met = 1
    p = 2
    #leaf = 300
    itr = 1

    M_train = inputDB_ingested
    N_train = vcat(inputDB_ingested, inputDB_ingested[inputDB_ingested.LABEL .== 2, :])
    M_val = inputDB_FNA
    M_test = inputDB_di
    M_test2 = inputDB_PMdi
    M_ext = inputDB_OldDi

    for k_n in k_n_r
        for leaf in leaf_r
            #for w in 1:2
                #for met in vcat(1,3)
                    #for p in p_r
            println("k_n=", k_n, ", leaf=", leaf, ", w=", w_r[w], ", met=", met_r[met], ", p=", p)
            println("## loading in data ##")
            Xx_train = deepcopy(M_train[:, 2:end-1])
            nn_train = deepcopy(N_train[:, 2:end-1])
            Xx_val = deepcopy(M_val[:, 2:end-1])
            Xx_test = deepcopy(M_test[:, 2:end-1])
            Xx_test2 = deepcopy(M_test2[:, 2:end-1])
            Xx_ext = deepcopy(M_ext[:, 2:end-1])
            #
            Yy_train = deepcopy(M_train[:, end])
            mm_train = deepcopy(N_train[:, end])
            Yy_val = deepcopy(M_val[:, end])
            Yy_test = deepcopy(M_test[:, end])
            Yy_test2 = deepcopy(M_test2[:, end])
            Yy_ext = deepcopy(M_ext[:, end])
            println("## Classification ##")
            reg = KNeighborsClassifier(n_neighbors=k_n, weights=w_r[w], leaf_size=leaf, p=p, metric=met_r[met])
            println("## fit ##")
            fit!(reg, Matrix(nn_train), Vector(mm_train))
            importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
            print(importances["importances_mean"])
            if itr == 1
                z[1,1] = k_n
                z[1,2] = leaf
                z[1,3] = w
                z[1,4] = met
                z[1,5] = p
                z[1,6] = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                z[1,7] = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                println("## CV ##")
                f1_5_train = cross_val_score(reg, Matrix(nn_train), Vector(mm_train); cv = 5, scoring=f1)
                z[1,10] = avgScore(f1_5_train, 5)
                z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                z[1,13] = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                z[1,14] = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                z[1,15] = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                z[1,16] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                z[1,17] = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                z[1,18] = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                z[1,19] = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                z[1,20] = rs
                z[1,21] = importances["importances_mean"][1]
                z[1,22] = importances["importances_mean"][2]
                z[1,23] = importances["importances_mean"][3]
                z[1,24] = importances["importances_mean"][4]
                z[1,25] = importances["importances_mean"][5]
                z[1,26] = importances["importances_mean"][6]
                z[1,27] = importances["importances_mean"][7]
                z[1,28] = importances["importances_mean"][8]
                z[1,29] = importances["importances_mean"][9]
                z[1,30] = importances["importances_mean"][10]
                z[1,31] = importances["importances_mean"][11]
                z[1,32] = importances["importances_mean"][12]
                z[1,33] = importances["importances_mean"][13]
                z[1,34] = importances["importances_mean"][14]
                z[1,35] = importances["importances_std"][1]
                z[1,36] = importances["importances_std"][2]
                z[1,37] = importances["importances_std"][3]
                z[1,38] = importances["importances_std"][4]
                z[1,39] = importances["importances_std"][5]
                z[1,40] = importances["importances_std"][6]
                z[1,41] = importances["importances_std"][7]
                z[1,42] = importances["importances_std"][8]
                z[1,43] = importances["importances_std"][9]
                z[1,44] = importances["importances_std"][10]
                z[1,45] = importances["importances_std"][11]
                z[1,46] = importances["importances_std"][12]
                z[1,47] = importances["importances_std"][13]
                z[1,48] = importances["importances_std"][14]
            else
                itrain = f1_score(Vector(Yy_train), predict(reg, Matrix(Xx_train)), average="weighted", sample_weight=sampleIngestedW)
                jtrain = matthews_corrcoef(Vector(Yy_train), predict(reg, Matrix(Xx_train)), sample_weight=sampleIngestedW)
                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), average="weighted", sample_weight=sampleFNAW)
                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampleFNAW)
                println("## CV ##")
                f1_5_train = cross_val_score(reg, Matrix(nn_train), Vector(mm_train); cv = 5, scoring=f1)
                traincvtrain = avgScore(f1_5_train, 5) 
                f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=sampleDiW)
                rec = recall_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), average="weighted", sample_weight=sampleDiW)
                f1s2 = f1_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                mccs2 = matthews_corrcoef(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), sample_weight=samplePMDiW)
                rec2 = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)), average="weighted", sample_weight=samplePMDiW)
                f1s3 = f1_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                mccs3 = matthews_corrcoef(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), sample_weight=sample3031OldDiW)
                rec3 = recall_score(Vector(Yy_ext), predict(reg, Matrix(Xx_ext)), average="weighted", sample_weight=sample3031OldDiW)
                im1 = importances["importances_mean"][1]
                im2 = importances["importances_mean"][2]
                im3 = importances["importances_mean"][3]
                im4 = importances["importances_mean"][4]
                im5 = importances["importances_mean"][5]
                im6 = importances["importances_mean"][6]
                im7 = importances["importances_mean"][7]
                im8 = importances["importances_mean"][8]
                im9 = importances["importances_mean"][9]
                im10 = importances["importances_mean"][10]
                im11 = importances["importances_mean"][11]
                im12 = importances["importances_mean"][12]
                im13 = importances["importances_mean"][13]
                im14 = importances["importances_mean"][14]
                sd1 = importances["importances_std"][1]
                sd2 = importances["importances_std"][2]
                sd3 = importances["importances_std"][3]
                sd4 = importances["importances_std"][4]
                sd5 = importances["importances_std"][5]
                sd6 = importances["importances_std"][6]
                sd7 = importances["importances_std"][7]
                sd8 = importances["importances_std"][8]
                sd9 = importances["importances_std"][9]
                sd10 = importances["importances_std"][10]
                sd11 = importances["importances_std"][11]
                sd12 = importances["importances_std"][12]
                sd13 = importances["importances_std"][13]
                sd14 = importances["importances_std"][14]
                z = vcat(z, [k_n leaf w met p itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 f1s3 mccs3 rec3 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14])
                println(z[end, :])
            end
            println("End of ", itr, " iterations")
            itr += 1
        end
                #end
            #end
        #end
    end
    z_df = DataFrame(k_n = z[:,1], leaf = z[:,2], weight = z[:,3], met = z[:,4], p = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_fnaVal = z[:,8], mcc_fnaVal = z[:,9], f1_5Ftrain = z[:,10], f1_DIn = z[:,11], mcc_DIn = z[:,12], recall_DIn = z[:,13], f1_PMDIn = z[:,14], mcc_PMDIn = z[:,15], recall_PMDIn = z[:,16], f1_oldDIn = z[:,17], mcc_oldDIn = z[:,18], recall_oldDIn = z[:,19], state = z[:,20], im1 = z[:,21], im2 = z[:,22], im3 = z[:,23], im4 = z[:,24], im5 = z[:,25], im6 = z[:,26], im7 = z[:,27], im8 = z[:,28], im9 = z[:,29], im10 = z[:,30], im11 = z[:,31], im12 = z[:,32], im13 = z[:,33], im14 = z[:,34], sd1 = z[:,35], sd2 = z[:,36], sd3 = z[:,37], sd4 = z[:,38], sd5 = z[:,39], sd6 = z[:,40], sd7 = z[:,41], sd8 = z[:,42], sd9 = z[:,43], sd10 = z[:,44], sd11 = z[:,45], sd12 = z[:,46], sd13 = z[:,47], sd14 = z[:,48])
    z_df_sorted = sort(z_df, [:recall_PMDIn, :recall_oldDIn, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call k-Nearest Neighbors ##
optiSearch_df = optimKNN(ingestedDEFSDf, fnaDEFSDf, diDEFSDf, di1011PMDEFSDf, di3031OldDEFSDf)

## save ##
savePath = "C:\\Users\\T1208\\PyLAB\\4_output_FIBproj\\4_3_Output_raMSIn\\modeling\\hyperparameterTuning_modelSelection_KNN1.csv"
CSV.write(savePath, optiSearch_df)

