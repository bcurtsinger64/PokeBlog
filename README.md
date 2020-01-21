Pokemon Anomaly Detection & Clustering
================
Bryce Curtsinger
November 20, 2018

Pokemon Anomaly Detection
-------------------------

When I was a child, I looked forward to spending my Saturday mornings watching cartoons, as millions of other kids did. Of all the shows, Pokemon was always my favorite. In 2001, my dreams of becoming a Pokemon master came true (kinda) when my mother bought me the newest game, Pokemon Crystal, for the Game Boy Color. This sparked my love of Pokemon games that followed me into my teenage years and adulthood. I fell out for a while after Fire Red/ Leaf Green, then picked up the series again with Pokemon X and Y. This time, I became aware of the competitive scene of Pokemon battling, and fell in love all over again.

In this post, I will use data of 702 Pokemon from generations 1-6 to identify anomalies and outliers in terms of base statistics, then attempt to predict outliers using a binary classification model. For those unfamiliar, in the video games each pokemon has a base value for each of 6 statistics: HP, Attack, Defense, Special Attack, Special Defense, & Speed. While different instances of the same pokemon can have completely different values in these statistics (for a number of reasons), the base values provide a reference point that allows pokemon to be compared across species.

The data used for this analysis comes from a list of data on all pokemon maintained on kaggle (<https://www.kaggle.com/rounakbanik/pokemon>).

Finally, a brief description of packages used is provided below.

tidyverse - provides a multitude of data manipulation functions, among other things, to augment base R data manipulation.

ggplot2 - comprehensive visualization tool

Rlof - used for anomaly detection with lof function

randomForest - used to create a random forest model to predict outliers

pROC - calculate ROC & AUC

caret - confusion matrix & performance for model

    ## Loading required package: pacman

``` r
library(tidyverse)
library(ggplot2)
library(Rlof)
library(randomForest)
library(pROC)
library(caret)

#Read in Data
basedat <- read_csv('pokemon.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double(),
    ##   abilities = col_character(),
    ##   attack = col_integer(),
    ##   base_egg_steps = col_integer(),
    ##   base_happiness = col_integer(),
    ##   base_total = col_integer(),
    ##   capture_rate = col_character(),
    ##   classfication = col_character(),
    ##   defense = col_integer(),
    ##   experience_growth = col_integer(),
    ##   hp = col_integer(),
    ##   japanese_name = col_character(),
    ##   name = col_character(),
    ##   pokedex_number = col_integer(),
    ##   sp_attack = col_integer(),
    ##   sp_defense = col_integer(),
    ##   speed = col_integer(),
    ##   type1 = col_character(),
    ##   type2 = col_character(),
    ##   generation = col_integer(),
    ##   is_legendary = col_integer()
    ## )

    ## See spec(...) for full column specifications.

``` r
pokemon <- basedat %>% 
  filter(generation != 7) %>% 
  select(-c(1:19), -japanese_name, -experience_growth, -base_happiness, -base_egg_steps, -base_total, -capture_rate, - classfication, -percentage_male) %>% #get rid of these variables
  select(pokedex_number, name, type1, type2, height_m, weight_kg,
          hp, attack, defense, sp_attack, sp_defense, speed, generation, is_legendary) #reorder variables

pokemon <- pokemon %>% #Change column types
  mutate(type1 = as.factor(type1), type2 = as.factor(type2), total = hp + attack + defense + sp_attack + sp_defense + speed, type2 = as.character(type2)) %>% 
  replace_na(replace = list('type2' = 'none')) %>%
  mutate(type2 = as.factor(type2)) %>%
  filter(!is.na(height_m))
#Inspect Data
head(pokemon)
```

    ## # A tibble: 6 x 15
    ##   pokedex_number name  type1 type2 height_m weight_kg    hp attack defense
    ##            <int> <chr> <fct> <fct>    <dbl>     <dbl> <int>  <int>   <int>
    ## 1              1 Bulb~ grass pois~    0.700      6.90    45     49      49
    ## 2              2 Ivys~ grass pois~    1.00      13.0     60     62      63
    ## 3              3 Venu~ grass pois~    2.00     100.      80    100     123
    ## 4              4 Char~ fire  none     0.600      8.50    39     52      43
    ## 5              5 Char~ fire  none     1.10      19.0     58     64      58
    ## 6              6 Char~ fire  flyi~    1.70      90.5     78    104      78
    ## # ... with 6 more variables: sp_attack <int>, sp_defense <int>,
    ## #   speed <int>, generation <int>, is_legendary <int>, total <int>

``` r
summary(pokemon)
```

    ##  pokedex_number      name               type1          type2    
    ##  Min.   :  1.0   Length:702         water  :105   none    :355  
    ##  1st Qu.:194.2   Class :character   normal : 89   flying  : 87  
    ##  Median :369.5   Mode  :character   grass  : 65   poison  : 31  
    ##  Mean   :368.3                      bug    : 63   ground  : 27  
    ##  3rd Qu.:544.8                      psychic: 46   psychic : 26  
    ##  Max.   :721.0                      fire   : 45   fighting: 19  
    ##                                     (Other):289   (Other) :157  
    ##     height_m       weight_kg             hp             attack      
    ##  Min.   : 0.10   Min.   :  0.100   Min.   :  1.00   Min.   :  5.00  
    ##  1st Qu.: 0.60   1st Qu.:  9.425   1st Qu.: 50.00   1st Qu.: 53.00  
    ##  Median : 1.00   Median : 28.000   Median : 65.00   Median : 73.50  
    ##  Mean   : 1.15   Mean   : 57.223   Mean   : 69.01   Mean   : 76.96  
    ##  3rd Qu.: 1.50   3rd Qu.: 61.375   3rd Qu.: 80.00   3rd Qu.: 96.50  
    ##  Max.   :14.50   Max.   :950.000   Max.   :255.00   Max.   :185.00  
    ##                                                                     
    ##     defense         sp_attack        sp_defense         speed       
    ##  Min.   :  5.00   Min.   : 10.00   Min.   : 20.00   Min.   :  5.00  
    ##  1st Qu.: 50.00   1st Qu.: 45.25   1st Qu.: 50.00   1st Qu.: 45.00  
    ##  Median : 67.50   Median : 65.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 72.43   Mean   : 71.20   Mean   : 70.50   Mean   : 66.56  
    ##  3rd Qu.: 90.00   3rd Qu.: 90.00   3rd Qu.: 86.75   3rd Qu.: 86.75  
    ##  Max.   :230.00   Max.   :194.00   Max.   :230.00   Max.   :180.00  
    ##                                                                     
    ##    generation     is_legendary         total      
    ##  Min.   :1.000   Min.   :0.00000   Min.   :180.0  
    ##  1st Qu.:2.000   1st Qu.:0.00000   1st Qu.:320.8  
    ##  Median :3.000   Median :0.00000   Median :430.0  
    ##  Mean   :3.379   Mean   :0.07407   Mean   :426.6  
    ##  3rd Qu.:5.000   3rd Qu.:0.00000   3rd Qu.:505.0  
    ##  Max.   :6.000   Max.   :1.00000   Max.   :780.0  
    ## 

Outlier Identification
----------------------

Now that we've joined our datasets, let's begin with outlier detection. In this case, we will focus our analysis on the primary statistics of each pokemon: HP, attack, defense, special attack, special defense, and speed. Anyone familiar with competitive pokemon will immediately think of several pokemon that are outliers in terms of these statistics, so let's see if the results are what we would expect. The outlier detection technique used is LOF (local outlier factor), which identifies density-based outliers using a nearest-neighbors approach. Here, we considered 5, 6, 7, 8, 9, and 10 neighbors. Let's plot the density of the outliers using k = 5 for now:

![](Pokemon_Blog_files/figure-markdown_github/Outlier%20Density-1.png)

This plot indicates the presence of outliers because of the right-skewed distribution. Now we have to narrow down how many neighbors we want to use. We will use sort of a hackjob approach for this: find the top 20 outliers for each number of neighbors, find how much each iteration of k overlaps with the others, and select the one that has the most overlap. Using this approach, we find that 7 neighbors has the most overlap with each other iteration. Now, let's examine the top 20 outliers:

    ##           k5        k6        k7        k8        k9       k10
    ## 1   Beedrill  Beedrill  Beedrill  Beedrill  Beedrill  Beedrill
    ## 2    Slowbro      Abra      Abra      Abra      Abra      Abra
    ## 3   Cloyster   Slowbro   Slowbro  Alakazam    Gastly    Gastly
    ## 4     Gastly    Gastly    Gastly   Slowbro      Onix      Onix
    ## 5    Chansey   Chansey      Onix    Gastly   Chansey   Chansey
    ## 6     Horsea    Horsea   Chansey      Onix    Horsea    Horsea
    ## 7     Staryu  Magikarp    Horsea   Chansey  Magikarp  Magikarp
    ## 8   Magikarp   Shuckle  Magikarp    Horsea   Shuckle   Shuckle
    ## 9    Shuckle  Smoochum   Shuckle  Magikarp   Blissey   Blissey
    ## 10   Blissey   Blissey   Blissey   Shuckle   Ninjask   Ninjask
    ## 11   Ninjask   Ninjask   Ninjask   Blissey  Shedinja  Shedinja
    ## 12  Shedinja  Shedinja  Shedinja   Ninjask  Trapinch  Trapinch
    ## 13  Trapinch  Trapinch  Trapinch  Shedinja    Feebas    Feebas
    ## 14    Feebas    Feebas    Feebas  Trapinch    Regice    Regice
    ## 15    Wynaut    Regice    Deoxys    Feebas    Deoxys    Deoxys
    ## 16    Deoxys    Deoxys  Cranidos    Deoxys  Cranidos  Cranidos
    ## 17  Cranidos  Cranidos   Happiny  Cranidos   Happiny   Happiny
    ## 18  Accelgor  Accelgor  Accelgor   Happiny  Munchlax  Munchlax
    ## 19 Aegislash Aegislash Aegislash Aegislash Aegislash Aegislash
    ## 20   Zygarde   Zygarde   Zygarde   Zygarde   Zygarde   Zygarde

    ## [1] 0.8333333 0.8833333 0.9166667 0.9000000 0.9000000 0.9000000

![](Pokemon_Blog_files/figure-markdown_github/Examine%20Outliers-1.png)

    ## # A tibble: 20 x 9
    ##    pokedex_number name        hp attack defense sp_attack sp_defense speed
    ##             <int> <chr>    <int>  <int>   <int>     <int>      <int> <int>
    ##  1             15 Beedrill    65    150      40        15         80   145
    ##  2             63 Abra        25     20      15       105         55    90
    ##  3             80 Slowbro     95     75     180       130         80    30
    ##  4             92 Gastly      30     35      30       100         35    80
    ##  5             95 Onix        35     45     160        30         45    70
    ##  6            113 Chansey    250      5       5        35        105    50
    ##  7            116 Horsea      30     40      70        70         25    60
    ##  8            129 Magikarp    20     10      55        15         20    80
    ##  9            213 Shuckle     20     10     230        10        230     5
    ## 10            242 Blissey    255     10      10        75        135    55
    ## 11            291 Ninjask     61     90      45        50         50   160
    ## 12            292 Shedinja     1     90      45        30         30    40
    ## 13            328 Trapinch    45    100      45        45         45    10
    ## 14            349 Feebas      20     15      20        10         55    80
    ## 15            386 Deoxys      50     95      90        95         90   180
    ## 16            408 Cranidos    67    125      40        30         30    58
    ## 17            440 Happiny    100      5       5        15         65    30
    ## 18            617 Accelgor    80     70      40       100         60   145
    ## 19            681 Aegisla~    60    150      50       150         50    60
    ## 20            718 Zygarde    216    100     121        91         95    85
    ## # ... with 1 more variable: is_legendary <int>

Many of these outliers make sense to anyone familiar with the games. Chansey, Blissey, Shuckle, Cloyster, and Ninjask are all well known for having one or two incredibly high stats, but being weak in others. We also see Pokemon that are generally abysmal, notably Magikarp who evolves into a much more powerful pokemon than one would expect. There are surprisingly few Legendary pokemon in this list; only Deoxys (with the highest speed in the game) and Zygarde (with a massive health pool compared to most other Legendaries) show up. Only one of these Pokemon (Legendaries aside) stands out as exceptionally powerful: Aegislash. Aegislash boasts massive defense and special defense stats. What truly makes this pokemon shine is a variable hidden from this analysis: its ability. Aegislash's ability allows it to swap its offensive and defensive stats when it uses an offensive/defensive move. When properly used, Aegislash can serve as an offensive powerhouse and a wall simultaneously.

As a final step in our analysis, we will attempt to predict outliers with a binary classification model. First, we will have to create a flag to indicate outliers and select a cutoff of our outlier score values. Let's consider a cutoff of 1.3, which will give us 102 "outliers". The density plot we considered above also indicates that this could be a good cutoff to use due to the slight bump followed by decay at this point. Since in this case being an outlier is not necessarily good or bad, it seems appropriate to use a fairly low cutoff. We are interested in identifying pokemon with strange base stat distributions rather than identifying some data problem, so we want to be fairly liberal with our designation (and give the model plenty of data points to work with).

Predicting Outliers
-------------------

Now, we can attempt to model these outliers. Let's use a random forest model to classify these outliers.

``` r
#Train/Test split
set.seed(2112)
trainrows <- sample(1:nrow(pokemon), .6*nrow(pokemon), replace = F)

train <- pokemon[trainrows,]
test <- pokemon[-trainrows,]

#Confirm that there are outliers in both sets
summary(train)
```

    ##  pokedex_number      name               type1         type2    
    ##  Min.   :  1.0   Length:421         water  : 62   none   :214  
    ##  1st Qu.:204.0   Class :character   normal : 48   flying : 53  
    ##  Median :387.0   Mode  :character   bug    : 43   poison : 20  
    ##  Mean   :377.2                      grass  : 38   ground : 13  
    ##  3rd Qu.:550.0                      fire   : 29   steel  : 13  
    ##  Max.   :719.0                      psychic: 28   dark   : 11  
    ##                                     (Other):173   (Other): 97  
    ##     height_m       weight_kg            hp             attack      
    ##  Min.   :0.100   Min.   :  0.10   Min.   :  1.00   Min.   : 10.00  
    ##  1st Qu.:0.500   1st Qu.:  9.00   1st Qu.: 50.00   1st Qu.: 55.00  
    ##  Median :1.000   Median : 25.00   Median : 65.00   Median : 73.00  
    ##  Mean   :1.119   Mean   : 56.16   Mean   : 67.33   Mean   : 76.66  
    ##  3rd Qu.:1.400   3rd Qu.: 58.00   3rd Qu.: 80.00   3rd Qu.: 95.00  
    ##  Max.   :7.000   Max.   :950.00   Max.   :165.00   Max.   :180.00  
    ##                                                                    
    ##     defense         sp_attack        sp_defense         speed       
    ##  Min.   : 15.00   Min.   : 10.00   Min.   : 20.00   Min.   :  5.00  
    ##  1st Qu.: 50.00   1st Qu.: 48.00   1st Qu.: 50.00   1st Qu.: 45.00  
    ##  Median : 67.00   Median : 65.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 71.63   Mean   : 71.57   Mean   : 70.49   Mean   : 67.44  
    ##  3rd Qu.: 88.00   3rd Qu.: 90.00   3rd Qu.: 86.00   3rd Qu.: 90.00  
    ##  Max.   :230.00   Max.   :194.00   Max.   :230.00   Max.   :180.00  
    ##                                                                     
    ##    generation     is_legendary         total           score       
    ##  Min.   :1.000   Min.   :0.00000   Min.   :194.0   Min.   :0.9438  
    ##  1st Qu.:2.000   1st Qu.:0.00000   1st Qu.:316.0   1st Qu.:1.0124  
    ##  Median :4.000   Median :0.00000   Median :430.0   Median :1.0690  
    ##  Mean   :3.458   Mean   :0.07126   Mean   :425.1   Mean   :1.1250  
    ##  3rd Qu.:5.000   3rd Qu.:0.00000   3rd Qu.:500.0   3rd Qu.:1.1678  
    ##  Max.   :6.000   Max.   :1.00000   Max.   :780.0   Max.   :2.9285  
    ##                                                                    
    ##  is_outlier
    ##  0:371     
    ##  1: 50     
    ##            
    ##            
    ##            
    ##            
    ## 

``` r
summary(test)
```

    ##  pokedex_number     name               type1          type2    
    ##  Min.   :  2    Length:281         water  : 43   none    :141  
    ##  1st Qu.:185    Class :character   normal : 41   flying  : 34  
    ##  Median :338    Mode  :character   grass  : 27   psychic : 17  
    ##  Mean   :355                       bug    : 20   ground  : 14  
    ##  3rd Qu.:520                       psychic: 18   poison  : 11  
    ##  Max.   :721                       rock   : 17   fighting:  9  
    ##                                    (Other):115   (Other) : 55  
    ##     height_m        weight_kg            hp             attack     
    ##  Min.   : 0.100   Min.   :  0.10   Min.   : 20.00   Min.   :  5.0  
    ##  1st Qu.: 0.600   1st Qu.: 10.00   1st Qu.: 50.00   1st Qu.: 52.0  
    ##  Median : 1.000   Median : 30.60   Median : 70.00   Median : 75.0  
    ##  Mean   : 1.196   Mean   : 58.82   Mean   : 71.52   Mean   : 77.4  
    ##  3rd Qu.: 1.500   3rd Qu.: 71.00   3rd Qu.: 85.00   3rd Qu.:100.0  
    ##  Max.   :14.500   Max.   :550.00   Max.   :255.00   Max.   :185.0  
    ##                                                                    
    ##     defense         sp_attack        sp_defense         speed       
    ##  Min.   :  5.00   Min.   : 10.00   Min.   : 20.00   Min.   :  5.00  
    ##  1st Qu.: 50.00   1st Qu.: 45.00   1st Qu.: 50.00   1st Qu.: 45.00  
    ##  Median : 70.00   Median : 65.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 73.64   Mean   : 70.64   Mean   : 70.51   Mean   : 65.23  
    ##  3rd Qu.: 90.00   3rd Qu.: 92.00   3rd Qu.: 90.00   3rd Qu.: 85.00  
    ##  Max.   :230.00   Max.   :170.00   Max.   :200.00   Max.   :150.00  
    ##                                                                     
    ##    generation    is_legendary         total           score       
    ##  Min.   :1.00   Min.   :0.00000   Min.   :180.0   Min.   :0.9457  
    ##  1st Qu.:2.00   1st Qu.:0.00000   1st Qu.:330.0   1st Qu.:1.0145  
    ##  Median :3.00   Median :0.00000   Median :431.0   Median :1.0700  
    ##  Mean   :3.26   Mean   :0.07829   Mean   :428.9   Mean   :1.1551  
    ##  3rd Qu.:5.00   3rd Qu.:0.00000   3rd Qu.:515.0   3rd Qu.:1.2055  
    ##  Max.   :6.00   Max.   :1.00000   Max.   :720.0   Max.   :2.4280  
    ##                                                                   
    ##  is_outlier
    ##  0:232     
    ##  1: 49     
    ##            
    ##            
    ##            
    ##            
    ## 

``` r
rf <- randomForest(is_outlier ~ hp + attack + defense + sp_attack + sp_defense + speed, data = train)

rfpred <- predict(rf, test)
plot(roc(test$is_outlier, as.numeric(rfpred)))
```

![](Pokemon_Blog_files/figure-markdown_github/predict%20outliers-1.png)

``` r
confusionMatrix(as.factor(rfpred),test$is_outlier)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 229  31
    ##          1   3  18
    ##                                          
    ##                Accuracy : 0.879          
    ##                  95% CI : (0.835, 0.9147)
    ##     No Information Rate : 0.8256         
    ##     P-Value [Acc > NIR] : 0.009032       
    ##                                          
    ##                   Kappa : 0.4575         
    ##  Mcnemar's Test P-Value : 3.649e-06      
    ##                                          
    ##             Sensitivity : 0.9871         
    ##             Specificity : 0.3673         
    ##          Pos Pred Value : 0.8808         
    ##          Neg Pred Value : 0.8571         
    ##              Prevalence : 0.8256         
    ##          Detection Rate : 0.8149         
    ##    Detection Prevalence : 0.9253         
    ##       Balanced Accuracy : 0.6772         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

This model, which predicts if a pokemon is an outlier from just its 6 base stats, doesn't do a great job at predicting which ones are actually outliers. Let's try a few more models below and experiment with raising the cutoff to be considered an outlier:

``` r
#Full model
rf <- randomForest(is_outlier~. - pokedex_number - name - score, data = train)

rfpred <- predict(rf, test)
plot(roc(test$is_outlier, as.numeric(rfpred)))
```

![](Pokemon_Blog_files/figure-markdown_github/more%20models-1.png)

``` r
confusionMatrix(as.factor(rfpred),test$is_outlier)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 230  38
    ##          1   2  11
    ##                                           
    ##                Accuracy : 0.8577          
    ##                  95% CI : (0.8112, 0.8963)
    ##     No Information Rate : 0.8256          
    ##     P-Value [Acc > NIR] : 0.08825         
    ##                                           
    ##                   Kappa : 0.3039          
    ##  Mcnemar's Test P-Value : 3.13e-08        
    ##                                           
    ##             Sensitivity : 0.9914          
    ##             Specificity : 0.2245          
    ##          Pos Pred Value : 0.8582          
    ##          Neg Pred Value : 0.8462          
    ##              Prevalence : 0.8256          
    ##          Detection Rate : 0.8185          
    ##    Detection Prevalence : 0.9537          
    ##       Balanced Accuracy : 0.6079          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
#Changing Cutoff to 1.5
pokemon$score <- outlier.scores[,3]
pokemon$is_outlier <- as.factor(ifelse(pokemon$score >= 1.5, 1, 0))

set.seed(2112)
trainrows <- sample(1:nrow(pokemon), .6*nrow(pokemon), replace = F)

train <- pokemon[trainrows,]
test <- pokemon[-trainrows,]


rf <- randomForest(is_outlier~. - pokedex_number - name - score, data = train)

rfpred <- predict(rf, test)
plot(roc(test$is_outlier, as.numeric(rfpred)))
```

![](Pokemon_Blog_files/figure-markdown_github/more%20models-2.png)

``` r
confusionMatrix(as.factor(rfpred),test$is_outlier)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 259  19
    ##          1   0   3
    ##                                           
    ##                Accuracy : 0.9324          
    ##                  95% CI : (0.8964, 0.9588)
    ##     No Information Rate : 0.9217          
    ##     P-Value [Acc > NIR] : 0.297           
    ##                                           
    ##                   Kappa : 0.2254          
    ##  Mcnemar's Test P-Value : 3.636e-05       
    ##                                           
    ##             Sensitivity : 1.0000          
    ##             Specificity : 0.1364          
    ##          Pos Pred Value : 0.9317          
    ##          Neg Pred Value : 1.0000          
    ##              Prevalence : 0.9217          
    ##          Detection Rate : 0.9217          
    ##    Detection Prevalence : 0.9893          
    ##       Balanced Accuracy : 0.5682          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
#Changing Cutoff to 1.8
pokemon$score <- outlier.scores[,3]
pokemon$is_outlier <- as.factor(ifelse(pokemon$score >= 1.6, 1, 0))

set.seed(2112)
trainrows <- sample(1:nrow(pokemon), .6*nrow(pokemon), replace = F)

train <- pokemon[trainrows,]
test <- pokemon[-trainrows,]

#Confirm that there are outliers in both sets
summary(train)
```

    ##  pokedex_number      name               type1         type2    
    ##  Min.   :  1.0   Length:421         water  : 62   none   :214  
    ##  1st Qu.:204.0   Class :character   normal : 48   flying : 53  
    ##  Median :387.0   Mode  :character   bug    : 43   poison : 20  
    ##  Mean   :377.2                      grass  : 38   ground : 13  
    ##  3rd Qu.:550.0                      fire   : 29   steel  : 13  
    ##  Max.   :719.0                      psychic: 28   dark   : 11  
    ##                                     (Other):173   (Other): 97  
    ##     height_m       weight_kg            hp             attack      
    ##  Min.   :0.100   Min.   :  0.10   Min.   :  1.00   Min.   : 10.00  
    ##  1st Qu.:0.500   1st Qu.:  9.00   1st Qu.: 50.00   1st Qu.: 55.00  
    ##  Median :1.000   Median : 25.00   Median : 65.00   Median : 73.00  
    ##  Mean   :1.119   Mean   : 56.16   Mean   : 67.33   Mean   : 76.66  
    ##  3rd Qu.:1.400   3rd Qu.: 58.00   3rd Qu.: 80.00   3rd Qu.: 95.00  
    ##  Max.   :7.000   Max.   :950.00   Max.   :165.00   Max.   :180.00  
    ##                                                                    
    ##     defense         sp_attack        sp_defense         speed       
    ##  Min.   : 15.00   Min.   : 10.00   Min.   : 20.00   Min.   :  5.00  
    ##  1st Qu.: 50.00   1st Qu.: 48.00   1st Qu.: 50.00   1st Qu.: 45.00  
    ##  Median : 67.00   Median : 65.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 71.63   Mean   : 71.57   Mean   : 70.49   Mean   : 67.44  
    ##  3rd Qu.: 88.00   3rd Qu.: 90.00   3rd Qu.: 86.00   3rd Qu.: 90.00  
    ##  Max.   :230.00   Max.   :194.00   Max.   :230.00   Max.   :180.00  
    ##                                                                     
    ##    generation     is_legendary         total           score       
    ##  Min.   :1.000   Min.   :0.00000   Min.   :194.0   Min.   :0.9438  
    ##  1st Qu.:2.000   1st Qu.:0.00000   1st Qu.:316.0   1st Qu.:1.0124  
    ##  Median :4.000   Median :0.00000   Median :430.0   Median :1.0690  
    ##  Mean   :3.458   Mean   :0.07126   Mean   :425.1   Mean   :1.1250  
    ##  3rd Qu.:5.000   3rd Qu.:0.00000   3rd Qu.:500.0   3rd Qu.:1.1678  
    ##  Max.   :6.000   Max.   :1.00000   Max.   :780.0   Max.   :2.9285  
    ##                                                                    
    ##  is_outlier
    ##  0:406     
    ##  1: 15     
    ##            
    ##            
    ##            
    ##            
    ## 

``` r
summary(test)
```

    ##  pokedex_number     name               type1          type2    
    ##  Min.   :  2    Length:281         water  : 43   none    :141  
    ##  1st Qu.:185    Class :character   normal : 41   flying  : 34  
    ##  Median :338    Mode  :character   grass  : 27   psychic : 17  
    ##  Mean   :355                       bug    : 20   ground  : 14  
    ##  3rd Qu.:520                       psychic: 18   poison  : 11  
    ##  Max.   :721                       rock   : 17   fighting:  9  
    ##                                    (Other):115   (Other) : 55  
    ##     height_m        weight_kg            hp             attack     
    ##  Min.   : 0.100   Min.   :  0.10   Min.   : 20.00   Min.   :  5.0  
    ##  1st Qu.: 0.600   1st Qu.: 10.00   1st Qu.: 50.00   1st Qu.: 52.0  
    ##  Median : 1.000   Median : 30.60   Median : 70.00   Median : 75.0  
    ##  Mean   : 1.196   Mean   : 58.82   Mean   : 71.52   Mean   : 77.4  
    ##  3rd Qu.: 1.500   3rd Qu.: 71.00   3rd Qu.: 85.00   3rd Qu.:100.0  
    ##  Max.   :14.500   Max.   :550.00   Max.   :255.00   Max.   :185.0  
    ##                                                                    
    ##     defense         sp_attack        sp_defense         speed       
    ##  Min.   :  5.00   Min.   : 10.00   Min.   : 20.00   Min.   :  5.00  
    ##  1st Qu.: 50.00   1st Qu.: 45.00   1st Qu.: 50.00   1st Qu.: 45.00  
    ##  Median : 70.00   Median : 65.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 73.64   Mean   : 70.64   Mean   : 70.51   Mean   : 65.23  
    ##  3rd Qu.: 90.00   3rd Qu.: 92.00   3rd Qu.: 90.00   3rd Qu.: 85.00  
    ##  Max.   :230.00   Max.   :170.00   Max.   :200.00   Max.   :150.00  
    ##                                                                     
    ##    generation    is_legendary         total           score       
    ##  Min.   :1.00   Min.   :0.00000   Min.   :180.0   Min.   :0.9457  
    ##  1st Qu.:2.00   1st Qu.:0.00000   1st Qu.:330.0   1st Qu.:1.0145  
    ##  Median :3.00   Median :0.00000   Median :431.0   Median :1.0700  
    ##  Mean   :3.26   Mean   :0.07829   Mean   :428.9   Mean   :1.1551  
    ##  3rd Qu.:5.00   3rd Qu.:0.00000   3rd Qu.:515.0   3rd Qu.:1.2055  
    ##  Max.   :6.00   Max.   :1.00000   Max.   :720.0   Max.   :2.4280  
    ##                                                                   
    ##  is_outlier
    ##  0:265     
    ##  1: 16     
    ##            
    ##            
    ##            
    ##            
    ## 

``` r
rf <- randomForest(is_outlier~. - pokedex_number - name - score, data = train)

rfpred <- predict(rf, test)
plot(roc(test$is_outlier, as.numeric(rfpred)))
```

![](Pokemon_Blog_files/figure-markdown_github/more%20models-3.png)

``` r
confusionMatrix(as.factor(rfpred),test$is_outlier)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 265  14
    ##          1   0   2
    ##                                           
    ##                Accuracy : 0.9502          
    ##                  95% CI : (0.9178, 0.9725)
    ##     No Information Rate : 0.9431          
    ##     P-Value [Acc > NIR] : 0.362041        
    ##                                           
    ##                   Kappa : 0.2123          
    ##  Mcnemar's Test P-Value : 0.000512        
    ##                                           
    ##             Sensitivity : 1.0000          
    ##             Specificity : 0.1250          
    ##          Pos Pred Value : 0.9498          
    ##          Neg Pred Value : 1.0000          
    ##              Prevalence : 0.9431          
    ##          Detection Rate : 0.9431          
    ##    Detection Prevalence : 0.9929          
    ##       Balanced Accuracy : 0.5625          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

We see that these models all have high accuracy, but a low true negative rate. Since there are so few outliers, the models just tend to classify everything as not an outlier.

Conclusion
----------

In the case of pokemon, outliers are easy to identify but hard to predict. Using a nearest-neighbors approach to outlier identification does a pretty good job pointing out pokemon that any competitive players would recognize as statistical outliers. Trying to predict these outliers is a much more difficult task. It could be due to deep interactions between these statistics that our models have difficulty picking up. It's best to stick to catching-em-all than predicting-em-all
