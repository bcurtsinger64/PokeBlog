Pokemon Anomaly Detection & Clustering
================
Bryce Curtsinger
November 20, 2018

Pokemon Clustering
------------------

When I was a child, I looked forward to spending my Saturday mornings watching cartoons, as millions of other kids did. Of all the shows, Pokemon was always my favorite. In 2001, my dreams of becoming a Pokemon master came true (kinda) when my mother bought me the newest game, Pokemon Crystal, for the Game Boy Color. This sparked my love of Pokemon games that followed me into my teenage years and adulthood. I fell out for a while after Fire Red/ Leaf Green, then picked up the series again with Pokemon X and Y. This time, I became aware of the competitive scene of Pokemon battling, and fell in love all over again.

In this post, I will use data of 721 Pokemon to perform a cluster analysis with the purpose of identifying outliers and comparing clusters with Smogon's compeitive tier list. For those unfamiliar, Smogon is an entity which maintains a comprehensive online resource for competitive pokemon battling and fosters a tournament scene with tier lists for different types of battles. The data used for this analysis comes from two sources. One is a list of data on all pokemon maintained on kaggle (<https://www.kaggle.com/rounakbanik/pokemon>). The second dataset, also from kaggle (<https://www.kaggle.com/notgibs/smogon-6v6-pokemon-tiers/version/1>), overlaps heavily with the first dataset, but also contains information about each Pokemon's status on Smogon's tier list. Note that the second dataset does not contain the newest generation, so this analysis will only include generations 1-6.

Finally, a brief description of packages used is provided below.

pacman - facilitates easy installation and loading of other R packages

tidyverse - provides a multitude of data manipulation functions, among other things, to augment base R data manipulation.

ggplot2 - comprehensive visualization tool

Rlof - used for anomaly detection with lof function

``` r
if(!require(pacman)) install.packages(pacman)
```

    ## Loading required package: pacman

``` r
p_load(tidyverse)
p_load(ggplot2)
p_load(Rlof)

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
  select(-c(1:19), -japanese_name, -experience_growth, -base_happiness, -base_egg_steps, -base_total, -capture_rate, - classfication) %>% #get rid of these variables
  select(pokedex_number, name, type1, type2, percentage_male, height_m, weight_kg,
          hp, attack, defense, sp_attack, sp_defense, speed, generation, is_legendary) #reorder variables

#Smogon Data:
smogon <- read_csv('smogon.csv') #Read in data from smogon
```

    ## Parsed with column specification:
    ## cols(
    ##   X. = col_integer(),
    ##   Name = col_character(),
    ##   Type.1 = col_character(),
    ##   Type.2 = col_character(),
    ##   Total = col_integer(),
    ##   HP = col_integer(),
    ##   Attack = col_integer(),
    ##   Defense = col_integer(),
    ##   Sp..Atk = col_integer(),
    ##   Sp..Def = col_integer(),
    ##   Speed = col_integer(),
    ##   Generation = col_integer(),
    ##   Legendary = col_logical(),
    ##   Mega = col_logical(),
    ##   Tier = col_character()
    ## )

``` r
smogon <- smogon %>% 
  rename_all(tolower) %>% 
  filter(mega == F) %>% #Filter out mega evolutions that are not in the base data
  select(name, total, tier) #Keep the Stat total & smogon tier
pokemon <- pokemon %>% left_join(smogon, by = 'name') #join dataframes

pokemon <- pokemon %>% #Change column types
  mutate(type1 = as.factor(type1), type2 = as.factor(type2), tier = as.factor(tier), total = hp + attack + defense + sp_attack + sp_defense + speed) %>% 
  mutate(type2 = as.character(type2), tier = as.character(tier)) %>%
  replace_na(replace = list('type2' = 'none', 'tier' = 'BelowNU')) %>%
  mutate(type2 = as.factor(type2), tier = as.factor(tier))
#Inspect Data
head(pokemon)
```

    ## # A tibble: 6 x 17
    ##   pokedex_number name       type1 type2 percentage_male height_m weight_kg
    ##            <int> <chr>      <fct> <fct>           <dbl>    <dbl>     <dbl>
    ## 1              1 Bulbasaur  grass pois~            88.1    0.700      6.90
    ## 2              2 Ivysaur    grass pois~            88.1    1.00      13.0 
    ## 3              3 Venusaur   grass pois~            88.1    2.00     100.  
    ## 4              4 Charmander fire  none             88.1    0.600      8.50
    ## 5              5 Charmeleon fire  none             88.1    1.10      19.0 
    ## 6              6 Charizard  fire  flyi~            88.1    1.70      90.5 
    ## # ... with 10 more variables: hp <int>, attack <int>, defense <int>,
    ## #   sp_attack <int>, sp_defense <int>, speed <int>, generation <int>,
    ## #   is_legendary <int>, total <int>, tier <fct>

``` r
summary(pokemon)
```

    ##  pokedex_number     name               type1         type2    
    ##  Min.   :  1    Length:721         water  :105   none   :355  
    ##  1st Qu.:181    Class :character   normal : 93   flying : 87  
    ##  Median :361    Mode  :character   grass  : 66   poison : 33  
    ##  Mean   :361                       bug    : 63   ground : 32  
    ##  3rd Qu.:541                       fire   : 47   psychic: 27  
    ##  Max.   :721                       psychic: 47   dark   : 20  
    ##                                    (Other):300   (Other):167  
    ##  percentage_male     height_m       weight_kg             hp        
    ##  Min.   :  0.00   Min.   : 0.10   Min.   :  0.100   Min.   :  1.00  
    ##  1st Qu.: 50.00   1st Qu.: 0.60   1st Qu.:  9.425   1st Qu.: 50.00  
    ##  Median : 50.00   Median : 1.00   Median : 28.000   Median : 65.00  
    ##  Mean   : 55.43   Mean   : 1.15   Mean   : 57.223   Mean   : 68.78  
    ##  3rd Qu.: 50.00   3rd Qu.: 1.50   3rd Qu.: 61.375   3rd Qu.: 80.00  
    ##  Max.   :100.00   Max.   :14.50   Max.   :950.000   Max.   :255.00  
    ##  NA's   :77       NA's   :19      NA's   :19                        
    ##      attack          defense         sp_attack        sp_defense    
    ##  Min.   :  5.00   Min.   :  5.00   Min.   : 10.00   Min.   : 20.00  
    ##  1st Qu.: 55.00   1st Qu.: 50.00   1st Qu.: 45.00   1st Qu.: 50.00  
    ##  Median : 75.00   Median : 68.00   Median : 65.00   Median : 65.00  
    ##  Mean   : 77.11   Mean   : 72.45   Mean   : 70.87   Mean   : 70.39  
    ##  3rd Qu.: 98.00   3rd Qu.: 90.00   3rd Qu.: 90.00   3rd Qu.: 86.00  
    ##  Max.   :185.00   Max.   :230.00   Max.   :194.00   Max.   :230.00  
    ##                                                                     
    ##      speed          generation     is_legendary         total      
    ##  Min.   :  5.00   Min.   :1.000   Min.   :0.00000   Min.   :180.0  
    ##  1st Qu.: 45.00   1st Qu.:2.000   1st Qu.:0.00000   1st Qu.:320.0  
    ##  Median : 65.00   Median :3.000   Median :0.00000   Median :430.0  
    ##  Mean   : 66.59   Mean   :3.323   Mean   :0.07351   Mean   :426.2  
    ##  3rd Qu.: 88.00   3rd Qu.:5.000   3rd Qu.:0.00000   3rd Qu.:505.0  
    ##  Max.   :180.00   Max.   :6.000   Max.   :1.00000   Max.   :780.0  
    ##                                                                    
    ##       tier    
    ##  BelowNU:299  
    ##  PU     :168  
    ##  NU     : 56  
    ##  UU     : 55  
    ##  RU     : 49  
    ##  OU     : 36  
    ##  (Other): 58

Outlier Identification
----------------------

Now that we've joined our datasets, let's begin with outlier detection. In this case, we will focus our analysis primarily on the primary statistics of each pokemon: HP, attack, defense, special attack, special defense, and speed. Anyone familiar with competitive pokemon will immediately think of several pokemon that are outliers in terms of these statistics, so let's see if the results are what we would expect. The first outlier detection technique used is LOF (local outlier factor), which identifies density-based outliers using a nearest-neighbors approach. Here, we compared to the 5 nearest neighbors to identify outliers. Let's plot the density of the outliers:

![](Pokemon_Blog_files/figure-markdown_github/Outlier%20Density-1.png)

This plot indicates the presence of outliers because of the right-skewed distribution. Let's examine the top 20 outliers:

``` r
names <- data.frame('k5' = rep(NA,20),
                    'k6' = rep(NA,20),
                    'k7' = rep(NA,20),
                    'k8' = rep(NA,20),
                    'k9' = rep(NA,20),
                    'k10' = rep(NA,20))
for(i in 1:6){
  threshold <- sort(outlier.scores[,i], decreasing=T)[20]
  outlierIDs <- which(outlier.scores[,i]>=threshold)
  names[,i] <- pokemon[outlierIDs,'name']
}
names
```

    ##           k5        k6        k7        k8        k9       k10
    ## 1   Beedrill  Beedrill  Beedrill  Beedrill  Beedrill  Beedrill
    ## 2   Cloyster   Slowbro      Abra      Abra      Abra      Abra
    ## 3     Gastly  Cloyster   Slowbro  Alakazam    Gastly    Gastly
    ## 4    Chansey    Gastly  Cloyster   Slowbro      Onix      Onix
    ## 5     Horsea      Onix    Gastly    Gastly   Chansey   Chansey
    ## 6     Staryu   Chansey      Onix      Onix    Horsea    Horsea
    ## 7   Magikarp    Horsea   Chansey   Chansey  Magikarp  Magikarp
    ## 8    Shuckle  Magikarp    Horsea    Horsea   Steelix   Steelix
    ## 9   Smoochum   Shuckle  Magikarp  Magikarp   Shuckle   Shuckle
    ## 10   Blissey   Blissey   Shuckle   Shuckle   Blissey   Blissey
    ## 11   Ninjask   Ninjask   Blissey   Blissey   Ninjask   Ninjask
    ## 12  Shedinja  Shedinja   Ninjask   Ninjask  Shedinja  Shedinja
    ## 13  Trapinch  Trapinch  Shedinja  Shedinja    Aggron    Aggron
    ## 14    Feebas    Regice  Trapinch  Trapinch  Trapinch  Trapinch
    ## 15    Wynaut    Deoxys    Deoxys    Feebas    Feebas    Deoxys
    ## 16    Deoxys  Cranidos  Cranidos    Deoxys    Deoxys  Cranidos
    ## 17  Cranidos  Munchlax   Happiny  Cranidos  Cranidos   Happiny
    ## 18  Accelgor  Accelgor  Accelgor   Happiny   Happiny  Munchlax
    ## 19 Aegislash Aegislash Aegislash Aegislash Aegislash Aegislash
    ## 20   Zygarde   Zygarde   Zygarde   Zygarde   Zygarde   Zygarde

``` r
OutlierOverlap <- matrix(rep(NA, 36), nrow = 6, ncol = 6)

for( i in 1:6) {
  for( j in 1:6){
    OutlierOverlap[i,j] <- sum(names[,i] %in% names[,j])/20
  }
}
apply(OutlierOverlap, 1, mean)
```

    ## [1] 0.8000000 0.8416667 0.8833333 0.8666667 0.8666667 0.8583333

``` r
threshold <- sort(outlier.scores[,3], decreasing=T)[20]
  outlierIDs <- which(outlier.scores[,3]>=threshold)
  pokemon %>% slice(outlierIDs) %>% select(pokedex_number, name, hp, attack, defense, sp_attack, sp_defense, speed, is_legendary)
```

    ## # A tibble: 20 x 9
    ##    pokedex_number name        hp attack defense sp_attack sp_defense speed
    ##             <int> <chr>    <int>  <int>   <int>     <int>      <int> <int>
    ##  1             15 Beedrill    65    150      40        15         80   145
    ##  2             63 Abra        25     20      15       105         55    90
    ##  3             80 Slowbro     95     75     180       130         80    30
    ##  4             91 Cloyster    50     95     180        85         45    70
    ##  5             92 Gastly      30     35      30       100         35    80
    ##  6             95 Onix        35     45     160        30         45    70
    ##  7            113 Chansey    250      5       5        35        105    50
    ##  8            116 Horsea      30     40      70        70         25    60
    ##  9            129 Magikarp    20     10      55        15         20    80
    ## 10            213 Shuckle     20     10     230        10        230     5
    ## 11            242 Blissey    255     10      10        75        135    55
    ## 12            291 Ninjask     61     90      45        50         50   160
    ## 13            292 Shedinja     1     90      45        30         30    40
    ## 14            328 Trapinch    45    100      45        45         45    10
    ## 15            386 Deoxys      50     95      90        95         90   180
    ## 16            408 Cranidos    67    125      40        30         30    58
    ## 17            440 Happiny    100      5       5        15         65    30
    ## 18            617 Accelgor    80     70      40       100         60   145
    ## 19            681 Aegisla~    60    150      50       150         50    60
    ## 20            718 Zygarde    216    100     121        91         95    85
    ## # ... with 1 more variable: is_legendary <int>

Many of these outliers make sense to anyone familiar with the games. Chansey & Blissey, Shuckle, Cloyster, and Ninjask are all well known for having one or two incredibly high stats, but being weak in others. We also see Pokemon that are generally abysmal, notably Magikarp who evolves into a much more powerful pokemon than one would expect. There are surprisingly few Legendary pokemon in this list; only Deoxys (with the highest speed in the game) and Zygarde (with a massive health pool compared to most other Legendaries) show up. Only one of these Pokemon (Legendaries aside) stands out as exceptionally powerful: Aegislash. One of only two non-Legendary pokemon banned from OU tier, Aegislash boasts massive defense and special defense stats. What truly makes this pokemon shine is a variable hidden from this analysis: its ability. Aegislash's ability allows it to swap its offensive and defensive stats when it uses an offensive/defensive move. When properly used, Aegislash can serve as an offensive powerhouse and a wall simultaneously.
