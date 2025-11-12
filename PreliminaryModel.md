Intermediate Deliverable 1
================
Third Time’s the Charm
2025-10-23

## Step 0: Why?

**Dataset Chosen:**  
Hotel Reservations (~35,000 rows)  
Predicts hotel booking cancellations and potential no-shows. The
business use case focuses on **optimizing overbooking decisions** to
maximize hotel occupancy and revenue.

### Context (from Kaggle)

> “The online hotel reservation channels have dramatically changed
> booking possibilities and customers’ behavior. A significant number of
> hotel reservations are called-off due to cancellations or no-shows.
> The typical reasons for cancellations include change of plans,
> scheduling conflicts, etc. This is often made easier by the option to
> do so free of charge or preferably at a low cost which is beneficial
> to hotel guests but it is a less desirable and possibly
> revenue-diminishing factor for hotels to deal with.”

### Business Use Case

We are building a model to predict which hotel reservations are most
likely to cancel.  
Although cancellations often result from a change of plans or scheduling
conflicts, they impose a direct financial cost on the hotel.  
To mitigate these losses and **maintain 100% occupancy**, the hotel can
strategically **overbook** based on model predictions of cancellation
likelihood.

**Main Question:**  
Can we predict which hotel reservations are most likely to be canceled,
and use these predictions to make more profitable overbooking decisions?

**Why it’s answerable with this dataset:**  
- The dataset includes **booking-level data** for over 35 000
reservations with both categorical and numerical features (lead time,
room type, guest history, etc.).  
- The dependent variable `booking_status` is binary (Canceled / Not
Canceled), making it well-suited for **classification via logistic
regression**.  
- With exploratory cleaning and modeling, we can estimate how much
predictive signal exists and whether cancellation behavior is
predictable enough to support overbooking decisions.

### Financial Assumptions

- **Revenue per booked room:** \$300 per night  
- **Cancellations:** Guests can cancel free of charge at any time; each
  cancellation results in a \$300 revenue loss.  
- **Overbooking cost:** If the hotel overbooks and cannot accommodate a
  guest (because no cancellations occur), it incurs \$400 per night in
  re-accommodation, goodwill, or compensation costs.  
- **Variable costs per room:** Considered sunk and excluded from
  analysis (the cost of turning a room is identical regardless of
  occupancy).  
- **No difference** between cancellation and no-show costs.

### Cost–Benefit Matrix

| Case | Model Prediction | Actual Outcome | Financial Impact |
|----|----|----|----|
| **True Positive (TP)** | “Will Cancel” | Guest cancels | **+ \$300** (revenue from replacement guest) |
| **True Negative (TN)** | “Will Not Cancel” | Guest does not cancel | **\$0** (no change in revenue) |
| **False Negative (FN)** | “Will Not Cancel” | Guest cancels | **– \$300** (lost booking revenue) |
| **False Positive (FP)** | “Will Cancel” | Guest shows up | **– \$400** (overbooking penalty) |

### Objective

Develop a predictive model that helps hotels **balance the trade-off**
between overbooking risk and lost revenue due to cancellations, thereby
maximizing **expected net revenue** and improving operational
decision-making.

### Step 0.5: Load Packages

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(neuralnet)
library(class)
library(C50)
library(ggplot2)
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 4.5.2

    ## randomForest 4.7-1.2

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(kernlab)
```

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

## Step 1: Load Data

``` r
hotel <- read.csv("Hotel Reservations.csv", stringsAsFactors = TRUE)
```

## Step 2: Clean Data

``` r
str(hotel)
```

    ## 'data.frame':    36275 obs. of  19 variables:
    ##  $ Booking_ID                          : Factor w/ 36275 levels "INN00001","INN00002",..: 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ no_of_adults                        : int  2 2 1 2 2 2 2 2 3 2 ...
    ##  $ no_of_children                      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_weekend_nights                : int  1 2 2 0 1 0 1 1 0 0 ...
    ##  $ no_of_week_nights                   : int  2 3 1 2 1 2 3 3 4 5 ...
    ##  $ type_of_meal_plan                   : Factor w/ 4 levels "Meal Plan 1",..: 1 4 1 1 4 2 1 1 1 1 ...
    ##  $ required_car_parking_space          : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ room_type_reserved                  : Factor w/ 7 levels "Room_Type 1",..: 1 1 1 1 1 1 1 4 1 4 ...
    ##  $ lead_time                           : int  224 5 1 211 48 346 34 83 121 44 ...
    ##  $ arrival_year                        : int  2017 2018 2018 2018 2018 2018 2017 2018 2018 2018 ...
    ##  $ arrival_month                       : int  10 11 2 5 4 9 10 12 7 10 ...
    ##  $ arrival_date                        : int  2 6 28 20 11 13 15 26 6 18 ...
    ##  $ market_segment_type                 : Factor w/ 5 levels "Aviation","Complementary",..: 4 5 5 5 5 5 5 5 4 5 ...
    ##  $ repeated_guest                      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_previous_cancellations        : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_previous_bookings_not_canceled: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_price_per_room                  : num  65 106.7 60 100 94.5 ...
    ##  $ no_of_special_requests              : int  0 1 0 0 0 1 1 1 1 3 ...
    ##  $ booking_status                      : Factor w/ 2 levels "Canceled","Not_Canceled": 2 2 1 1 1 1 2 2 2 2 ...

``` r
summary(hotel)
```

    ##     Booking_ID     no_of_adults   no_of_children    no_of_weekend_nights
    ##  INN00001:    1   Min.   :0.000   Min.   : 0.0000   Min.   :0.0000      
    ##  INN00002:    1   1st Qu.:2.000   1st Qu.: 0.0000   1st Qu.:0.0000      
    ##  INN00003:    1   Median :2.000   Median : 0.0000   Median :1.0000      
    ##  INN00004:    1   Mean   :1.845   Mean   : 0.1053   Mean   :0.8107      
    ##  INN00005:    1   3rd Qu.:2.000   3rd Qu.: 0.0000   3rd Qu.:2.0000      
    ##  INN00006:    1   Max.   :4.000   Max.   :10.0000   Max.   :7.0000      
    ##  (Other) :36269                                                         
    ##  no_of_week_nights    type_of_meal_plan required_car_parking_space
    ##  Min.   : 0.000    Meal Plan 1 :27835   Min.   :0.00000           
    ##  1st Qu.: 1.000    Meal Plan 2 : 3305   1st Qu.:0.00000           
    ##  Median : 2.000    Meal Plan 3 :    5   Median :0.00000           
    ##  Mean   : 2.204    Not Selected: 5130   Mean   :0.03099           
    ##  3rd Qu.: 3.000                         3rd Qu.:0.00000           
    ##  Max.   :17.000                         Max.   :1.00000           
    ##                                                                   
    ##    room_type_reserved   lead_time       arrival_year  arrival_month   
    ##  Room_Type 1:28130    Min.   :  0.00   Min.   :2017   Min.   : 1.000  
    ##  Room_Type 2:  692    1st Qu.: 17.00   1st Qu.:2018   1st Qu.: 5.000  
    ##  Room_Type 3:    7    Median : 57.00   Median :2018   Median : 8.000  
    ##  Room_Type 4: 6057    Mean   : 85.23   Mean   :2018   Mean   : 7.424  
    ##  Room_Type 5:  265    3rd Qu.:126.00   3rd Qu.:2018   3rd Qu.:10.000  
    ##  Room_Type 6:  966    Max.   :443.00   Max.   :2018   Max.   :12.000  
    ##  Room_Type 7:  158                                                    
    ##   arrival_date     market_segment_type repeated_guest   
    ##  Min.   : 1.0   Aviation     :  125    Min.   :0.00000  
    ##  1st Qu.: 8.0   Complementary:  391    1st Qu.:0.00000  
    ##  Median :16.0   Corporate    : 2017    Median :0.00000  
    ##  Mean   :15.6   Offline      :10528    Mean   :0.02564  
    ##  3rd Qu.:23.0   Online       :23214    3rd Qu.:0.00000  
    ##  Max.   :31.0                          Max.   :1.00000  
    ##                                                         
    ##  no_of_previous_cancellations no_of_previous_bookings_not_canceled
    ##  Min.   : 0.00000             Min.   : 0.0000                     
    ##  1st Qu.: 0.00000             1st Qu.: 0.0000                     
    ##  Median : 0.00000             Median : 0.0000                     
    ##  Mean   : 0.02335             Mean   : 0.1534                     
    ##  3rd Qu.: 0.00000             3rd Qu.: 0.0000                     
    ##  Max.   :13.00000             Max.   :58.0000                     
    ##                                                                   
    ##  avg_price_per_room no_of_special_requests      booking_status 
    ##  Min.   :  0.00     Min.   :0.0000         Canceled    :11885  
    ##  1st Qu.: 80.30     1st Qu.:0.0000         Not_Canceled:24390  
    ##  Median : 99.45     Median :0.0000                             
    ##  Mean   :103.42     Mean   :0.6197                             
    ##  3rd Qu.:120.00     3rd Qu.:1.0000                             
    ##  Max.   :540.00     Max.   :5.0000                             
    ## 

Let’s remove booking id. Let’s also convert our dependent variable
(booking status) into 0s and 1s.

``` r
hotel$Booking_ID = NULL
hotel$booking_status <- ifelse(hotel$booking_status == "Canceled", 1, 0)
str(hotel)
```

    ## 'data.frame':    36275 obs. of  18 variables:
    ##  $ no_of_adults                        : int  2 2 1 2 2 2 2 2 3 2 ...
    ##  $ no_of_children                      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_weekend_nights                : int  1 2 2 0 1 0 1 1 0 0 ...
    ##  $ no_of_week_nights                   : int  2 3 1 2 1 2 3 3 4 5 ...
    ##  $ type_of_meal_plan                   : Factor w/ 4 levels "Meal Plan 1",..: 1 4 1 1 4 2 1 1 1 1 ...
    ##  $ required_car_parking_space          : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ room_type_reserved                  : Factor w/ 7 levels "Room_Type 1",..: 1 1 1 1 1 1 1 4 1 4 ...
    ##  $ lead_time                           : int  224 5 1 211 48 346 34 83 121 44 ...
    ##  $ arrival_year                        : int  2017 2018 2018 2018 2018 2018 2017 2018 2018 2018 ...
    ##  $ arrival_month                       : int  10 11 2 5 4 9 10 12 7 10 ...
    ##  $ arrival_date                        : int  2 6 28 20 11 13 15 26 6 18 ...
    ##  $ market_segment_type                 : Factor w/ 5 levels "Aviation","Complementary",..: 4 5 5 5 5 5 5 5 4 5 ...
    ##  $ repeated_guest                      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_previous_cancellations        : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ no_of_previous_bookings_not_canceled: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_price_per_room                  : num  65 106.7 60 100 94.5 ...
    ##  $ no_of_special_requests              : int  0 1 0 0 0 1 1 1 1 3 ...
    ##  $ booking_status                      : num  0 0 1 1 1 1 0 0 0 0 ...

``` r
summary(hotel)
```

    ##   no_of_adults   no_of_children    no_of_weekend_nights no_of_week_nights
    ##  Min.   :0.000   Min.   : 0.0000   Min.   :0.0000       Min.   : 0.000   
    ##  1st Qu.:2.000   1st Qu.: 0.0000   1st Qu.:0.0000       1st Qu.: 1.000   
    ##  Median :2.000   Median : 0.0000   Median :1.0000       Median : 2.000   
    ##  Mean   :1.845   Mean   : 0.1053   Mean   :0.8107       Mean   : 2.204   
    ##  3rd Qu.:2.000   3rd Qu.: 0.0000   3rd Qu.:2.0000       3rd Qu.: 3.000   
    ##  Max.   :4.000   Max.   :10.0000   Max.   :7.0000       Max.   :17.000   
    ##                                                                          
    ##     type_of_meal_plan required_car_parking_space   room_type_reserved
    ##  Meal Plan 1 :27835   Min.   :0.00000            Room_Type 1:28130   
    ##  Meal Plan 2 : 3305   1st Qu.:0.00000            Room_Type 2:  692   
    ##  Meal Plan 3 :    5   Median :0.00000            Room_Type 3:    7   
    ##  Not Selected: 5130   Mean   :0.03099            Room_Type 4: 6057   
    ##                       3rd Qu.:0.00000            Room_Type 5:  265   
    ##                       Max.   :1.00000            Room_Type 6:  966   
    ##                                                  Room_Type 7:  158   
    ##    lead_time       arrival_year  arrival_month     arrival_date 
    ##  Min.   :  0.00   Min.   :2017   Min.   : 1.000   Min.   : 1.0  
    ##  1st Qu.: 17.00   1st Qu.:2018   1st Qu.: 5.000   1st Qu.: 8.0  
    ##  Median : 57.00   Median :2018   Median : 8.000   Median :16.0  
    ##  Mean   : 85.23   Mean   :2018   Mean   : 7.424   Mean   :15.6  
    ##  3rd Qu.:126.00   3rd Qu.:2018   3rd Qu.:10.000   3rd Qu.:23.0  
    ##  Max.   :443.00   Max.   :2018   Max.   :12.000   Max.   :31.0  
    ##                                                                 
    ##     market_segment_type repeated_guest    no_of_previous_cancellations
    ##  Aviation     :  125    Min.   :0.00000   Min.   : 0.00000            
    ##  Complementary:  391    1st Qu.:0.00000   1st Qu.: 0.00000            
    ##  Corporate    : 2017    Median :0.00000   Median : 0.00000            
    ##  Offline      :10528    Mean   :0.02564   Mean   : 0.02335            
    ##  Online       :23214    3rd Qu.:0.00000   3rd Qu.: 0.00000            
    ##                         Max.   :1.00000   Max.   :13.00000            
    ##                                                                       
    ##  no_of_previous_bookings_not_canceled avg_price_per_room no_of_special_requests
    ##  Min.   : 0.0000                      Min.   :  0.00     Min.   :0.0000        
    ##  1st Qu.: 0.0000                      1st Qu.: 80.30     1st Qu.:0.0000        
    ##  Median : 0.0000                      Median : 99.45     Median :0.0000        
    ##  Mean   : 0.1534                      Mean   :103.42     Mean   :0.6197        
    ##  3rd Qu.: 0.0000                      3rd Qu.:120.00     3rd Qu.:1.0000        
    ##  Max.   :58.0000                      Max.   :540.00     Max.   :5.0000        
    ##                                                                                
    ##  booking_status  
    ##  Min.   :0.0000  
    ##  1st Qu.:0.0000  
    ##  Median :0.0000  
    ##  Mean   :0.3276  
    ##  3rd Qu.:1.0000  
    ##  Max.   :1.0000  
    ## 

## Step 3: Split Data

We have 36000 data entries. In order to build a second level model, we
are going to keep 30% for evaluation. Let’s normalize and dummify the
data.

``` r
hotel_dummies <- as.data.frame(model.matrix(~ . - booking_status - 1, data = hotel))
hotel_dummies$booking_status <- hotel$booking_status
minmax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

hotel_scaled <- as.data.frame(lapply(hotel_dummies, minmax))
summary(hotel_scaled)
```

    ##   no_of_adults    no_of_children    no_of_weekend_nights no_of_week_nights
    ##  Min.   :0.0000   Min.   :0.00000   Min.   :0.0000       Min.   :0.00000  
    ##  1st Qu.:0.5000   1st Qu.:0.00000   1st Qu.:0.0000       1st Qu.:0.05882  
    ##  Median :0.5000   Median :0.00000   Median :0.1429       Median :0.11765  
    ##  Mean   :0.4612   Mean   :0.01053   Mean   :0.1158       Mean   :0.12966  
    ##  3rd Qu.:0.5000   3rd Qu.:0.00000   3rd Qu.:0.2857       3rd Qu.:0.17647  
    ##  Max.   :1.0000   Max.   :1.00000   Max.   :1.0000       Max.   :1.00000  
    ##  type_of_meal_planMeal.Plan.1 type_of_meal_planMeal.Plan.2
    ##  Min.   :0.0000               Min.   :0.00000             
    ##  1st Qu.:1.0000               1st Qu.:0.00000             
    ##  Median :1.0000               Median :0.00000             
    ##  Mean   :0.7673               Mean   :0.09111             
    ##  3rd Qu.:1.0000               3rd Qu.:0.00000             
    ##  Max.   :1.0000               Max.   :1.00000             
    ##  type_of_meal_planMeal.Plan.3 type_of_meal_planNot.Selected
    ##  Min.   :0.0000000            Min.   :0.0000               
    ##  1st Qu.:0.0000000            1st Qu.:0.0000               
    ##  Median :0.0000000            Median :0.0000               
    ##  Mean   :0.0001378            Mean   :0.1414               
    ##  3rd Qu.:0.0000000            3rd Qu.:0.0000               
    ##  Max.   :1.0000000            Max.   :1.0000               
    ##  required_car_parking_space room_type_reservedRoom_Type.2
    ##  Min.   :0.00000            Min.   :0.00000              
    ##  1st Qu.:0.00000            1st Qu.:0.00000              
    ##  Median :0.00000            Median :0.00000              
    ##  Mean   :0.03099            Mean   :0.01908              
    ##  3rd Qu.:0.00000            3rd Qu.:0.00000              
    ##  Max.   :1.00000            Max.   :1.00000              
    ##  room_type_reservedRoom_Type.3 room_type_reservedRoom_Type.4
    ##  Min.   :0.000000              Min.   :0.000                
    ##  1st Qu.:0.000000              1st Qu.:0.000                
    ##  Median :0.000000              Median :0.000                
    ##  Mean   :0.000193              Mean   :0.167                
    ##  3rd Qu.:0.000000              3rd Qu.:0.000                
    ##  Max.   :1.000000              Max.   :1.000                
    ##  room_type_reservedRoom_Type.5 room_type_reservedRoom_Type.6
    ##  Min.   :0.000000              Min.   :0.00000              
    ##  1st Qu.:0.000000              1st Qu.:0.00000              
    ##  Median :0.000000              Median :0.00000              
    ##  Mean   :0.007305              Mean   :0.02663              
    ##  3rd Qu.:0.000000              3rd Qu.:0.00000              
    ##  Max.   :1.000000              Max.   :1.00000              
    ##  room_type_reservedRoom_Type.7   lead_time        arrival_year   
    ##  Min.   :0.000000              Min.   :0.00000   Min.   :0.0000  
    ##  1st Qu.:0.000000              1st Qu.:0.03837   1st Qu.:1.0000  
    ##  Median :0.000000              Median :0.12867   Median :1.0000  
    ##  Mean   :0.004356              Mean   :0.19240   Mean   :0.8204  
    ##  3rd Qu.:0.000000              3rd Qu.:0.28442   3rd Qu.:1.0000  
    ##  Max.   :1.000000              Max.   :1.00000   Max.   :1.0000  
    ##  arrival_month     arrival_date    market_segment_typeComplementary
    ##  Min.   :0.0000   Min.   :0.0000   Min.   :0.00000                 
    ##  1st Qu.:0.3636   1st Qu.:0.2333   1st Qu.:0.00000                 
    ##  Median :0.6364   Median :0.5000   Median :0.00000                 
    ##  Mean   :0.5840   Mean   :0.4866   Mean   :0.01078                 
    ##  3rd Qu.:0.8182   3rd Qu.:0.7333   3rd Qu.:0.00000                 
    ##  Max.   :1.0000   Max.   :1.0000   Max.   :1.00000                 
    ##  market_segment_typeCorporate market_segment_typeOffline
    ##  Min.   :0.0000               Min.   :0.0000            
    ##  1st Qu.:0.0000               1st Qu.:0.0000            
    ##  Median :0.0000               Median :0.0000            
    ##  Mean   :0.0556               Mean   :0.2902            
    ##  3rd Qu.:0.0000               3rd Qu.:1.0000            
    ##  Max.   :1.0000               Max.   :1.0000            
    ##  market_segment_typeOnline repeated_guest    no_of_previous_cancellations
    ##  Min.   :0.0000            Min.   :0.00000   Min.   :0.000000            
    ##  1st Qu.:0.0000            1st Qu.:0.00000   1st Qu.:0.000000            
    ##  Median :1.0000            Median :0.00000   Median :0.000000            
    ##  Mean   :0.6399            Mean   :0.02564   Mean   :0.001796            
    ##  3rd Qu.:1.0000            3rd Qu.:0.00000   3rd Qu.:0.000000            
    ##  Max.   :1.0000            Max.   :1.00000   Max.   :1.000000            
    ##  no_of_previous_bookings_not_canceled avg_price_per_room no_of_special_requests
    ##  Min.   :0.000000                     Min.   :0.0000     Min.   :0.0000        
    ##  1st Qu.:0.000000                     1st Qu.:0.1487     1st Qu.:0.0000        
    ##  Median :0.000000                     Median :0.1842     Median :0.0000        
    ##  Mean   :0.002645                     Mean   :0.1915     Mean   :0.1239        
    ##  3rd Qu.:0.000000                     3rd Qu.:0.2222     3rd Qu.:0.2000        
    ##  Max.   :1.000000                     Max.   :1.0000     Max.   :1.0000        
    ##  booking_status  
    ##  Min.   :0.0000  
    ##  1st Qu.:0.0000  
    ##  Median :0.0000  
    ##  Mean   :0.3276  
    ##  3rd Qu.:1.0000  
    ##  Max.   :1.0000

``` r
trainprop <- 0.7
set.seed(12345)
trainrows <- sample(1:nrow(hotel_scaled), trainprop * nrow(hotel_scaled))

hotel_train <- hotel_scaled[trainrows, ]
hotel_test  <- hotel_scaled[-trainrows, ]
```

## Step 4: Build Models

### Step 4.1: Build LogReg Model

Our baseline model will be a **vanilla Logistic Regression (LogReg)**,
which predicts the probability of a booking being canceled.  
This serves as the foundation for comparison with future models (e.g.,
neural networks, decision trees, etc.).

``` r
logreg_model <- glm(booking_status ~ ., data = hotel_train, family = "binomial")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

### Step 4.2: Build KNN Model

Here, we separate our predictors and target variables into training and
testing sets.

``` r
train_features <- subset(hotel_train, select = -booking_status)
train_labels <- hotel_train$booking_status
test_features <- subset(hotel_test, select = -booking_status)
test_labels <- hotel_test$booking_status
```

### Step 4.3: Build ANN Model

We trained a feed-forward artificial neural network (ANN) to predict
which bookings will cancel using the scaled training dataset. The
maximum number of iterations of the optimization algorithm (stepmax) is
set to a high value to ensure the model reaches convergence before it
stops training.

``` r
# ann_model <- neuralnet(
#   booking_status ~ .,
#   data = hotel_train,
#   lifesign = "minimal",
#   stepmax = 1e8,
#   hidden = 4
# )
# plot(ann_model)
```

### Step 4.4: Build Decision Tree Model

We used the C5.0 algorithm from the C50 package. Decision trees are
interpretable models that split the data based on the most informative
features, making them valuable for understanding the decision-making
process behind churn. We trained the model on the full set of predictors
in the training data and then generated booking status probability
predictions for the test set. These probabilities were converted into
binary classifications using a threshold of 0.5l. The model’s
performance was then evaluated using a confusion matrix, helping us
assess how well the tree correctly identified customers who were likely
to cancel.

``` r
m_decision <- C5.0(
  as.factor(booking_status) ~ .,
  data = hotel_train
)
```

### Step 4.5: Build SVM

``` r
hotel_svm <- ksvm(
  as.factor(booking_status) ~ .,
  data = hotel_train,
  kernel = "vanilladot"  # linear kernel
)
```

    ##  Setting default kernel parameters

### Step 4.6: Build Random Forest

``` r
hotel_train$booking_status <- as.factor(hotel_train$booking_status)
hotel_test$booking_status  <- as.factor(hotel_test$booking_status)

rf_model <- randomForest(
  booking_status ~ .,
  data = hotel_train,
  ntree = 500,
  mtry = sqrt(ncol(hotel_train) - 1),
  importance = TRUE
)

plot(rf_model)
```

![](PreliminaryModel_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
varImpPlot(rf_model)
```

![](PreliminaryModel_files/figure-gfm/unnamed-chunk-12-2.png)<!-- -->

## Step 5: Predict

To generate predictions, we applied a 0.5 threshold (standard).

### Step 5.1: Predict LogReg

``` r
logreg_predictions <- predict(logreg_model, newdata = hotel_test, type = "response")
logreg_pred_class <- ifelse(logreg_predictions >= 0.5, 1, 0)
```

### Step 5.2: Predict KNN Model

Let’s figure out the optimal k value. We want to balance sensitivity and
accuracy effectively.

``` r
accuracy_values <- c()
sensitivity_values <- c()
k_values <- seq(1, 31, by = 2)

train_labels <- factor(train_labels, levels = c(0, 1))
test_labels  <- factor(test_labels,  levels = c(0, 1))
for (k in k_values) {
  pred_knn <- knn(train = train_features,
                  test  = test_features,
                  cl    = train_labels,
                  k     = k)
  
  cm <- confusionMatrix(pred_knn, test_labels, positive = "1")
  accuracy_values <- c(accuracy_values, cm$overall["Accuracy"])
  sensitivity_values <- c(sensitivity_values, cm$byClass["Sensitivity"])
}

knn_results <- data.frame(k = k_values,
                          Accuracy = round(accuracy_values, 4),
                          Sensitivity = round(sensitivity_values, 4))
print(knn_results)
```

    ##     k Accuracy Sensitivity
    ## 1   1   0.8499      0.7788
    ## 2   3   0.8416      0.7458
    ## 3   5   0.8459      0.7439
    ## 4   7   0.8420      0.7382
    ## 5   9   0.8420      0.7360
    ## 6  11   0.8386      0.7323
    ## 7  13   0.8394      0.7346
    ## 8  15   0.8376      0.7329
    ## 9  17   0.8375      0.7250
    ## 10 19   0.8339      0.7185
    ## 11 21   0.8328      0.7160
    ## 12 23   0.8295      0.7056
    ## 13 25   0.8291      0.7056
    ## 14 27   0.8276      0.7014
    ## 15 29   0.8277      0.7033
    ## 16 31   0.8276      0.7008

Based on KNN tuning results, the optimal value of k = 5 achieves the
best balance between accuracy (84.5%) and sensitivity (74%). This
configuration minimizes overfitting while maintaining reliable detection
of likely cancellations, making it a practical choice for hotel
overbooking predictions.

Let’s build our final model now.

``` r
pred_knn <- knn(train = train_features, test = test_features, cl = train_labels, k = 5)
```

### Step 5.3: Predict ANN

``` r
# ann_predictions <- predict(ann_model, hotel_test)
# ann_pred_class <- ifelse(ann_predictions >= 0.5, 1, 0)
```

### Step 5.4: Predict Decision Tree

``` r
predict_entropy <- predict(m_decision, hotel_test, type = "prob")[, "1"]
decision_pred <- ifelse(predict_entropy >= 0.5, 1, 0)
```

### Step 5.5: Predict SVM

``` r
hotel_pred_svm <- predict(hotel_svm, hotel_test, type = "response")
svm_pred_class <- ifelse(hotel_pred_svm == "1", 1, 0)
```

### Step 5.6: Predict Random Forest

``` r
rf_prob <- predict(rf_model, hotel_test, type = "prob")
rf_prob_1 <- rf_prob[, "1"]
rf_pred_class <- ifelse(rf_prob_1 >= 0.5, 1, 0)
```

## Step 6: Evaluate

``` r
confusionMatrix(as.factor(logreg_pred_class), as.factor(hotel_test$booking_status), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 6548 1262
    ##          1  782 2291
    ##                                           
    ##                Accuracy : 0.8122          
    ##                  95% CI : (0.8047, 0.8195)
    ##     No Information Rate : 0.6735          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5575          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.6448          
    ##             Specificity : 0.8933          
    ##          Pos Pred Value : 0.7455          
    ##          Neg Pred Value : 0.8384          
    ##              Prevalence : 0.3265          
    ##          Detection Rate : 0.2105          
    ##    Detection Prevalence : 0.2824          
    ##       Balanced Accuracy : 0.7691          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

The baseline logistic regression model achieved an **accuracy of
81.5%**, meaning it correctly classified roughly four out of five
reservations.  
The model’s **sensitivity (65%)** indicates that it successfully
identified about two-thirds of actual cancellations, while its
**specificity (89%)** shows strong performance in recognizing
non-cancellations.  
This balance suggests the model is slightly conservative—it’s better at
avoiding false alarms (predicting cancellations that don’t happen) than
at catching every cancellation.  
For a hotel aiming to minimize overbooking costs, this trade-off is
acceptable for out vanilla model, though future tuning will hopefully
allow us to improve recall in order to maximize profits.

## Step 7: Profit

Using the financial assumptions from **Step 0**, we can estimate the
model’s impact on revenue compared to doing nothing (i.e., no
overbooking policy).

### Confusion Matrix Summary

| Outcome | Count | Description |
|----|----|----|
| **True Positives (TP)** | 2,291 | Predicted “will cancel” and guest did cancel |
| **True Negatives (TN)** | 6,548 | Predicted “will not cancel” and guest did not cancel |
| **False Positives (FP)** | 782 | Predicted “will cancel” but guest showed up |
| **False Negatives (FN)** | 1,262 | Predicted “will not cancel” but guest canceled |

### Financial Impact

| Case   | Profit / Loss per Case | Count | Total Impact |
|--------|------------------------|-------|--------------|
| **TP** | +\$300                 | 2,291 | +\$687,300   |
| **TN** | \$0                    | 6,548 | \$0          |
| **FP** | –\$400                 | 782   | –\$312,800   |
| **FN** | –\$300                 | 1,262 | –\$378,600   |

**Total Expected Profit = \$687,300 – \$312,800 – \$368,600 =
-\$13,100**

### Baseline Comparison (Before Model)

If the hotel does **no overbooking**, every canceled booking represents
a loss of \$300.  
There were **3,553 total cancellations (TP + FN)**, resulting in  
**Baseline Profit = 3,553 × (–\$300) = –\$1,065,900**

### Interpretation

- **Without the model:** expected loss of \$1,065,900  
- **With the model:** expected loss of \$13,100  
- an overall **improvement of about \$1,052,800** in expected revenue on
  the test sample.

Even this simple logistic regression adds meaningful value by
identifying likely cancellations and allowing profitable overbooking.  
Future models with higher recall could further reduce lost revenue and
increase profitability.
