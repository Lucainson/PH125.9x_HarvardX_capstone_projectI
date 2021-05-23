#============================================================================================#
#                       Netflix MovieLens Project | Lucainson RAYMOND                        #
#============================================================================================#



#Importing relevant libraries for our analysis
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(irlba))
suppressPackageStartupMessages(library(recommenderlab))
suppressPackageStartupMessages(library(recosystem))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(RColorBrewer)) 
suppressPackageStartupMessages(library(ggthemes)) 
suppressPackageStartupMessages(library(kableExtra))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(Matrix.utils))
suppressPackageStartupMessages(library(DT))


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, removed)

##################################Exploratory Data Analysis (EDA)###################################

# Data overview
edx%>%
  glimpse()

####Rating histogram
#I create a dataframe "explore_ratings" which contains half star and whole star ratings  from the edx set : 

group <-  ifelse((edx$rating == 1 |edx$rating == 2 | edx$rating == 3 | 
                    edx$rating == 4 | edx$rating == 5) ,
                 "whole_star", 
                 "half_star") 
explore_ratings <- data.frame(edx$rating, group)

ggplot(explore_ratings, aes(x= edx.rating, fill = group)) +
  geom_histogram( binwidth = 0.2) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_fill_manual(values = c("half_star"="#ebd078", "whole_star"="black")) +
  labs(x="Rating", y="Frequency of ratings", caption = "Source: edx set") +
  ggtitle("Frequency of ratings for each rating")




####Genre statistics
top_genres <- edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

top_genres


#####Title bar graph
#Data processing
top_title <- edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(20,count) %>%
  arrange(desc(count))


top_title %>% 
  ggplot(aes(x=reorder(title, count), y=count)) +
  geom_bar(stat='identity', fill="#395341") + 
  coord_flip() +
  labs(x="", y="Number of ratings") +
  labs(title="Top 20 movies title \n based on number of ratings" , caption = "source: edX dataset")



####Number of unique users
edx %>%
  summarize(n_users = n_distinct(userId))

#Plot of number of ratings by userId
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, color="black",
                  fill="#00AFBB") +
  scale_x_log10() + 
  ggtitle("Users") +
  labs(subtitle ="Number of ratings by UserId", 
       x="UserId" , 
       y="Number of ratings", caption = "source: edX dataset") +
  theme(panel.border = element_rect(colour="black", fill=NA))




#####Number of movies
edx %>%
  summarize(n_movies = n_distinct(movieId))

#Plot of number of ratings by movieId
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, color="black",
                  fill="#00AFBB") +
  scale_x_log10() + 
  ggtitle("Movies") +
  labs(subtitle  ="Number of ratings by movieId", 
       x="MovieId" , 
       y="Number of ratings", 
       caption = "source: edX dataset") +
  theme(panel.border = element_rect(colour="black", fill=NA))



####Matrix of User Ratings by Movies
s <- 200 
users <- sample(unique(edx$userId), s)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), s)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:s, 1:s,. , xlab="Movies", ylab="Users")


##############################Timestamps plot

#Average Ratings during week
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : Week")+
  labs(subtitle = "average ratings",
       caption = "source: edX dataset")


#Average Ratings during month
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : Month")+
  labs(subtitle = "average ratings",
       caption = "source: edX dataset")


#Average Ratings during year
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "year")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : Year")+
  labs(subtitle = "average ratings",
       caption = "source: edX dataset")



#######################################Modeling###################################################


#==================================Basic Method=====================================#

#####Model 1: Average movie rating model
mu<-mean(edx$rating)
avg_rmse <- RMSE(validation$rating, mu)
avg_rmse

rmse_table <- data_frame(Model = "Just the average", RMSE = avg_rmse)

rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")



######Model 2: Movie effect model
movie_average <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#Graph
movie_average %>% 
  qplot(b_i, geom ="histogram", bins = 10, data = .,fill= I("black"), color=I("white"),
        ylab = "Number of movies", main = "Number of movies with the computed b_i")


predicted_ratings <- mu +  validation %>%
  left_join(movie_average, by='movieId') %>%
  pull(b_i)
model_rmse_ME <- RMSE(predicted_ratings, validation$rating)
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Movie effect model",  
                                   RMSE = model_rmse_ME))
rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")




######Model 3: Movie and user effect model
user_average<- edx %>% 
  left_join(movie_average, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Graph
user_average%>% 
  qplot(b_u, geom ="histogram", bins = 30, data = ., fill= I("black"), color=I("white"))

user_average <- edx %>%
  left_join(movie_average, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#We can now construct predictors and see RMSE improves
predicted_ratings <- validation%>%
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_rmse_MUE <- RMSE(predicted_ratings, validation$rating)
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Movie and user effect model",  
                                   RMSE = model_rmse_MUE))
rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")





#######Model 4: Regularized Movie Effect Model
#Splitting the data into 5 parts
set.seed(1996)

cv_splits <- caret::createFolds(edx$rating, k=5, returnTrain =TRUE)
lambdas <- seq(0, 5, 0.1)
rmses <- matrix(nrow=5,ncol=length(lambdas))

#Perform 5-fold cross validation to determine the optimal lambda
for(k in 1:5) {
  train_set <- edx[cv_splits[[k]],]
  test_set <- edx[-cv_splits[[k]],]
  
  test_final <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  removed <- anti_join(test_set, test_final)
  train_final <- rbind(train_set, removed)
  
  mu <- mean(train_final$rating)
  just_the_sum <- train_final %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - mu), n_i = n())
  
  rmses[k,] <- sapply(lambdas, function(l){
    predicted_ratings <- test_final %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = mu + b_i) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses_reg1 <- colMeans(rmses)
qplot(lambdas,rmses_reg1)
lambda <- lambdas[which.min(rmses_reg1)]
lambda

#Using the optimized lambda, we can now perform prediction and evaluate the RMSE in the validation set
mu <- mean(edx$rating)
movie_reg_average <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
predicted_ratings_5 <- validation %>% 
  left_join(movie_reg_average, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
model_rmse_reg1 <- RMSE(predicted_ratings_5, validation$rating)
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Regularized Movie Effect Model",  
                                   RMSE = model_rmse_reg1))
rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")




########Model 5: Regularized Movie & User Effect Model (Version 1)
lambdas <- seq(0, 5, 0.1)
rmses_reg <- matrix(nrow=5,ncol=length(lambdas))
#Performing 5-fold cross validation to determine the optimal lambda
for(k in 1:5) {
  train_set <- edx[cv_splits[[k]],]
  test_set <- edx[-cv_splits[[k]],]
  
  test_final <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  removed <- anti_join(test_set, test_final)
  train_final <- rbind(train_set, removed)
  
  mu <- mean(train_final$rating)
  
  rmses_reg[k,] <- sapply(lambdas, function(l){
    b_i <- train_final %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_final %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <- 
      test_final %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses_reg
rmses_reg2 <- colMeans(rmses_reg)
rmses_reg2
qplot(lambdas,rmses_reg2)
lambda <- lambdas[which.min(rmses_reg2)]
lambda

#Now we use this parameter lambda to predict the validation dataset and evaluate the RMSE
mu <- mean(edx$rating)
b_i_reg1 <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u_reg1 <- edx %>% 
  left_join(b_i_reg1, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings_reg2 <- 
  validation %>% 
  left_join(b_i_reg1, by = "movieId") %>%
  left_join(b_u_reg1, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_rmse_reg2 <- RMSE(predicted_ratings_reg2, validation$rating)
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Regularized Movie & User Effect Model (Version 1)",  
                                   RMSE = model_rmse_reg2))
rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")





########Model 6: Regularized Movie & User Effect Model (Version 2)
lambda_i <- 2.2
lambdas_u <- seq(0, 5, 0.1)
rmses_op <- matrix(nrow=5,ncol=length(lambdas_u))

#Performing 5-fold cross validation to determine the optimal lambda
for(k in 1:5) {
  train_set <- edx[cv_splits[[k]],]
  test_set <- edx[-cv_splits[[k]],]
  
  test_final <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  removed <- anti_join(test_set, test_final)
  train_final <- rbind(train_set, removed)
  
  mu <- mean(train_final$rating)
  
  rmses_op[k,] <- sapply(lambdas_u, function(l){
    b_i <- train_final %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+lambda_i))
    b_u <- train_final %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <- 
      test_final %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses_op
rmses_reg3 <- colMeans(rmses_op)
rmses_reg3
qplot(lambdas_u,rmses_reg3)
lambda_u <-lambdas_u[which.min(rmses_reg3)]
lambda_u 


#Using the lambda_i (fixed lambda) and lambda_u we determined, 
#we generated the prediction model and evaluated the RMSE on the validation set.

lambda_i <- 2.2
lambda_u <- 5
mu <- mean(edx$rating)
b_i_reg2 <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda_i))
b_u_reg2 <- edx %>% 
  left_join(b_i_reg2, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))
predicted_ratings_reg3 <- 
  validation %>% 
  left_join(b_i_reg2, by = "movieId") %>%
  left_join(b_u_reg2, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_rmse_reg3 <- RMSE(predicted_ratings_reg3, validation$rating)
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Regularized Movie & User Effect Model (Version 2)",  
                                   RMSE = model_rmse_reg3 ))
rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")


#=============================Advanced Method======================================#

#We calculate the residuals of the best baseline model (model 5)
edx_residual <- edx %>% 
  left_join(b_i_reg1, by = "movieId") %>%
  left_join(b_u_reg1, by = "userId") %>%
  mutate(residual = rating - mu - b_i - b_u) %>%
  select(userId, movieId, residual)
head(edx_residual)


#Performing the matrix factorization process
#Matrix format
edx_for_mf <- as.matrix(edx_residual)
validation_for_mf <- validation %>% 
  select(userId, movieId, rating)
validation_for_mf <- as.matrix(validation_for_mf)

#Writing edx_for_mf and validation_for_mf tables on disk
write.table(edx_for_mf , file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(validation_for_mf, file = "validset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

#Using data_file() function to specify a data set from a file in the hard disk.
set.seed(1996) 
train_set <- data_file("trainset.txt")
valid_set <- data_file("validset.txt")

#Building a recommender object
r <-Reco()

#Tuning training set
opts <- r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                      costp_l1 = 0, costq_l1 = 0,
                                      nthread = 4, niter = 10))
opts


#Training the recommender model
r$train(train_set, opts = c(opts$min, nthread = 4, niter = 20))

# Making prediction on validation set and calculating RMSE:
pred_file <- tempfile()
r$predict(valid_set, out_file(pred_file))  
predicted_residuals_mtx_fact <- scan(pred_file)
predicted_ratings_mtx_fact <- predicted_ratings_reg2 + predicted_residuals_mtx_fact
rmse_mtx_fact <- RMSE(predicted_ratings_mtx_fact,validation$rating)

#RMSE results
rmse_table <- bind_rows(rmse_table,
                        data_frame(Model="Matrix Factorization",  
                                   RMSE = rmse_mtx_fact))

rmse_table %>%
  knitr::kable()%>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="black")


