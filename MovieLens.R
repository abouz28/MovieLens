if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

### Exploratory Data Analysis

str(edx)

n_distinct(edx$movieId)

n_distinct(edx$userId)

n_distinct(edx$genres)

library(stringr)
edx$year <- str_extract(edx$title, "\\((\\d{4})\\)")
edx$year <- as.numeric(gsub("\\D", "", edx$year))
range(edx$year)

genre_count = edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(Count = n()) %>%
  arrange(desc(Count)) 

unique(genre_count$genres)
genre_count %>% slice_head(n = 5)

genre_count = data.frame(Genres = c("Drama","Comedy","Action","Thriller","Adventure"),
                         Count = c(3910127,3540930,2560545,2325899,1908892))

genre_count <- genre_count[order(-genre_count$Count), ]

ggplot(genre_count, aes(x = reorder(Genres, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Genre Counts",
       x = "Genres",
       y = "Count") +
  theme_minimal() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = Count), vjust = -0.2, size = 3)

average_rating_per_genre <- edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(AverageRating = mean(rating, na.rm = TRUE))

average_rating_per_genre %>% arrange(desc(AverageRating)) %>%  filter(genres != "IMAX" & genres != "(no genres listed)") %>% slice_head(n = 5)

genre_rating_count = data.frame(Genres = c("Film-Noir","Documentary","War","Mystery","Drama"),
                                AverageRating = c(4.011625,3.783487,3.780813,3.677001,3.673131))

genre_rating_count <- genre_rating_count[order(-genre_rating_count$AverageRating), ]

ggplot(genre_rating_count, aes(x = reorder(Genres, -AverageRating), y = AverageRating)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Genre based on average rating",
       x = "Genres",
       y = "AverageRating") +
  theme_minimal() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = AverageRating), vjust = -0.2, size = 3)

top_10_titles <- edx %>%
  group_by(title) %>%
  summarize(Count = n()) %>%
  arrange(desc(Count)) %>%
  head(10)

ggplot(top_10_titles, aes(x = reorder(title, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Top 10 Titles",
       x = "Titles",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = Count), vjust = -0.1, size = 4,font="bold")

ggplot(edx, aes(x = rating)) +
  geom_bar(stat = "count", fill = "blue") +
  labs(title = "Rating Distribution",
       x = "Rating",
       y = "Count") +
  theme_minimal()
```

People prefer to rate movies with a max rating of 4

### Data Modelling

```{r}
set.seed(123) 
train_indices <- sample(1:nrow(edx), 0.8 * nrow(edx))
train_data <- edx[train_indices, ]
test_data <- edx[-train_indices, ]
lambdas <- seq(0, 10, 1)
rmses <- sapply(lambdas, function(l) {
  
  mu <- mean(edx$rating)
  
  b_i <- train_data %>%
    group_by(movieId) %>%
    dplyr::summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- train_data %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    dplyr::summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  predicted_ratings <-
    test_data %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% .$pred
  
  return(sqrt(mean((predicted_ratings- test_data$rating)^2,na.rm = TRUE)))
})

min(rmses)  # for test data
lambdas[which.min(rmses)]
```

### RMSE for final_holdout_test with optimal lambda = 5
```{r}
mu <- mean(edx$rating)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + 5))

b_u <- edx %>%
  left_join(b_i, by='movieId') %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() +5))

predicted_ratings <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i +  b_u) %>% .$pred

Rmse = sqrt(mean((predicted_ratings- final_holdout_test$rating)^2,na.rm = TRUE))
Rmse