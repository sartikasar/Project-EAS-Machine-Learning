# Install and load necessary packages
install.packages("keras")
library(keras)
library(dplyr)

# Load the dataset
digits <- read.csv("C:/Users/LENOVO/Downloads/Copy of Diabetes.csv")

# Assuming 'categorical_column' is a categorical variable in your dataset
digits <- digits %>%
  mutate(Gender = as.integer(factor(Gender)))

# Remove leading and trailing whitespaces
digits$CLASS <- trimws(digits$CLASS)

# Preprocess the data
X <- as.matrix(digits[, -ncol(digits)])  # Exclude the last column (CLASS)
Y <- as.factor(digits$CLASS)

# Standardize the features
X <- scale(X)


# Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
split_index <- sample(1:nrow(digits), 0.75 * nrow(digits))
x_train <- X[split_index, ]
y_train <- Y[split_index]
x_test <- X[-split_index, ]
y_test <- Y[-split_index]

str(x_train)
str(y_train)

x_train <- as.matrix(x_train)  # Convert to matrix or array
x_test <- as.matrix(x_test)
y_train <- as.integer(factor(y_train))
y_test <- as.integer(factor(y_test))

# Ensure labels are 0-based
y_train <- y_train - 1
y_test <- y_test - 1


# Build the Deep Belief Network (DBN) using keras
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = length(unique(Y)), activation = "softmax")

# Compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Train the model
model %>% fit(
  x_train, y_train,
  epochs = 10,  # You may need to adjust the number of epochs
  batch_size = 32
)

# Evaluate the model on the test set
metrics <- model %>% evaluate(x_test, y_test)
# Print the accuracy
cat("Accuracy of Prediction: ", metrics[["accuracy"]], "\n")

