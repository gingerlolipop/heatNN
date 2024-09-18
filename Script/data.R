data_path <- "C:/Users/jing007.stu/Documents/Project Data/Heat Data/data raw/267694abe82f13224bad1f62f4c5adce/data_0.nc"

install.packages("terra")

library(terra)
nc_data <- rast(data_path)

# 1. Get basic metadata information
print(nc_data)

# 2. Inspect the names of layers and their corresponding variable
layer_names <- names(nc_data)
head(layer_names)  # View the first few layer names
length(layer_names) #2880
tail(layer_names)

# 3. Check the variables and associated metadata
variables <- sources(nc_data)
print(variables)  # variables exist in the NetCDF file: t2m, u10, v10, tp

# 4. Check how time is structured (if available)
time_values <- time(nc_data)  # Extract time information, if present
print(head(time_values))  # Display time values (timestamps)

# 5. Visualize a few layers to understand how the data evolves over time
plot(nc_data[[1:6]])  # Plot the first six layers to visualize

# 7. Check CRS (Coordinate Reference System)
crs(nc_data)  # This will confirm the coordinate system being used

# 8. Get a subset of values to explore their structure
values_subset <- values(nc_data[[1:3]], na.rm = TRUE)  # Extract values from the first three layers without NaN
head(values_subset)  # Preview the first few extracted values

# 9. Extract a specific variable (e.g., temperature "t2m") and time slice (e.g., first hour of June 1961)
t2m_layer <- nc_data[[grep("t2m", layer_names)]]
plot(t2m_layer[[1]])  # Plot the first time slice of temperature data