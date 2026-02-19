library(dplyr)
library(tidyr)
library(readr)
library(stringr)
# Set Seed
set.seed(123)

# FMA Dataset
# https://github.com/mdeff/fma
# Pull Data
temp <- paste(tempfile(), ".zip", sep = "")
options(timeout = 60 * 10)
download.file("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", temp)

# Feature Data
# Consolidate multiline header
echonest_colnames <- unz(temp, "fma_metadata/echonest.csv") %>%
  read_csv(n_max = 0, skip = 2) %>%
  rename(track_ID = "...1") %>%
  names()
# Read the data
echonest_raw <- unz(temp, "fma_metadata/echonest.csv") %>%
  read_csv(skip = 4, col_names = echonest_colnames) %>%
  # Transform track_ID to integer for tibble merging
  mutate(track_ID = as.integer(track_ID))

# Remove temporal features
echonest <- echonest_raw[, c(1:26)]

# Metadata
# Consolidate multiline header
metadata_colnames_a <- unz(temp, "fma_metadata/tracks.csv") %>%
  read_csv(n_max = 0, skip = 1) %>%
  rename(track_ID = `...1`) %>%
  names() %>%
  # Removing strange encoding
  sub("\\...*", "", .)

metadata_colnames_b <- unz(temp, "fma_metadata/tracks.csv") %>%
  read_csv(n_max = 0) %>%
  names() %>%
  # Removing strange encoding
  sub("\\...*", "", .)

metadata_colnames <- paste(metadata_colnames_b, metadata_colnames_a, sep = "_")

# Read the data
metadata_raw <- unz(temp, "fma_metadata/tracks.csv") %>%
  read_csv(skip = 3, col_names = metadata_colnames) %>%
  rename(track_ID = `_track_ID`) %>%
  # Transform track_ID to integer for tibble merging
  mutate(track_ID = as.integer(track_ID))
# Combine the data and metadata
data <- inner_join(metadata_raw, echonest, by = "track_ID")

# Read in the genres.csv file
genres <- unz(temp, "fma_metadata/genres.csv") %>%
  read_csv()

# Clean up downloaded files
unlink(temp)

# Tidy Data
df_tidy <- data %>%
  select(c("track_ID", "artist_id", "artist_name.x", 
           "track_duration", "track_genre_top", "track_genres",
           "track_listens", "acousticness", "danceability", "energy", 
           "instrumentalness", "liveness", "speechiness", "tempo", "valence")
         )

# Adding names to track_genres based on genres dataset
df_tidy <- df_tidy %>%
  # Separate `track_genres` into rows (one genre per row) by removing brackets and splitting by comma
  mutate(track_genres = str_remove_all(track_genres, "\\[|\\]")) %>%
  separate_rows(track_genres, sep = ",") %>%
  mutate(track_genres = as.integer(track_genres)) %>%
  # Join with genres.csv to get genre names
  left_join(genres, by = c("track_genres" = "genre_id")) %>%
  # Group back by track_ID and collapse genre names into a single string
  group_by(track_ID) %>%
  mutate(track_genres_named = str_c(title, collapse = ", ")) %>%
  ungroup() %>%
  # Select relevant columns and drop duplicates
  select(-title, -track_genres, -parent, -top_level, -`#tracks`) %>%
  distinct() %>%
  # Dropping rows with missing genre
  drop_na(., track_genre_top)

# Write .RData
save(df_tidy, file = "df_tidy.RData")
