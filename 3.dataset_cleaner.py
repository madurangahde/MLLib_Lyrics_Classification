import pandas as pd

print("Loading datasets...")
mendaly_musi = pd.read_csv('./data/tcc_ceds_music.csv')
rock_music = pd.read_csv('./data/rock_student_dataset.csv')
print(" Datasets loaded") 


df = mendaly_musi[["artist_name", "track_name", "release_date", "genre", "lyrics"]]
print("Shape after column selection:", df.shape)


df = df.dropna(subset=['release_date'])
print("Shape after dropna:", df.shape)

print("Processing rock_music dataset...")
rock_music['release_date'] = pd.to_datetime(rock_music['release_date'], errors='coerce')

invalid_dates = rock_music[rock_music['release_date'].isna()]
print("Invalid dates found:", len(invalid_dates))

rock_music = rock_music.dropna(subset=['release_date'])
print("Shape after removing invalid dates:", rock_music.shape)

rock_music['release_date'] = rock_music['release_date'].dt.year.astype('int64')
print("Converted release_date to year")

rock_music['genre'] = rock_music['genre'].replace(['retro'], 'country')
print("Replaced 'retro' genre")

print("\nSample rock_music data:")
print(rock_music.head())

print("Saving cleaned rock_music dataset...")
rock_music.to_csv('./data/Student_dataset.csv', index=False)
print(" Saved Student_dataset.csv")



print("Combining datasets...")
combined_df = pd.concat([df, rock_music], ignore_index=True)
print("Shape of the combined df:", combined_df.shape)

print("\n🔹 Cleaning text columns...")
for col in ["artist_name", "track_name", "lyrics"]:
    combined_df[col] = combined_df[col].str.replace('\n', ' ')
    combined_df[col] = combined_df[col].str.replace('\t', ' ')
    combined_df[col] = combined_df[col].str.replace(',', ' ')
    combined_df[col] = combined_df[col].str.replace(r'\s+', ' ', regex=True)

print(" Text cleaning done")

print(" Saving final merged dataset...")
combined_df.to_csv('./data/Merged_dataset.csv', index=False)
print("Saved Merged_dataset.csv")

print(" All processing completed successfully!")