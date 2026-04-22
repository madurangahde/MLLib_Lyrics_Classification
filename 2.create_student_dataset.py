import lyricsgenius
import pandas as pd

# 1. Setup API Connection
# Replace 'YOUR_ACCESS_TOKEN' with your actual Genius token
token = "EL1w2HokC2RINndXmwhSvXYkkpOo9ETQM3b-u1u1gc5WSZvAJSih9t7AwxueVCD3"
genre = "Alternative Rock"  # Genre to search for
genius = lyricsgenius.Genius(token)

# Optional: Clean up the data as it's being pulled
genius.remove_section_headers = True  # Removes [Chorus], [Verse], etc.
genius.skip_non_songs = True  # Skips interviews and tracklists


def get_genre_song_data(count=100):
    songs_data = []
    page = 1

    print(f"Starting search for {count} {genre} songs...")

    # We search using a tag/keyword and iterate until we hit the count
    while len(songs_data) < count:
        # Searching for '$genre' returns songs tagged with the genre
        res = genius.search_songs(genre, per_page=50, page=page)

        for hit in res["hits"]:
            if len(songs_data) >= count:
                break

            song_id = hit["result"]["id"]
            try:
                # Fetch full song object to get lyrics and release date
                song = genius.song(song_id)["song"]

                # Extract specific fields
                song_info = {
                    "artist_name": song.get("primary_artist", {}).get("name"),
                    "track_name": song.get("title"),
                    "release_date": song.get("release_date", "Unknown"),
                    "genre": genre,
                    "lyrics": genius.lyrics(song_id),
                }

                songs_data.append(song_info)
                print(f"Fetched [{len(songs_data)}/{count}]: {song_info['track_name']}")

            except Exception as e:
                print(f"Skipping song due to error: {e}")
                continue

        page += 1  # Move to next page of results

    return songs_data


# 2. Execute and Save
data = get_genre_song_data(100)

# 3. Convert to DataFrame for easy viewing/saving
df = pd.DataFrame(data)
df.to_csv("Student_dataset.csv", index=False)

print(f"\nProcessing complete. Data saved to 'Student_dataset.csv'.")
