import pandas as pd

def merge_data(movies, user):
    merged_data=pd.merge(movies,user,on='movieId', how='outer')
    merged_data= merged_data.drop('timestamp',axis=1)
    merged_data.to_csv('./Data/user_data.csv', index=False)


def final_user(df):
    # Explode genres into separate rows
    df_exploded = df.assign(genre=df['genres'].str.split('|')).explode('genre')
    
    # Drop unnecessary columns
    df_exploded = df_exploded.drop(columns=['genres', 'title'])
    
    # Compute rating count & rating average per user
    user_stats = df.groupby('userId')['rating'].agg(['count', 'mean']).reset_index()
    user_stats.rename(columns={'count': 'rating_count', 'mean': 'rating_ave'}, inplace=True)

    # Pivot table to get the average rating per genre for each user
    genre_ratings = df_exploded.pivot_table(index='userId', columns="genre", values="rating", aggfunc='mean').fillna(0)
    genre_ratings.reset_index(inplace=True)

    # Merge with user stats
    final_df = user_stats.merge(genre_ratings, on='userId', how='left')

    # Expand to match the original number of user-movie ratings
    expanded_df = df[['userId', 'movieId']].merge(final_df, on='userId', how='left')

    # Save to CSV
    expanded_df.to_csv('./Data/content_user_train.csv', index=False)




def final_movies(df):
    # Aggregate the most common title and compute the average rating
    df_edited = df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        title=('title', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        genres=('genres', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    ).reset_index()

    # Split genres into separate rows
    df_exploded = df_edited.assign(genres=df_edited['genres'].str.split('|')).explode('genres')

    # Get unique genre list for one-hot encoding
    unique_genres = sorted(set('|'.join(df_edited['genres']).split('|')))

    # One-hot encode genres
    for genre in unique_genres:
        df_edited[genre] = df_edited['genres'].apply(lambda x: 1 if genre in x else 0)

    # Extract movie release year (assuming year is in the title like "Movie (1999)")
    df_edited['year'] = df_edited['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    
    df_edited['avg_rating']=df_edited['avg_rating'].round(1)
    # Select final columns
    final_columns = ['movieId', 'year', 'avg_rating'] + unique_genres
    df_final = df_edited[final_columns]
    df_final=df_final.drop(columns='(no genres listed)')
    print(df_final.isnull().sum())
    df_final.to_csv('./Data/content_item_train.csv', index=False)



