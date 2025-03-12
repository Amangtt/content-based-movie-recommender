import pandas as pd
import matplotlib.pyplot as plt

class analysis:
        def __init__(self,user,movie,y_train,genre,list,logger):
             self.user=user
             self.movie=movie
             self.y_train=y_train
             self.genre=genre
             self.list=list
             self.logger=logger

        def bygenre(self):
            self.logger.info('classify by genre')
            return self.genre
 

        def rating(self):
            try:
                plt.hist(self.y_train, bins=20)
                plt.title('counts of each rating')
                plt.xlabel("Ratings")
                plt.ylabel("Count")
                plt.show()
                self.logger.info('sucessfully plotted ratings')

            except Exception as e:
                error_message = f"Failed to group and plot rating: {e}"
                self.logger.error(error_message)
  

        def no_of_user_mov(self):
            try:
                count_u=self.user['user_id'].nunique()
                count_m=self.movie['movie_id'].nunique()
                print(f"Number of users: {count_u}")
                print(f"Number of movies: {count_m}")
                self.logger.info('sucessfully calculated number of users and movies')
            except Exception as e:
                error_message = f"Failed to calculate number of users and movies: {e}"
                self.logger.error(error_message)
        def most_watched(self):
            # Count how many times each movie_id appears (number of ratings per movie)
            mov = self.movie['movie_id'].value_counts().reset_index()
    
            # Rename columns for clarity
            mov.columns = ['movie_id', 'count']
            top_10=mov.head(10)
            filtered = self.list[self.list['movieId'].isin(top_10['movie_id'])]
                # Preserve order by using Categorical sorting
            filtered['movieId'] = pd.Categorical(filtered['movieId'], categories=top_10['movie_id'], ordered=True)
            filtered = filtered.sort_values('movieId')
            return filtered 



       
