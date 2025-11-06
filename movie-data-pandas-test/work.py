import pandas as pd
import ast

# load
credits = pd.read_csv("./data/tmdb_5000_credits.csv")
movies = pd.read_csv("./data/tmdb_5000_movies.csv")

# info
# print(movies.info())
# print("*"*40)
# print(credits.info())
# print("*"*40)

# merge
data = pd.merge(movies, credits, left_on="id", right_on="movie_id")

# info
# print(data.info())
# print("*"*40)

# cleaning as needed
# genres
def genreParser(genre):
    try:
        list = ast.literal_eval(genre)
        names = []
        for obj in list:
            names.append(obj['name'])
        return names 
    except(ValueError, SyntaxError):
        return []
# new column genre_list
data["genre_list"] = data["genres"].apply(genreParser)

print(data.info())
print("*"*40)

# analysis

# 1- 10 most common genres??

# explode and count
all_genres = data['genre_list'].explode()
print("Top 10 most common genres:")
print(all_genres.value_counts().head(10))
print("*"*40)

# 2- top 10 most profitable movies?

# create a profit column
data["profit"] = data["revenue"] - data["budget"]

top10 = data[['title_x', 'profit']].sort_values(by='profit', ascending=False)
print(top10.head(10))

# 3- top 10 highest-rated movies (vote_average) that also have at least 1000 votes (vote_count)
print("top 10 highest-rated movies with vote count greater than 1000")
highestRated = data[["title_x", "vote_average", "vote_count"]]
print(highestRated[highestRated["vote_count"] >= 1000].sort_values(by="vote_average", ascending=False).head(10))
