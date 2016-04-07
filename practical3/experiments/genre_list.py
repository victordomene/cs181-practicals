# generate list of genres expressed in the training data.
import csv
import gzip
import musicbrainzngs

if __name__ == "__main__":
  musicbrainzngs.set_useragent("cs181", "1.0", "harvard@harvard.edu")
  musicbrainzngs.auth("jvh181", "finale1")

  with gzip.open('../data/artists.csv.gz') as artists_fh:
    artists_csv = csv.reader(artists_fh, delimiter=',', quotechar='"')
    genres = {}
    next(artists_csv, None)
    for row in artists_csv:
      artist_id = row[0]
      artist_name = row[1]

      print artist_id

      result = musicbrainzngs.get_artist_by_id(artist_id, includes=["tags", "user-tags"])
      if 'tag-list' in result['artist']:
        for tag in result['artist']['tag-list']:
          if not tag['name'] in genres:
            genres[tag['name']] = {}
      else:
        print "tag-list not found in artist {0}".format(result['artist']['name'])

    with open('genre_list.txt', 'w') as results:
      results.write(','.join(genres.keys()))
