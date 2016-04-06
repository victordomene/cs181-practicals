# Random Forest

One random forest per artist. User features:

* Artist bias (average # of plays for this artist);
* Global bias (average # of plays for all users);
* Sex;
* Age;
* Gender;
* Country;
* Total number of plays for each artist (yes, this is a lot of stuff).

If there is no information on number of plays for each artist, all information we have is on sex, age, gender and country.
Also, we can use the global median here, in case it has bad predictions for these cases.