<h1>Movie rating prediction</h1>

To use this model, just run the main.py and it will ask you to input n ratings for n random movie names, then predicting your behaviors on the rest movies in a particular movie set
<br>
You can change the data in the 'data' file to personalize your choices and predictions and change the iteration(I set 200 times) in main.py
<br>
<p>A few notes for the data file:</p>
<br>
  <p>Y suffix stands for existing ratings for i movies and j users(i * j matrix, row for index of movies, j for index of users) </p>
  <p>X suffix stands for the initialized features of movies (you can customize or just randomize due to your decision, from 0-1 e.g. 0 for non-violence included and 1 for very violent)</p>
  <p>R suffix stands for whether jth user has rated ith movie</p>
  <p>W suffix stands for the predicted preference of each user on X and b stands for each user's bias, matrix multiplication of w and x plus b gives you the predicted rating for a user</p>

