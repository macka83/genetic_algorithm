# genetic_algorithm

Inspired by David R Miller's work
https://github.com/davidrmiller/biosim4


-------------------
requirements  
Python 3.9.12  
conda 4.13.0  

Presented framework is my interpretation of biosim4 prepared by David R Miller and most of assumption come from his work. Great explanation is provided by D.R. Miller's video https://www.youtube.com/watch?v=N3tRFayqVtk. 

Here I'll only highlight changes, obstacles and things to do

1) Creating initial population is almost the same as original except input and output neurons number to simplify calculations.
    * input neurons sensitive to  
        + input0 - close obstacle 1 point ahead)  
        + input1 - distant obstacle (up to 5 points ahead)  
    * output neurons move  
        + output0 - north  
        + output1 - south  
        + output2 - east  
        + output3 - west  
        + output4 - random movement 
2) in this experiment only asexual reproduction and punctual mutation are implemented

TODO:
1) ~~improve main loop performance and increase ability to process larger population~~ - increase speed by replacing recursive loop prevent_overlap_movement() from steps_in_generation() with for loop
2) add config file
3) split code into smaller chunks
4) add sexual reproducion
5) early stopper
6) summary statistic
7) add images to readme

CHECK
1) if small population give biased result in comparison to larger population
2) if mutation works fine
3) why initial population move mostly north-east
