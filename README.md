# genetic_algorithm

Inspired by David R Miller's work
https://github.com/davidrmiller/biosim4


-------------------
requirements
Python 3.9.12
conda 4.13.0

Presented framework is my interpretation of biosim4 prepared by David R Miller and most of assumption come from his work. Here I'll only mention general rules Great explanation is provided by D.R. Miller's video https://www.youtube.com/watch?v=N3tRFayqVtk. 

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
        + output4 - randomly 
2) in this experiment only asexual reproduction and punctual mutation are implemented

Script is working and is easy to run, but there are still a lot things to change/ fix

1) improve main loop performance
2) add sexual 



